'''
Docstring for engine.executor
12-18 改动
将prompt 从原来的engine逻辑 转移到request里
kvcache 储存在kvcahe的python里

max token 从request中取
'''
import torch
from engine.model_loader import ModelLoader
from engine.tokenizer import Tokenizer
from engine.request import Request
from engine.kv_cache import KVCache

class Executor:
    """
    Phase2:
    只支持单请求、贪心解码。
    后面 scheduler 进来时只需要在外面把一堆 Request 丢进来即可。
    
    Phase3.2:
    增加 stop条件 (eos/stop_token_ids)就提前停止
    """
    def __init__(self,
                 model_loader:ModelLoader,
                 tokenizer:Tokenizer
                 ):
        self.model = model_loader.get_model()
        self.tokenizer = tokenizer
        self.device = self.model.device # 与loader的device保持一致
    
    def _resolve_eos_token_id(self,request:Request):
        #优先从Request 上 显式指定 然后再从tokenizer取
        if getattr(request,"eos_token_id",None) is not None:
            return request.eos_token_id
        return getattr(self.tokenizer,"eos_token_id",None)
    
    def _should_stop(self, request:Request, token_id:int) -> bool:
        #优先stop_token_id 
        if getattr(request,"stop_token_ids",None) is not None:
            if token_id in request.stop_token_ids:
                return True
        
        #然后再看stop_on_eos (默认True)
        if getattr(request, "stop_on_eos", True):
            eos_id = self._resolve_eos_token_id(request)
            if eos_id is not None and token_id ==eos_id:
                return True
        
        return False
    
    
    
    @torch.no_grad()
    def _prefill_step(self,
                    request:Request):
        """
        Docstring for prefil_step
        输入：
        1. request :Request  单个 Request 
        
        返回:
        1. past_key_values          传递给decode阶段 需要过去的键值对
#       2. generated_ids: list[int]  用来包括已经生成的token的ids
#       3. next_token_id: int       第一个预测出来的新token的 int值      
  
        目的：
        1.将原本cpu上的input_id放入device
        2.转换单个 prompt (seq_len) -> (1,seq_len)  
        3. 使用model的cache 版本返回 past加new _token 的output的kv对
        4. 将目前的input_ids变成list[int] 用于最后decode
        

        """
        #此处是prefil step 把prompt转换的input_ids放进去device  目前是shape是（seq_len）
        # model 的__call__ 需要的是 （batchsize，seq_len）

        # # 输入的prompt再tokenizer encode后的(seq_len) 单个prompt的ids
        input_ids = self.tokenizer.encode(request.prompt).to(self.device)# (seq_len,)
        request.input_ids = input_ids
        
        input_id_batch_1 = input_ids.unsqueeze(0)#  → (1, seq_len)
        
        outputs = self.model(
                            input_id_batch_1,
                            use_cache=True
        )
                
        kv_cache = KVCache(outputs.past_key_values)
        # flatten existing tokens        
        generated_ids = input_ids.tolist()
        next_token_id = torch.argmax(outputs.logits[:,-1,:],dim=-1).item()        
        
        # 取第一个 next token
        #  Batch，sequence_size, vocab_size
        #  此处batch是1 因为tokenizer 的机制 每次只有一个prompt
        # 此处要求的是最后一个token的概率分布 所以是-1
        # 此处的dim 是qwen2的vocab size 所以大概是15k
        # dim = -1 代表的是在最后一个维度做argmax 因为取了最后一个token在[:, -1, :]
        # 所以此时的维度变成了(1, 152064) 且在 vocab上做argmax 所以是（1，1） Tensor（[一个int]）
        # generated id 要把这个id加进去 所以就变了这个 item（） 将这个tensor变成int append到list里
        
        generated_ids.append(next_token_id)
        
        # 原本返回的kv 和 ids 放入了request里 逻辑变成了只是返回next_token_id 
        request.generated_ids = generated_ids
        request.kv_cache = kv_cache
        
        return  next_token_id

    @torch.no_grad()
    def _decode_step(self,
               next_token_id: int,
               kv_cache: KVCache):
        """
        Docstring for decode_step
        输入: 1. next_token_id : int 由前一步生成的token的id
              2. kv_cache     上一步生成的键值对 
        输出: 

        单步 decode:
        - 构造 (1,1) 的 next_input
        - 带着 past_key_values 进入 model
        - 返回新的 next_token_id 和新的 pkv

        """
                
        next_input = torch.tensor(
                        [[next_token_id]],
                        device=self.device,
                        dtype=torch.long,   # 很重要，必须是 long，才能拿去做 embedding
                        )
        
        outputs = self.model(
            next_input,
            past_key_values = kv_cache.data,
            use_cache = True
        )

        
        kv_cache.data = outputs.past_key_values
        next_token_id = torch.argmax(outputs.logits[:, -1, :], dim=-1).item()

        return next_token_id
    
    @torch.no_grad()
    def run(self,request: Request)->str:
    
        # 1. Prefill: 建立 KV Cache
        next_token_id = self._prefill_step(request)

        # 2. 如果prefill就 触发了stop（eos或者stop token id）立马结束
        if self._should_stop(request= request,token_id= next_token_id):
            request.finished = True
            return self.tokenizer.decode(request.generated_ids)

        # 3. Decode Loop
        for _ in range(request.max_new_tokens - 1):
            next_token_id = self._decode_step(next_token_id=next_token_id, kv_cache=request.kv_cache)
            request.generated_ids.append(next_token_id)
        
            if self._should_stop(request= request, token_id= next_token_id):
                break
        
        request.finished = True
        
        # 4. Decode text
        return self.tokenizer.decode(request.generated_ids)