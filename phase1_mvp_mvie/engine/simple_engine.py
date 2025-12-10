import torch
from engine.model_loader import ModelLoader
from engine.tokenizer import Tokenizer

class SimpleEngine:
    def __init__(self,
                 model_loader:ModelLoader,
                 tokenizer:Tokenizer,
                 max_new_tokens :int = 50
                 ):
        self.model = model_loader.get_model()
        self.tokenizer = tokenizer
        self.device = self.model.device # 与loader的device保持一致
        self.max_new_tokens = max_new_tokens
        
        
    @torch.no_grad()
    def generate(self,prompt:str):
        # -----------------------------
        # 1. Tokenize
        # -----------------------------
        input_ids = self.tokenizer.encode(prompt).to(self.device)# (seq_len,)
        input_ids_batch_1 = input_ids.unsqueeze(0)  #  → (1, seq_len)
        # -----------------------------        
        # 2. Prefill: 建立 KV Cache
        # -----------------------------
        outputs = self.model(
            input_ids_batch_1,
            use_cache=True
        )
        past_key_values = outputs.past_key_values
        #把prompt中的kv存起来 后期只需要连接concatenate就可以了
        
        # flatten existing tokens
        generated_ids = input_ids.tolist()

        # 取第一个 next token
        #  Batch，sequence_size, vocab_size
        #  此处batch是1 因为tokenizer 的机制 每次只有一个prompt
        # 此处要求的是最后一个token的概率分布 所以是-1
        # 此处的dim 是qwen2的vocab size 所以大概是15k
        # dim = -1 代表的是在最后一个维度做argmax 因为取了最后一个token在[:, -1, :]
        # 所以此时的维度变成了(1, 152064) 且在 vocab上做argmax 所以是（1，1） Tensor（[一个int]）
        # generated id 要把这个id加进去 所以就变了这个 item（） 将这个变成int append到list里
        next_token_id = torch.argmax(outputs.logits[:, -1, :], dim=-1).item()
        generated_ids.append(next_token_id)

        # -----------------------------
        # 3. Decode Loop
        # -----------------------------
        
        #已经算了一个token了
        for _ in range(self.max_new_tokens -1):

            # 把最新的token放进去再算
            next_input = torch.tensor(
                                      [[next_token_id]],
                                      device=self.device,
                                      dtype=torch.long,   # 很重要，必须是 long，才能拿去做 embedding
                                    )
            
            outputs = self.model(
              next_input,
              past_key_values = past_key_values,
              use_cache = True
            )
            
            past_key_values = outputs.past_key_values
            
            next_token_id = torch.argmax(outputs.logits[:, -1, :], dim=-1).item()
            generated_ids.append(next_token_id)
            
        # -----------------------------
        # 4. Decode text
        # -----------------------------
        return self.tokenizer.decode(generated_ids)