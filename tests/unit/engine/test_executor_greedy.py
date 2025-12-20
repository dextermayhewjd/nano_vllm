'''
Docstring for tests.unit.engine.test_executor_greedy

首先得要固定tokenizer 的输出
否则永远测试不到

其次tokenizer生成的

LLM model 必须要要能够做到每次生成的token都是可预测的
并且能够区分出prefill和decode阶段

其次还要模拟kvcache 

Executor 依赖三样东西：
model_loader.get_model()
tokenizer.encode / decode
model(input_ids, past_key_values)


'''
from engine.executor import Executor
from engine.request import Request
import torch
from types import SimpleNamespace


class DummyTokenizer:
    '''
    要能够encode和decode
    '''
    def __init__(self,eos_token_id: int = 9):
        self.eos_token_id = eos_token_id
        
    # 返回tensor(seq_len,) 这里固定返回 torch.tensor([1,2,3])
    def encode(self,prompt: str)-> torch.LongTensor:
        return torch.tensor([1,2,3], dtype=torch.long)
    # 
    def decode(self,token_ids)->str:
         return " ".join(str(x) for x in token_ids)
       

class DummyCausalLM:
    '''
    因为只在两处调用model
    1. prefill阶段
    outputs = self.model(input_id_batch_1, use_cache=True)
    next_token_id = argmax(outputs.logits[:, -1, :])
    
    2. decode阶段 
    outputs = self.model(next_input, past_key_values=..., use_cache=True)
    next_token_id = argmax(outputs.logits[:, -1, :])
    
    - 所以需要__call__ 
    - 返回:
          logits(Batch,sequence_size,vocab_size)
          past_key_values:任意可传递对象
          
    设计理念是下一个token是什么变成完全可推到的规则
    - next_token = last_token + 1 
    每次输出 logits, 让 argmax == “上一个 token + 1”
    - past_key_values 用 int 计步（只为了可传递）
    '''
    def __init__(self,vocab_size = 20):
        self.vocab_size = vocab_size
        self.device = torch.device("cpu")
        
    def __call__(self,input_ids:torch.LongTensor ,past_key_values=None,use_cache:bool = True):
        
        step = 0 if past_key_values is None else int(past_key_values)

        last_token = int(input_ids[0,-1].item())
        next_token = (last_token + 1) % self.vocab_size
        next_id = next_token
        
        batch_size, sequense_len = input_ids.shape
        # logits shape: (B, T, V), Executor 只用 [:, -1, :]
        logits = torch.zeros((batch_size,sequense_len,self.vocab_size),dtype=torch.float32)
        logits[:,-1,next_id] = 1.0
        return SimpleNamespace(logits = logits,past_key_values = step+1)
class DummyModelLoader:
    '''
    
    1. 需要有get_model()
    2. device 在model里 
    '''     
    def __init__(self,model):
        self._model = model
    
    def get_model(self):
        return self._model
      

 
    
'''
设计的时候应该考虑的是

1. 应该先看是否有 stop token id
    如果有 
        1.查看 当前token id  
        2.查看 stop token id的list 
        3.是的话 直接断开
2. 是否有 stop eos 的id
    1. 在request里吗
    2. 在tokenizer里吗 
    3. 找了list 看当前id 是否在是这个id 
'''


'''
prefill 完后命中 stop token id 
prefill 完后命中 eos 
'''

'''
decode 阶段命中 stop token id (先可能发生的事情)
decode 阶段命中 eos()
'''

'''

'''

'''
stop eos 存放在的tokenizer还是 request里
且request 比 tokenizer先 
'''

'''

'''

'''
完全没触发 只受max_new tokens 限制
'''

def _run_and_parse_ids(ex: Executor, req: Request) -> list[int]:
    out = ex.run(req)
    got = [int(x) for x in out.split()] if out.strip() else []
    return got


# ---------- 9 tests for branch coverage ----------

def test_01_prefill_hits_stop_token_ids_returns_immediately():
    # prefill token = 4, stop_token_ids 命中 -> prefill 后立刻 return
    model = DummyCausalLM()
    ex = Executor(DummyModelLoader(model), DummyTokenizer(eos_token_id=999))

    req = Request(id=1, prompt="hi", max_new_tokens=10, stop_on_eos=True, stop_token_ids=[4])
    got = _run_and_parse_ids(ex, req)

    assert got == [1, 2, 3, 4]
    assert req.finished is True


def test_02_decode_hits_stop_token_ids_breaks():
    # decode 序列：prefill 4 -> decode 5 -> decode 6 命中 stop_token_ids -> break
    model = DummyCausalLM()
    ex = Executor(DummyModelLoader(model), DummyTokenizer(eos_token_id=999))

    req = Request(id=2, prompt="hi", max_new_tokens=50, stop_on_eos=True, stop_token_ids=[6])
    got = _run_and_parse_ids(ex, req)

    assert got == [1, 2, 3, 4, 5, 6]
    assert req.finished is True


def test_03_prefill_hits_eos_from_tokenizer_returns_immediately():
    # eos 来自 tokenizer，prefill token=4 命中 eos -> return
    model = DummyCausalLM()
    ex = Executor(DummyModelLoader(model), DummyTokenizer(eos_token_id=4))

    req = Request(id=3, prompt="hi", max_new_tokens=50, stop_on_eos=True)
    got = _run_and_parse_ids(ex, req)

    assert got == [1, 2, 3, 4]
    assert req.finished is True


def test_04_decode_hits_eos_from_tokenizer_breaks():
    # eos 来自 tokenizer，decode 到 6 命中 eos -> break
    model = DummyCausalLM()
    ex = Executor(DummyModelLoader(model), DummyTokenizer(eos_token_id=6))

    req = Request(id=4, prompt="hi", max_new_tokens=50, stop_on_eos=True)
    got = _run_and_parse_ids(ex, req)

    assert got == [1, 2, 3, 4, 5, 6]
    assert req.finished is True


def test_05_request_eos_overrides_tokenizer_eos_priority():
    # tokenizer eos=6，但 request eos=4，应当在 prefill 4 就停（验证 request 优先级）
    model = DummyCausalLM()
    ex = Executor(DummyModelLoader(model), DummyTokenizer(eos_token_id=6))

    req = Request(id=5, prompt="hi", max_new_tokens=50, stop_on_eos=True, eos_token_id=4)
    got = _run_and_parse_ids(ex, req)

    assert got == [1, 2, 3, 4]  # 若错误用 tokenizer 优先，会继续到 6（序列里会出现 5）
    assert req.finished is True


def test_06_stop_on_eos_false_disables_eos_stop_even_if_token_equals_eos():
    # stop_on_eos=False：即使 prefill token=4 == eos(4)，也不能因为 eos 停
    model = DummyCausalLM()
    ex = Executor(DummyModelLoader(model), DummyTokenizer(eos_token_id=4))

    # max_new_tokens=2：保证至少走 1 次 decode（prefill 4 + decode 5）
    req = Request(id=6, prompt="hi", max_new_tokens=2, stop_on_eos=False)
    got = _run_and_parse_ids(ex, req)

    assert got == [1, 2, 3, 4, 5]
    assert req.finished is True


def test_07_eos_id_none_means_eos_never_triggers_stop_even_when_stop_on_eos_true():
    # eos_id=None 分支：request.eos_token_id 不给，tokenizer 没有 eos_token_id 属性
    model = DummyCausalLM()
    tok = DummyTokenizer(eos_token_id=123)
    delattr(tok, "eos_token_id")  # 让 getattr(tokenizer, "eos_token_id", None) -> None
    ex = Executor(DummyModelLoader(model), tok)

    req = Request(id=7, prompt="hi", max_new_tokens=2, stop_on_eos=True)
    got = _run_and_parse_ids(ex, req)

    assert got == [1, 2, 3, 4, 5]  # 不会因 eos 停
    assert req.finished is True


def test_08_stop_token_ids_present_but_not_hit_then_eos_hits_in_prefill():
    # stop_token_ids 存在但不命中（membership False），然后 eos 命中（prefill 4）
    model = DummyCausalLM()
    ex = Executor(DummyModelLoader(model), DummyTokenizer(eos_token_id=4))

    req = Request(id=8, prompt="hi", max_new_tokens=50, stop_on_eos=True, stop_token_ids=[999])
    got = _run_and_parse_ids(ex, req)

    assert got == [1, 2, 3, 4]
    assert req.finished is True


def test_09_stop_token_ids_present_but_not_hit_and_eos_not_hit_runs_to_max_new_tokens():
    # stop_token_ids membership False + eos compare False，然后只受 max_new_tokens 控制
    # max_new_tokens=2：prefill 4 + decode 5；eos=7 不会命中
    model = DummyCausalLM()
    ex = Executor(DummyModelLoader(model), DummyTokenizer(eos_token_id=7))

    req = Request(id=9, prompt="hi", max_new_tokens=2, stop_on_eos=True, stop_token_ids=[999])
    got = _run_and_parse_ids(ex, req)

    assert got == [1, 2, 3, 4, 5]
    assert req.finished is True