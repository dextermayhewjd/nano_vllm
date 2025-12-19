# tests/test_executor_single.py
from engine import ModelLoader,Tokenizer
from engine.simple_engine import SimpleEngine
from engine.executor import Executor
from engine.request import Request

def test_executor_matches_simple_engine():
    model_name = "qwen2_7b"
    prompt = "你好"
    
    tokenizer = Tokenizer(model_name)
    loader = ModelLoader(model_name, device="cuda:0")

    # 用老的 SimpleEngine
    simple_engine = SimpleEngine(loader, tokenizer, max_new_tokens=20)
    out_simple = simple_engine.generate(prompt)
    
        # 用新的 Executor + Request
    executor = Executor(loader, tokenizer)
    req = Request(id=1, prompt=prompt, max_new_tokens=20)
    out_exec = executor.run(req)
    
    print("SimpleEngine:", out_simple)
    print("Executor    :", out_exec)

    assert out_simple == out_exec
    
    