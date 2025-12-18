from engine import ModelLoader, Tokenizer
from engine.simple_engine import SimpleEngine

def generate(model_name: str, prompt: str, max_new_tokens: int = 50, device: str = "cuda:1") -> str:
    """
    高层 API：一行把 model + tokenizer + engine 串起来
    是原来minimal_generate.py 部分的 最小生成搬过来了
    """
    tokenizer = Tokenizer(model_name)
    model_loader = ModelLoader(model_name, device=device)
    engine = SimpleEngine(model_loader=model_loader, tokenizer=tokenizer, max_new_tokens=max_new_tokens)
    return engine.generate(prompt)
