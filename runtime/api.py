from engine import ModelLoader, Tokenizer

from engine.executor import Executor
from engine.request import Request

# 简单的 request id 生成器，后面可以换成 UUID
_request_id_counter = 0
def _next_request_id():
    global _request_id_counter
    _request_id_counter += 1
    return _request_id_counter



def generate(model_name: str, prompt: str, max_new_tokens: int = 50, device: str = "cuda:1") -> str:
    """
    高层 API:一行把 model + tokenizer + executor 串起来
    是原来minimal_generate.py 部分的 最小生成搬过来了
    """
    tokenizer = Tokenizer(model_name)
    model_loader = ModelLoader(model_name, device=device)
    # engine = SimpleEngine(model_loader=model_loader, tokenizer=tokenizer, max_new_tokens=max_new_tokens)
    executor = Executor(model_loader=model_loader,tokenizer=tokenizer)
    request = Request(
            id=_next_request_id(),
            prompt=prompt,
            max_new_tokens=max_new_tokens,
    )
    return executor.run(request)
