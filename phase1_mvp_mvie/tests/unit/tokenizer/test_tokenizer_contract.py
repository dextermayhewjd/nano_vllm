import torch
from engine.tokenizer import Tokenizer

def test_tokenizer_encode_decode_contract():
    """
    Contract:
    - encode 返回 1D torch.Tensor
    - dtype 为 long
    - decode(encode(x)) 能得到非空字符串
    """
    tokenizer = Tokenizer("qwen2_7b")

    text = "你好，世界！"
    token_ids = tokenizer.encode(text)

    assert isinstance(token_ids, torch.Tensor)
    assert token_ids.ndim == 1
    assert token_ids.dtype == torch.long

    decoded = tokenizer.decode(token_ids)
    assert isinstance(decoded, str)
    assert len(decoded) > 0
