import torch
from engine.tokenizer import Tokenizer
from engine.config_loader import load_model_path

def main():
    # model_path = "/home/dexterding/models/Qwen2-7B" 不硬编码了
    # model_path = load_model_path("qwen2_7b")
    model_name = "qwen2_7b"  # 对应 configs/model_paths.yaml 的 key
    tokenizer = Tokenizer(model_name)

    text = "你好，世界！"
    token_ids = tokenizer.encode(text)
    # 1. 检查返回类型 & 维度
    assert isinstance(token_ids, torch.Tensor)
    assert token_ids.ndim == 1  # (seq_len,)
    
    
    
    decode_prompt = tokenizer.decode(token_ids)
    print("Original:", text)
    print("Token IDs:", token_ids)     #此处返回的是这个prompt的tensorid
    print("Decoded:", decode_prompt)

if __name__ == "__main__":
    main()      