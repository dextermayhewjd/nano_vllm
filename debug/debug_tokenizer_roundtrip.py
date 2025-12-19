import torch
from engine.tokenizer import Tokenizer

def main():
    tokenizer = Tokenizer("qwen2_7b")

    text = "你好，世界！"
    token_ids = tokenizer.encode(text)

    print("Text:", text)
    print("Token IDs:", token_ids, token_ids.shape)

    decoded = tokenizer.decode(token_ids)
    print("Decoded:", decoded)

if __name__ == "__main__":
    main()