from engine.tokenizer import Tokenizer
from engine.config_loader import load_model_path

def main():
    # model_path = "/home/dexterding/models/Qwen2-7B" 不硬编码了
    # model_path = load_model_path("qwen2_7b")
    tok = Tokenizer("qwen2_7b")

    token_ids = tok.encode("你好，世界")
    print("Token IDs:", token_ids)
    #此处返回的是这个prompt的tensorid
    
    decode_prompt = tok.decode(token_ids)
    print("Decoded:", decode_prompt)

if __name__ == "__main__":
    main()