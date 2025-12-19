# from engine.tokenizer import Tokenizer
# from engine.model_loader import ModelLoader
# from engine.simple_engine import SimpleEngine
# from engine.config_loader import load_model_path

# def main():
#     model_name = "qwen2_7b"
#     tokenizer = Tokenizer(model_name)
#     model_loader = ModelLoader(model_name)

#     engine = SimpleEngine(model_loader, tokenizer, max_new_tokens=50)

#     output = engine.generate("你好，世界")
#     print("Model output:\n", output)

# if __name__ == "__main__":
#     main()

import argparse
from runtime import generate

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name",type = str, required=True,  help = "Key in configs/model_paths.yaml")
    parser.add_argument("--prompt", type=str, default="你好，介绍一下你自己")
    parser.add_argument("--max-new-tokens", type=int, default= 50)
    parser.add_argument("--device", type=str, default="cuda:1")
    args = parser.parse_args()
    
    output = generate(
        model_name=args.model_name,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        device=args.device,
    )
    print("=== Prompt ===")
    print(args.prompt)
    print("=== Output ===")
    print(output)

if __name__ == "__main__":
    main()    
'''
python ./examples/minimal_generate.py --model-name qwen_7b
'''