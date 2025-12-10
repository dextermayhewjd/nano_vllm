from engine.tokenizer import Tokenizer
from engine.model_loader import ModelLoader
from engine.simple_engine import SimpleEngine
from engine.config_loader import load_model_path

def main():
    model_name = "qwen2_7b"
    tokenizer = Tokenizer(model_name)
    model_loader = ModelLoader(model_name)

    engine = SimpleEngine(model_loader, tokenizer, max_new_tokens=50)

    output = engine.generate("你好，世界")
    print("Model output:\n", output)

if __name__ == "__main__":
    main()