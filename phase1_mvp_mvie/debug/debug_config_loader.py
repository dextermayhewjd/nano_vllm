"""
Docstring for debug.debug_config_loader
用来查看 config_loader 所返回的 model path是什么样子的
"""
from engine.config_loader import load_model_path

def main():
    model_path = load_model_path("qwen2_7b")
    print("Loaded model path:", model_path)
if __name__ == "__main__":
    main()