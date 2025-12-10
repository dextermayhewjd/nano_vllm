from engine.config_loader import load_model_path

def main():
    model_path = load_model_path("qwen2_7b")
    print("Loaded model path:", model_path)
if __name__ == "__main__":
    main()