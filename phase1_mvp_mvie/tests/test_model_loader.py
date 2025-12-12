from engine.model_loader import ModelLoader

def main():
    loader = ModelLoader("qwen2_7b")
    model = loader.get_model()
    
    #得到 模型本身的config
    print("Fetched Model:", model.config) 
    
    
    # print 每一层的结构
    for name, module in model.named_modules():
        print(name, ":", module)  

if __name__ == "__main__":
    main()