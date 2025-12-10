import torch
from transformers import AutoModelForCausalLM
from engine.config_loader import load_model_path

class ModelLoader:
    def __init__(self, model_name: str, device: str = "cuda:1"):
        model_path = load_model_path(model_name)
      
        self.device = torch.device(device)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            local_files_only=True,# 关键：禁止网络访问)
            torch_dtype=torch.float16   # ⭐ 必须 FP16 才能省显存
        )  
        self.model = self.model.to(self.device)
        self.model.eval()

    def get_model(self):
        return self.model


# 不用自动挡  
# class ModelLoader:
#   def __init__(self,model_name:str,device:str = "cuda"):
#       self.model = AutoModelForCausalLM.from_pretrained(
#         model_name,
#         torch_dtype = "auto",
#         device_map = "auto"
#       )
#       self.model.eval()
      
#   def get_model(self):
#       return self.model
    
  
  