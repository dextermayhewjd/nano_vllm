from transformers import AutoTokenizer
import torch
class Tokenizer:
    def __init__(self, model_name:str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        
    def encode(
        self, 
        prompt: str,
        add_special_tokens:bool = True
        )-> torch.Tensor: # 1D tensor: (seq_len,)
      #__call__语法糖 
        out = self.tokenizer(
            prompt,
            add_special_tokens=add_special_tokens,
            return_tensors="pt",
        ) # 此处返回的是batchEncoding 
      
        # out["input_ids"] shape: (1, seq_len) 
        # 要变成→ return 1D 
        
        # return out["input_ids"] 
        # #test Token IDs: tensor([[108386,   3837,  99489]])
        
        return out["input_ids"][0]
        #Token IDs: tensor([108386,   3837,  99489])
    
    def decode(self,
               token_ids,
               skip_special_tokens:bool = True):
      return self.tokenizer.decode(token_ids,skip_special_tokens = skip_special_tokens)
    
    
    
    
    
    
    
    
'''
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

ids = tokenizer.encode("hello world!")
print(ids)
# [101, 7592, 2088, 999, 102]

print(tokenizer.decode(ids))
# [CLS] hello world! [SEP]

print(tokenizer.decode(ids, skip_special_tokens=True))
# hello world!
'''
