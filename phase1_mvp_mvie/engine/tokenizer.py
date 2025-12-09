from transformers import AutoTokenizer

class Tokenizer:
    def __init__(self, model_name:str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        
    def encode(self, string: str):
      return self.tokenizer.encode(
                                    text=string,
                                    return_tensors="pt")
    
    def decode(self,
               token_ids):
      return self.tokenizer.decode(token_ids,skip_special_tokens = True)
    
    
    
    
    
    
    
    
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
