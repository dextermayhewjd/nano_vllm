开发日志

2025-12-9

- 今日处理wsl和shadowsocket

  1. 记得开allow other device to connect

  2. 但是同时也要记得把防火墙打开（在不下载的时候）

  3. 代理部分要考虑调回

  4. 其次需要条python interpreter

      4.1需要vscode支持 wsl版本的
  
      4.2需要确保能够pip能够使用 要装一个额外包才能用pysock好像
  
      4.3反复验证是否能够

  - # tokenizer 
  使用的是huggingface的autokenizer 
  ```python
    from transformers import AutoTokenizer

  # Download vocabulary from huggingface.co and cache.
  tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")

  # Download vocabulary from huggingface.co (user-uploaded) and cache.
  tokenizer = AutoTokenizer.from_pretrained("dbmdz/bert-base-german-cased")

  # If vocabulary files are in a directory (e.g. tokenizer was saved using *save_pretrained('./test/saved_model/')*)
  # tokenizer = AutoTokenizer.from_pretrained("./test/bert_saved_model/")

  # Download vocabulary from huggingface.co and define model-specific arguments
  tokenizer = AutoTokenizer.from_pretrained("FacebookAI/roberta-base", add_prefix_space=True)

  # Explicitly use the tokenizers backend
  tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/llama-tokenizer", backend="tokenizers")

  # Explicitly use the sentencepiece backend
  tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/llama-tokenizer", backend="sentencepiece")
  ```

  然后对于tokenizer的encode和decode 在这里  
  huggingface的encode和decode怎么用可以看[[https://huggingface.co/docs/transformers/v5.0.0rc0/zh/main_classes/tokenizer#transformers.TokenizersBackend.encode]]


  # decode和 encode 
  encode 加了 return tensors="pt" （pytorch tensor）  
  因为要返回2d的tensor（batch）为1  
  1. 不加  
  [15492,123] (python list[int])  

  2. 加了  
  tensor([[15496,  995]])  
  返回 二维 tensor（shape = [1, seq_len]）  
  因为encode一般是batch encode  
  但我只是一个prompt 所以默认取用[0]  
  ```python
        # out["input_ids"] shape: (1, seq_len) 
        # 要变成→ return 1D 
        
        # return out["input_ids"] 
        # #test Token IDs: tensor([[108386,   3837,  99489]])
        
        return out["input_ids"][0]
        #Token IDs: tensor([108386,   3837,  99489])
  ```



  如果直接使用tokenizer本身 那么看__call__（语法糖）  

    return_tensors (str or TensorType, optional) —  
    If set, will return tensors    
    instead of list of python integers. Acceptable values are:  
    'pt': Return PyTorch torch.Tensor objects.  
    'np': Return Numpy np.ndarray objects.  
    这里可以看出可以返回pt 然后后续return生成的里面包含的一项是tensor  
return type是BatchEncoding
[[https://huggingface.co/docs/transformers/v5.0.0rc0/zh/main_classes/tokenizer#transformers.TokenizersBackend.__call__.return_tensors]]

## 下载qwen2-7b 
```bash
pip install "huggingface_hub[cli]"
```
下载到指定目录：
```bash
huggingface-cli download \
    qwen/Qwen2-7B \
    --local-dir /home/yourname/models/Qwen2-7B
```
自己用肯定是要换名字啦
```bash
huggingface-cli download \
    qwen/Qwen2-7B \
    --local-dir /home/dexterding/models/Qwen2-7B
```
## 开发过程中写小test来明白在干嘛

写了一个小test 用来了解每一步tokenizer在干什么  
在/run_test/test_tokenizer_qwen2.py 中  

```bash
python run_test/test_tokenizer_qwen2.py
```

```python
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
    
```
可以看出几点  
第一需要时的时候需要先定义 model_path 最好避免硬编码  
第二在使用自己写的东西的时候 不同文件夹下面是from 文件夹.python文件名 import 具体class  
第三使用的时候打印

## 避免硬编码

为了以后避免硬编码路径，可以写：

configs/model_paths.yaml

```yaml
qwen2_7b: "/home/dexterding/models/Qwen2-7B"
```


然后写一个 loader： 
在engine/config_loader.py  
输入要的模型名字 
会返回位置 
存放位置的地方在config的model_paths.yaml里


这样可以随时可以切换模型，不影响测试代码。 