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

# 2025-12-10
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


## Model Loader 模型加载器

### 和engine的区分
1. 这里只拿模型本身
2. 利用config loader在config里yaml里拿定义的模型（存在wsl里的）的path
3. 此处调用huggingface的接口 AutoCausalLM 用来直接得到对应的模型架构（不用自己手写）
3.1 model_path 必须提供 如果提供的是名字 那么会去下载
3.2 需要额外禁止一下localfile only
3.3 这里练习基础 不直接使用huggingface的自动挡 例如把模型放在哪张显卡啊放内存啊之类的 （本质是torch的语法）
3.4 开启eval模式


# ModelLoader 的作用与 Engine 的区分

### 1. ModelLoader 的职责是“加载模型本体”

- 仅负责从磁盘/WsL 加载模型权重与架构。
- 不负责推理流程、不负责生成逻辑、不负责 KV cache，不负责 tokenizer。

- 所有“运行时行为（如生成 token 流、状态维护）”交给 Engine。

### 2. 从配置系统中读取模型路径

- 通过 config loader 从 configs/model_paths.yaml 中解析模型路径。

- path 指向 WSL 本地目录：例如 /home/dexterding/models/Qwen2-7B

- 这样避免在代码中写死路径，使模型切换更灵活。

### 3. 利用 HuggingFace AutoModelForCausalLM 获取模型架构

- 通过 AutoModelForCausalLM.from_pretrained(model_path) 自动构建 Qwen2 的网络结构。

- 不需要手动编写 PyTorch 的 transformer 层，实现“框架即架构”。

- 模型类基于 config 自动选择正确的实现（如 Qwen2ForCausalLM）。

### 4. 模型路径设定相关细节

- 必须提供 model_path，否则会默认下载网络模型。

- 开发阶段应强制：
```python
local_files_only=True
```
  避免误触发在线下载行为。

### 5.不使用 HF 的“自动分布式/自动 placement”能力（刻意训练基础能力）

- 不使用 Accelerate

- 不使用 device_map="auto"
→ 目的是练习底层 PyTorch 的模型放置语法：

- model.to("cuda:0")
 并理解显存管理，而不是被自动挡抽象遮蔽。

### 6. 开启 eval() 模式

禁止梯度：节约显存并防止误反向。

推理模式下 LayerNorm、Dropout 会切换到 inference 行为。



完善了MVP
LLM 选用的是 qwen2-7b 使用单卡24gb可运行
model loader和tokenizer loader输入model_name后自动调用config loader的load model path
注意的点1
tokenizer的init是使用model name
modelLoader init也是model name
但是对应的AutoTokenizer.from_pretrained
和 AutoModelForCausalLM.from_pretrained
都是需要的model path 并且是访问本地local file
local file存放在config的yaml中
注意的点2 目前的tokenizer 实现的是
输出2d tensor （1,seq_len） 转1d tensor（seq_len）
因为只有一个prompt 但是tokenizer默认的是batch_size
目前实现的mvp对应的单个prompt且一次一个token 输出
注意的点3
模型运行的device 是取决于 modeLoader的device的
不是很好应该下个版本改
在simpleengine里也是从model_loader的device里获取的device
注意的点4 
目前因为model 需要的是batch size的input 所以单个的prompt转换成的id
需要从（sequence） 转化为（1，sequence）
outputs里存的 除了logit之外因为有use_cache 所以会返回cache
prefil的model是输入prompt+ids和use_cache
但是之后的一个一个算token的部分 
使用的是
1.单个最新生成的token转化为tensor哦
2. 以及过去的past key values
3.一次次覆盖for loop之前的 
model其实是两种算法状态 prefill和decode是两个
最后使用tokenizer的decode


### 2025-12-11 开发日志｜MVP 梳理

**1. MVP 状态**

- 完成最小可运行版本（MVP）
- LLM 选用 **qwen2-7b**，在单卡 **24GB** 显存上可正常运行

---

**2. Model / Tokenizer / Config 关系**

- `ModelLoader` 和 `Tokenizer` 的初始化参数都是 `model_name`
- 内部会通过 `ConfigLoader`：
  - 根据 `model_name` 读取对应的 **model path**（本地路径），配置存放在 `config/*.yaml` 中  
  - `AutoTokenizer.from_pretrained(...)` 和 `AutoModelForCausalLM.from_pretrained(...)` 实际使用的是 **本地 model path**，只访问 local file，不依赖在线下载

---

**3. Tokenizer 实现细节**

- `Tokenizer.__init__` 只接收 **model name**，不直接接触路径
- 当前 `encode` 的行为：
  - HuggingFace 默认返回 **2D tensor**：`(1, seq_len)`（因为默认有 batch 维度）
  - 由于当前仅支持单个 prompt，当前取`[0]`把输出压成 **1D tensor**：`(seq_len,)`
- 当前 MVP 假设：
  - **只处理单个 prompt**
  - **一次只生成一个 token** 的自回归输出逻辑

---

**4. Device 管理（有待改进）**

- 目前模型运行所在的 `device` 由 `ModelLoader.device` 决定
- `SimpleEngine` 内部也是直接从 `model_loader.device` 取 `device`
- 这种设计耦合度偏高：  
  - 设备逻辑绑在 `ModelLoader` 上并不理想  
  - 计划在下个版本重构 device 管理（从 Engine 或更上层统一下发）

---

**5. 推理流程 & KV Cache 行为**

- **输入形状处理**
  - 模型需要 batch 维度的输入
  - 单个 prompt 的 token ids 需要从 `(seq_len,)` 转成 `(1, seq_len)`

- **Prefill 阶段**
  - 调用：`model(input_ids, use_cache=True)`
  - 输入：完整的 prompt token ids（带 batch 维度）
  - 输出：
    - `logits`
    - `past_key_values`（KV cache，用于后续解码）

- **Decode 循环阶段**
  - 每一步：
    1. 取最新生成的 **单个 token id**，转成 tensor（带 batch & seq 维度，如 `(1, 1)`）
    2. 连同上一轮的 `past_key_values` 一起送入模型
    3. 使用新返回的 `past_key_values` 覆盖旧值
  - 模型在内部对 **prefill** 和 **decode** 有不同的计算路径：
    - Prefill：一次性处理完整序列，建立完整 KV cache
    - Decode：每次只处理一个新 token，复用历史 KV cache

- **输出还原**
  - 收集所有生成的 `token_ids`
  - 使用 `tokenizer.decode(...)` 转回文本

---

**6. 当前版本的限制**

- 只支持：
  - 单个 prompt
  - 一次生成一个 token 的循环解码
- Device 逻辑暂时耦合在 `ModelLoader`，计划在后续版本重构

# 2025-12-12 

1. 原本的 minimal generate部分的代码逻辑 转移到了runtime/api.py
   把参数部分 提了出来 
   1. 模型名字        str
   2. prompt本身是真么 str
   3. max new token 提了出来
   4. 把设备在哪里 放着里了（但是目前还是耦合的 默认设置在model_loader处）
    是否应该设计一下 先使用assert 使得loader和engine是在一个device cuda：1上的

  本身的minimal_generate.py 加入paser
  paser 涵盖上述四个点

  paser 含有三步
```python
    #1.创建一个parser
    parser = argparse.ArgumentParser()
    
    #2.  一堆要加进去给args的
    parser.add_argument("--model-name",type = str, required=True, help = "Key in configs/model_paths.yaml")
    parser.add_argument("--prompt", type=str, default="你好，介绍一下你自己") 

    #3. args = 
    args = parser.parse_args()  

```
注意 arg中间 - 连接的 后面都要变成 _
```python
    output = generate(
        model_name=args.model_name,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        device=args.device,
    )
```
   涉及改动 
```bash
  [CHANGED]   minimal_generate.py  
  [NEW]       runtime/api.py 
```
    