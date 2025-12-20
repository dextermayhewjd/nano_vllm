from dataclasses import dataclass, field
from typing import List,Optional, Any
import torch

@dataclass
class Request:
    id: int
    prompt: str
    max_new_tokens:int = 50
    
    '''
    stop conditions 停下来的条件
    '''
    stop_token_ids: Optional[List[int]] = None
    stop_on_eos: bool = True
    eos_token_id: Optional[int] = None  
    # 不填时，后续由 Executor/Tokenizer 推断    
    
    
    
    '''
    runtime fields 运行时候填进去的 
    '''
    input_ids: Optional[torch.Tensor] = None
    # 延迟确定：请求对象先进入队列，真正执行时才补齐必要的运行时资源。
    
    generated_ids:List[int] = field(default_factory=list)
    # 如果不这么写 那这个列表会被所有 Request 实例共享（经典坑）
    # default_factory=list 会保证每个 Request 都有自己的新 list。
    
    finished:bool = False
    
    
    
    # 预留到时候KV Cache 使用的句柄 （到时候换了真的kv block 句柄）
    kv_cache : Any = None
    
    def __post_init__(self) -> None:
        if not isinstance(self.id, int) or self.id < 0:
            raise ValueError("Request.id must be a non-negative int")
        if not isinstance(self.prompt, str):
            raise ValueError("Request.prompt must be a str")
        if not isinstance(self.max_new_tokens, int) or self.max_new_tokens < 1:
            raise ValueError("Request.max_new_tokens must be >= 1")

        if self.stop_token_ids is not None:
            if not isinstance(self.stop_token_ids, list) or not all(isinstance(x, int) for x in self.stop_token_ids):
                raise ValueError("Request.stop_token_ids must be List[int] or None")