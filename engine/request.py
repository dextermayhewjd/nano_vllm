from dataclasses import dataclass, field
from typing import List,Optional
import torch

@dataclass
class Request:
    id: int
    prompt: str
    max_new_tokens:int = 50
    
    
    input_ids: Optional[torch.Tensor] = None
    generated_ids:List[int] = field(default_factory=list)
    finished:bool = False
    
    
    # 预留到时候KV Cache 使用的句柄 （到时候换了真的kv block 句柄）
    kv_cache = None