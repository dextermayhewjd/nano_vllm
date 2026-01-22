from typing import Any

class KVCache:
    '''
    Docstring for KVCache
    最小封装 以便日后实现
    目前只是把huggingface的past_key_values 装起来
    '''
    def __init__(self,past_key_values:Any = None):
        self._pkv = past_key_values
        
    @property
    def data(self):
        return self._pkv
    
    @data.setter 
    def data(self,value):
        self._pkv = value
    
    '''
    fluent python chapter22
    '''