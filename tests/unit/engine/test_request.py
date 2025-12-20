import pytest
from engine.request import Request

def test_request_validation():
  #id 不能是负数
    with pytest.raises(ValueError):
        Request(id= -1,prompt='x')
  
  #max new token必须大于等于1
    with pytest.raises(ValueError):
        Request(id= 0, prompt="x",max_new_tokens=0)
  
  #prompt本身需要是str
    with pytest.raises(ValueError):
        Request(id= 1, prompt=123,)
  
  #stop token id must be List[int] or None
    with pytest.raises(ValueError):
        Request(id= 1, prompt='x', stop_token_ids= "bad")
        
    with pytest.raises(ValueError):
        Request(id= 1, prompt='x', stop_token_ids= [1,'2'])
        
    r = Request(id= 1, prompt= 'x', max_new_tokens= 1)
    assert r.max_new_tokens == 1
    assert r.stop_on_eos is True 