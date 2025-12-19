from engine.simple_engine import SimpleEngine

class DummyModel:
    device = "cpu"

class DummyLoader:
    """
    假 ModelLoader：
    只提供 device 属性，避免加载真实模型
    """
    device = "cpu"
    
    def get_model(self):
        return DummyModel()
    
class DummyTokenizer:
    def encode(self, text):
        return [1, 2, 3]

    def decode(self, ids):
        return "dummy"

def test_simple_engine_init_only():
    """
    Contract:
    - SimpleEngine 能被正确初始化
    - 不强依赖真实模型
    """
    engine = SimpleEngine(
        model_loader=DummyLoader(),
        tokenizer=DummyTokenizer(),
        max_new_tokens=1,
    )

    assert engine.max_new_tokens == 1