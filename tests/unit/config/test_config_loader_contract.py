from engine.config_loader import load_model_path

def test_load_model_path_returns_valid_string():
    """
    Contract:
    - 给定合法 model_name
    - 返回一个字符串路径
    """
    path = load_model_path("qwen2_7b")

    assert isinstance(path, str)
    assert len(path) > 0
