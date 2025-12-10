import yaml
import os

def load_model_path(name: str) -> str:
    # 获取项目根目录
    root_dir = os.path.dirname(os.path.dirname(__file__))

    # 定位 configs/model_paths.yaml
    config_file = os.path.join(root_dir, "configs", "model_paths.yaml")

    if not os.path.exists(config_file):
        raise FileNotFoundError(f"Config file not found: {config_file}")

    with open(config_file, "r", encoding="utf-8") as f:
        models_data = yaml.safe_load(f)

    if name not in models_data:
        raise KeyError(f"Model name '{name}' not found in {config_file}")

    return models_data[name]
