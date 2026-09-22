import yaml
import os


def load_config(path: str = None) -> dict:
    if path is None:
        path = os.path.join(os.path.dirname(__file__), "..", "config.yaml")
    with open(path, "r") as f:
        return yaml.safe_load(f)
