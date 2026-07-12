"""Configuration loader. config.yaml is the contract between engineering and
the bank's risk team: risk policy changes there, never in code."""
import os
from functools import lru_cache

import yaml

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_PATH = os.path.join(_ROOT, "config.yaml")


@lru_cache(maxsize=4)
def load_config(path: str = DEFAULT_PATH) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def product_terms(cfg: dict) -> dict:
    return cfg["policy"]["products"]


def products(cfg: dict) -> list:
    return list(cfg["policy"]["products"].keys())
