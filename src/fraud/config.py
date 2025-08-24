from __future__ import annotations
import os, yaml
from dataclasses import dataclass
from typing import Any, Dict

@dataclass
class Config:
    seed: int
    reference_date: str
    train: Dict[str, Any]
    features: Dict[str, Any]

def load_config(path: str = "configs/default.yaml") -> Config:
    with open(path, "r", encoding="utf-8") as f:
        d = yaml.safe_load(f)
    return Config(**d)
