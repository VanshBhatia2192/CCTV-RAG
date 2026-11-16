import yaml
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

def load(name):
    p = ROOT / "configs" / f"{name}.yaml"
    if not p.exists():
        p = Path("configs") / f"{name}.yaml"
    return yaml.safe_load(open(p, encoding="utf-8"))

PATHS, MODELS, PIPE = load("paths"), load("models"), load("pipeline")
import os
SHARD = int(os.getenv("SHARD", 0))
NSHARDS = int(os.getenv("NSHARDS", 1))
