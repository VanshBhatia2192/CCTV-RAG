import yaml
from pathlib import Path

def load(name):
    return yaml.safe_load(open(Path("configs") / f"{name}.yaml"))

PATHS, MODELS, PIPE = load("paths"), load("models"), load("pipeline")
import os
SHARD = int(os.getenv("SHARD", 0))
NSHARDS = int(os.getenv("NSHARDS", 1))
