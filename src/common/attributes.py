import numpy as np
from src.common import clip

COL = ["black", "white", "gray", "red", "blue", "green", "yellow", "brown", "orange", "pink"]
LABELS = {
  "person": {
    "upper_color": [(c, f"a person wearing a {c} top") for c in COL],
    "lower_color": [(c, f"a person wearing {c} pants") for c in COL],
    "upper_type": [(t, f"a person wearing a {t}") for t in ["t-shirt", "jacket", "hoodie", "coat", "shirt", "dress"]],
    "bag": [("backpack", "a person carrying a backpack"), ("handbag", "a person carrying a handbag"),
            ("shopping bag", "a person carrying a shopping bag"), ("none", "a person with no bag")],
    "hat": [("hat", "a person wearing a hat"), ("none", "a person with no hat")],
  },
  "vehicle": {
    "color": [(c, f"a {c} vehicle") for c in COL],
    "type": [(t, f"a photo of a {t}") for t in ["car", "suv", "van", "truck", "bus", "motorcycle"]],
  },
}
_cache = {}

def zero_shot(kind, vec):
    out = {}
    for attr, items in LABELS[kind].items():
        if (kind, attr) not in _cache:
            _cache[(kind, attr)] = clip.embed_text([p for _, p in items])
        out[attr] = items[int(np.argmax(_cache[(kind, attr)] @ vec))][0]
    return out
