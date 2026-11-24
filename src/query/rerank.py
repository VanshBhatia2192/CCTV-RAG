import torch
from pathlib import Path
from PIL import Image
from src.common.clip import DEV
from src.common.video import frames
from src.incident import caption

def _images(kind, c):
    if kind == "incident":
        ims = [Image.fromarray(f) for f in frames(c["path"], c["rel_start"], c["rel_end"], 4)]
    else:
        ims = [Image.open(p).convert("RGB") for p in sorted(Path(c["crops_dir"]).glob("*.jpg"))[:3]]
    for im in ims:
        im.thumbnail((448, 448))
    return ims

@torch.no_grad()
def ask_yes(ims, question):
    """Logit margin yes - no for a yes/no question about the images (> 0 means the VLM leans yes)."""
    m, p = caption._load()
    prompt = f"{question} Answer yes or no."
    msg = [{"role": "user", "content": [{"type": "image"} for _ in ims] + [{"type": "text", "text": prompt}]}]
    t = p.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
    inp = p(text=[t], images=ims, return_tensors="pt").to(DEV)
    lg = m(**inp).logits[0, -1].float()
    tok = p.tokenizer
    y = max(lg[tok.encode(w)[0]] for w in ("yes", "Yes"))
    n = max(lg[tok.encode(w)[0]] for w in ("no", "No"))
    return float(y - n)

def yes_score(ims, text):
    return ask_yes(ims, f'Does this match the description: "{text}"?')

def rerank(kind, text, cands):
    for c in cands:
        ims = _images(kind, c)
        c["vlm"] = yes_score(ims, text) if ims else -99.0
    return sorted(cands, key=lambda c: c["vlm"], reverse=True)
