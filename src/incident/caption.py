import argparse, torch
from PIL import Image
from tqdm import tqdm
from src.common.clip import DEV
from src.common.config import MODELS, SHARD, NSHARDS
from src.common.video import frames
from src.index import vector_db
from src.index.sql_db import connect

PROMPT = ("These are frames from a surveillance clip in time order. Describe what happens in 1-2 sentences. "
          "Mention people, vehicles, their actions, and any unusual incident.")
_m = _p = None

def _load():
    global _m, _p
    if _m is None:
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
        path = MODELS["vlm"]["path"]
        _m = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            path, torch_dtype=torch.bfloat16 if DEV == "cuda" else torch.float16).to(DEV).eval()
        _p = AutoProcessor.from_pretrained(path)
    return _m, _p

@torch.no_grad()
def caption(frs, n=4):
    m, p = _load()
    imgs = []
    for f in frs[:: max(len(frs) // n, 1)][:n]:
        im = Image.fromarray(f); im.thumbnail((448, 448)); imgs.append(im)
    msg = [{"role": "user", "content": [{"type": "image"} for _ in imgs] + [{"type": "text", "text": PROMPT}]}]
    text = p.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
    inp = p(text=[text], images=imgs, return_tensors="pt").to(DEV)
    out = m.generate(**inp, max_new_tokens=80, do_sample=False)
    return p.batch_decode(out[:, inp["input_ids"].shape[1]:], skip_special_tokens=True)[0].strip()

def main(limit):
    db = connect()
    rows = db.execute("SELECT w.*, c.path FROM windows w JOIN chunks c ON c.id=w.chunk_id "
                      "WHERE w.caption IS NULL AND w.id % ? = ? LIMIT ?", (NSHARDS, SHARD, limit)).fetchall()
    for r in tqdm(rows, desc="captions"):
        cap = caption(frames(r["path"], r["rel_start"], r["rel_end"], 8))
        db.execute("UPDATE windows SET caption=? WHERE id=?", (cap, r["id"])); db.commit()
        if r["embedded"]:
            vector_db.client().set_payload("incident", {"caption": cap}, points=[r["id"]])

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--limit", type=int, default=50)
    main(ap.parse_args().limit)
