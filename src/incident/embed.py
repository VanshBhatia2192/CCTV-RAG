import argparse, torch
from pathlib import Path
from tqdm import tqdm
from src.common.clip import DEV
from src.common.config import MODELS
from src.common.video import frames
from src.index import vector_db
from src.index.sql_db import connect

_m = _p = None

def _load():
    global _m, _p
    if _m is None:
        from transformers import XCLIPModel, XCLIPProcessor
        p = Path(MODELS["text_video"]["path"])
        path = str(p) if p.exists() else "microsoft/xclip-base-patch32"
        _m = XCLIPModel.from_pretrained(path).to(DEV).eval()
        _p = XCLIPProcessor.from_pretrained(path)
    return _m, _p

def _t(x):
    x = x if torch.is_tensor(x) else x.pooler_output
    assert x.shape[-1] == 512, f"unexpected embedding shape {tuple(x.shape)}"
    return torch.nn.functional.normalize(x, dim=-1)

@torch.no_grad()
def embed_video(frs):
    m, p = _load()
    pv = p(videos=[list(frs)], return_tensors="pt")["pixel_values"].to(DEV)
    return _t(m.get_video_features(pixel_values=pv))[0].cpu().numpy()

@torch.no_grad()
def embed_text(texts):
    m, p = _load()
    t = p(text=list(texts), return_tensors="pt", padding=True).to(DEV)
    return _t(m.get_text_features(input_ids=t["input_ids"], attention_mask=t["attention_mask"])).cpu().numpy()

def main(limit):
    db = connect()
    rows = db.execute("SELECT w.*, c.path, c.camera FROM windows w JOIN chunks c ON c.id=w.chunk_id "
                      "WHERE w.embedded=0 LIMIT ?", (limit,)).fetchall()
    for r in tqdm(rows, desc="incident embed"):
        frs = frames(r["path"], r["rel_start"], r["rel_end"], 8)
        if len(frs) < 8:
            continue
        v = embed_video(frs)
        payload = {"window_id": r["id"], "chunk_id": r["chunk_id"], "camera": r["camera"], "path": r["path"],
                   "rel_start": r["rel_start"], "rel_end": r["rel_end"], "start_s": r["start_s"], "end_s": r["end_s"]}
        vector_db.upsert("incident", [r["id"]], [v.tolist()], [payload])
        db.execute("UPDATE windows SET embedded=1 WHERE id=?", (r["id"],)); db.commit()

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--limit", type=int, default=100)
    main(ap.parse_args().limit)
