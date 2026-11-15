import json
from pathlib import Path
import torch, torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from src.person import t2p
from src.index import vector_db
from src.index.sql_db import connect

def main():
    dev = "cuda"
    m, _ = t2p.load("models/finetuned/text2person/clip_t2p.pt", dev); m.eval()
    db = connect()
    rows = db.execute("SELECT t.*, c.camera FROM tracks t JOIN chunks c ON c.id=t.chunk_id WHERE t.kind='person'").fetchall()
    ids, vecs, pays, n = [], [], [], 0
    for r in tqdm(rows, desc="person_ft embed"):
        files = sorted(Path(r["crops_dir"]).glob("*.jpg"))
        if not files:
            continue
        x = torch.stack([t2p.prep(Image.open(f)) for f in files]).to(dev)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            f = t2p.img_feat(m, x)
        v = F.normalize(F.normalize(f.float(), dim=-1).mean(0), dim=0).cpu().numpy()
        a = json.loads(r["attrs_json"] or "{}")
        ids.append(r["id"]); vecs.append(v.tolist())
        pays.append({"track_row": r["id"], "chunk_id": r["chunk_id"], "camera": r["camera"],
                     "start_s": r["start_s"], "end_s": r["end_s"], **a})
        if len(ids) >= 256:
            vector_db.upsert("person_ft", ids, vecs, pays); n += len(ids); ids, vecs, pays = [], [], []
    if ids:
        vector_db.upsert("person_ft", ids, vecs, pays); n += len(ids)
    print("person_ft indexed:", n)

if __name__ == "__main__":
    main()
