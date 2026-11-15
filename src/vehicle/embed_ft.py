import json, sqlite3
from pathlib import Path
import torch, torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from src.vehicle import t2v
from src.index import vector_db
from src.index.sql_db import connect

def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = "models/finetuned/text2vehicle/clip_t2v.pt"
    if not Path(ckpt).exists():
        raise SystemExit(f"Checkpoint not found at {ckpt}. Please run training.vehicle.train first.")
    m, _ = t2v.load(ckpt, dev)
    m.eval()
    db_path = Path("data/indexes/sqlite/meta.db").resolve()
    try:
        db = sqlite3.connect(f"file:{db_path}?mode=ro&nolock=1", uri=True)
    except Exception:
        db = connect()
    db.row_factory = sqlite3.Row
    rows = db.execute("SELECT t.*, c.camera FROM tracks t JOIN chunks c ON c.id=t.chunk_id WHERE t.kind='vehicle'").fetchall()
    ids, vecs, pays, n = [], [], [], 0
    for r in tqdm(rows, desc="vehicle_ft embed"):
        files = sorted(Path(r["crops_dir"]).glob("*.jpg"))
        if not files:
            continue
        # Use up to 8 frames per vehicle track
        x = torch.stack([t2v.prep(Image.open(f)) for f in files[:8]]).to(dev)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16 if dev == "cuda" else torch.float32):
            f = t2v.img_feat(m, x)
        v = F.normalize(F.normalize(f.float(), dim=-1).mean(0), dim=0).cpu().numpy()
        a = json.loads(r["attrs_json"] or "{}")
        ids.append(r["id"])
        vecs.append(v.tolist())
        pays.append({"track_row": r["id"], "chunk_id": r["chunk_id"], "camera": r["camera"],
                     "start_s": r["start_s"], "end_s": r["end_s"], **a})
        if len(ids) >= 256:
            vector_db.upsert("vehicle_ft", ids, vecs, pays)
            n += len(ids)
            ids, vecs, pays = [], [], []
    if ids:
        vector_db.upsert("vehicle_ft", ids, vecs, pays)
        n += len(ids)
    print("vehicle_ft indexed:", n)

if __name__ == "__main__":
    main()

