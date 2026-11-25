"""Embed indexed tracks with a fine-tuned CLIP image tower into a *_ft collection
(mean of the L2-normalised crop embeddings, re-normalised). Default: vehicles -> vehicle_ft."""
import argparse, json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import torch, torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from src.person import t2p
from src.index import vector_db
from src.index.sql_db import connect


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", default="vehicle"); ap.add_argument("--ckpt", default="models/finetuned/vehicle/clip_vehicle.pt")
    ap.add_argument("--collection", default="vehicle_ft"); ap.add_argument("--limit", type=int, default=0); ap.add_argument("--block", type=int, default=256)
    ap.add_argument("--dataset", default=None, help="only tracks of this dataset")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m, _ = t2p.load(a.ckpt, dev); m.eval(); db = connect()
    rows = db.execute("SELECT t.*, c.camera, coalesce(c.dataset, 'meva') as dataset FROM tracks t JOIN chunks c ON c.id=t.chunk_id WHERE t.kind=?" + (" AND coalesce(c.dataset, 'meva')=?" if a.dataset else ""),
                      (a.kind, a.dataset) if a.dataset else (a.kind,)).fetchall()
    if a.limit:
        rows = rows[:a.limit]
    print(f"{len(rows)} {a.kind} tracks -> {a.collection} | device {dev}", flush=True)
    pool = ThreadPoolExecutor(8); n = skipped = 0

    def load(r):
        files = sorted(Path(r["crops_dir"]).glob("*.jpg"))
        return [t2p.prep(Image.open(f)) for f in files]

    for i in tqdm(range(0, len(rows), a.block), desc=a.collection):
        blk = rows[i:i + a.block]; crops = list(pool.map(load, blk))
        keep = [(r, c) for r, c in zip(blk, crops) if c]
        skipped += len(blk) - len(keep)
        if not keep:
            continue
        x = torch.cat([torch.stack(c) for _, c in keep]); feats = []
        for j in range(0, len(x), 512):
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=(dev == "cuda")):
                feats.append(F.normalize(t2p.img_feat(m, x[j:j + 512].to(dev)).float(), dim=-1).cpu())
        feats = torch.cat(feats); off = 0; ids, vecs, pays = [], [], []
        for r, c in keep:
            v = F.normalize(feats[off:off + len(c)].mean(0), dim=0); off += len(c)
            ids.append(r["id"]); vecs.append(v.tolist())
            pays.append({"track_row": r["id"], "chunk_id": r["chunk_id"], "camera": r["camera"], "dataset": r["dataset"], "start_s": r["start_s"], "end_s": r["end_s"], **json.loads(r["attrs_json"] or "{}")})
        vector_db.upsert(a.collection, ids, vecs, pays); n += len(ids)
    print(f"{a.collection} indexed: {n} | tracks without crops skipped: {skipped}", flush=True)


if __name__ == "__main__":
    main()
