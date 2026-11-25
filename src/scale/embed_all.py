import argparse, json
from pathlib import Path
import torch, torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from transformers import CLIPImageProcessor, CLIPModel
from src.common import attributes, motion
from src.index import vector_db
from src.index.sql_db import connect
from src.person import t2p

NAME = "openai/clip-vit-base-patch32"
FT = Path("models/finetuned/text2person/clip_t2p.pt")

class DS(torch.utils.data.Dataset):
    def __init__(s, rows): s.rows, s.ip = rows, None
    def __len__(s): return len(s.rows)
    def __getitem__(s, i):
        if s.ip is None:
            s.ip = CLIPImageProcessor.from_pretrained(NAME)
        r, ims = s.rows[i], []
        for f in sorted(Path(r["crops_dir"]).glob("*.jpg")):
            try: ims.append(Image.open(f).convert("RGB"))
            except Exception: pass
        if not ims:
            return r["id"], None, None
        zs = s.ip(images=ims, return_tensors="pt")["pixel_values"]
        ft = torch.stack([t2p.prep(im) for im in ims]) if r["kind"] == "person" else None
        return r["id"], zs, ft

def main(limit, workers):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    db = connect()
    rows = [dict(r) for r in db.execute(
        "SELECT t.id, t.kind, t.chunk_id, t.start_s, t.end_s, t.fps, t.crops_dir, t.path_json, c.camera, "
        "coalesce(c.dataset, 'meva') AS dataset FROM tracks t "
        "JOIN chunks c ON c.id=t.chunk_id WHERE t.attrs_json IS NULL LIMIT ?", (limit,)).fetchall()]
    print(len(rows), "tracks to embed on", dev, flush=True)
    if not rows:
        return
    by = {r["id"]: r for r in rows}
    zm = CLIPModel.from_pretrained(NAME).to(dev).eval()
    fm = None
    if FT.exists() and any(r["kind"] == "person" for r in rows):
        fm, _ = t2p.load(str(FT), dev); fm.eval()
    slim = [{"id": r["id"], "kind": r["kind"], "crops_dir": r["crops_dir"]} for r in rows]
    dl = torch.utils.data.DataLoader(DS(slim), batch_size=None, num_workers=workers, prefetch_factor=4)
    buf = {"vehicle": ([], [], []), "person": ([], [], []), "person_ft": ([], [], [])}
    done = []
    def flush():
        for k, (ids, vs, ps) in buf.items():
            if ids:
                vector_db.upsert(k, ids, vs, ps)
            ids.clear(); vs.clear(); ps.clear()
        for tid, a in done:
            db.execute("UPDATE tracks SET attrs_json=? WHERE id=?", (json.dumps(a), tid))
        db.commit(); done.clear()
    for tid, zs, ft in tqdm(dl, total=len(slim), mininterval=30):
        r = by[tid]
        if zs is None:
            done.append((tid, {"no_crops": True})); continue
        with torch.no_grad():
            f = zm.visual_projection(zm.vision_model(pixel_values=zs.to(dev)).pooler_output)
            v = F.normalize(F.normalize(f.float(), dim=-1).mean(0), dim=0).cpu().numpy()
            vf = None
            if fm is not None and ft is not None:
                with torch.autocast(dev, dtype=torch.bfloat16, enabled=(dev == "cuda")):
                    g = t2p.img_feat(fm, ft.to(dev))
                vf = F.normalize(F.normalize(g.float(), dim=-1).mean(0), dim=0).cpu().numpy()
        kind = r["kind"]
        a = attributes.zero_shot(kind, v)
        a.update((motion.person_motion if kind == "person" else motion.vehicle_motion)(json.loads(r["path_json"]), r["fps"]))
        pay = {"track_row": tid, "chunk_id": r["chunk_id"], "camera": r["camera"], "dataset": r["dataset"], "start_s": r["start_s"], "end_s": r["end_s"], **a}
        ids, vs, ps = buf[kind]; ids.append(tid); vs.append(v.tolist()); ps.append(pay)
        if vf is not None:
            ids, vs, ps = buf["person_ft"]; ids.append(tid); vs.append(vf.tolist()); ps.append(pay)
        done.append((tid, a))
        if len(done) >= 128:
            flush()
    flush(); print("embedded", len(rows), flush=True)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=10**7); ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args(); main(a.limit, a.workers)
