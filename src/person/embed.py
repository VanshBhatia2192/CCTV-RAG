import argparse, json, sqlite3
from pathlib import Path
import numpy as np
from tqdm import tqdm
from src.common import clip, attributes, motion
from src.index import vector_db
from src.index.sql_db import connect

def main(kind, limit):
    db = connect()
    try:
        db.execute("ALTER TABLE tracks ADD COLUMN caption TEXT")
        db.commit()
    except sqlite3.OperationalError:
        pass
    mot = motion.person_motion if kind == "person" else motion.vehicle_motion
    rows = db.execute("SELECT t.*, c.camera, coalesce(c.dataset, 'meva') as dataset FROM tracks t JOIN chunks c ON c.id=t.chunk_id "
                      "WHERE t.kind=? AND t.attrs_json IS NULL LIMIT ?", (kind, limit)).fetchall()
    for r in tqdm(rows, desc=f"{kind} embed"):
        imgs = sorted(Path(r["crops_dir"]).glob("*.jpg"))
        if not imgs:
            continue
        v = clip.embed_images(imgs).mean(0); v /= np.linalg.norm(v)
        attrs = attributes.zero_shot(kind, v)
        attrs.update(mot(json.loads(r["path_json"]), r["fps"]))
        payload = {"track_row": r["id"], "chunk_id": r["chunk_id"], "camera": r["camera"], "dataset": r["dataset"],
                   "start_s": r["start_s"], "end_s": r["end_s"], **attrs}
        vector_db.upsert(kind, [r["id"]], [v.tolist()], [payload])
        
        # skeleton gait vectors need consecutive frames: built separately by src/gait/index_gait.py

        # Populate descriptive caption from detected attributes if not already present
        r_dict = dict(r)
        cap = r_dict.get("caption")
        if not cap:
            if kind == "person":
                cap = f"a person wearing a {attrs.get('upper_color', '')} {attrs.get('upper_type', 'top')} and {attrs.get('lower_color', '')} pants, {attrs.get('motion', '')}"
                if attrs.get("bag") and attrs.get("bag") != "none":
                    cap += f", carrying a {attrs['bag']}"
            else:
                cap = f"a {attrs.get('color', '')} {attrs.get('type', 'vehicle')} {attrs.get('motion', '')}"
                if attrs.get("turn") and attrs.get("turn") != "none":
                    cap += f", turning {attrs['turn']}"
            cap += "."
            db.execute("UPDATE tracks SET attrs_json=?, caption=? WHERE id=?", (json.dumps(attrs), cap, r["id"]))
        else:
            db.execute("UPDATE tracks SET attrs_json=? WHERE id=?", (json.dumps(attrs), r["id"]))
        db.commit()

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", choices=["person", "vehicle"], default="person")
    ap.add_argument("--limit", type=int, default=500)
    a = ap.parse_args()
    main(a.kind, a.limit)
