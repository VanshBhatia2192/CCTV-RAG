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
    rows = db.execute("SELECT t.*, c.camera FROM tracks t JOIN chunks c ON c.id=t.chunk_id "
                      "WHERE t.kind=? AND t.attrs_json IS NULL LIMIT ?", (kind, limit)).fetchall()
    for r in tqdm(rows, desc=f"{kind} embed"):
        imgs = sorted(Path(r["crops_dir"]).glob("*.jpg"))
        if not imgs:
            continue
        v = clip.embed_images(imgs).mean(0); v /= np.linalg.norm(v)
        attrs = attributes.zero_shot(kind, v)
        attrs.update(mot(json.loads(r["path_json"]), r["fps"]))
        payload = {"track_row": r["id"], "chunk_id": r["chunk_id"], "camera": r["camera"],
                   "start_s": r["start_s"], "end_s": r["end_s"], **attrs}
        vector_db.upsert(kind, [r["id"]], [v.tolist()], [payload])
        
        # Extract 2D skeleton pose gait if enough frames exist
        if kind == "person" and len(imgs) >= 8:
            try:
                import cv2
                from src.gait.pose_gait import encode_pose_gait
                im_arrays = [cv2.imread(str(f)) for f in imgs]
                im_arrays = [im for im in im_arrays if im is not None]
                if len(im_arrays) >= 8:
                    pose_emb, bio_metrics = encode_pose_gait(im_arrays, fps=r["fps"] or 25.0)
                    if pose_emb is not None:
                        attrs.update(bio_metrics)
                        payload.update(bio_metrics)
                        vector_db.upsert("person_gait", [r["id"]], [pose_emb.tolist()], [payload])
                        vector_db.upsert("person_pose_gait", [r["id"]], [pose_emb.tolist()], [payload])
            except Exception:
                pass

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
