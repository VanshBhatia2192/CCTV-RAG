"""Build `person_gait` vectors (256-d, cosine) for indexed person tracks with the trained skeleton encoder.

For each person track, re-read up to K windows of 64 consecutive frames from its chunk video (gait needs
consecutive frames; the saved crops_dir only holds a few sampled crops), run ViTPose on the crops, embed
each window with the ST-GCN encoder (src/gait/pose_gait.py) and store the mean. Interpretable kinematics
(cadence, symmetry, ...) go into the Qdrant payload and the track's attrs_json.

    python -m src.gait.index_gait --recreate                 # all tracks
    python -m src.gait.index_gait --part 0/4 --skip-existing # one of 4 parallel workers
"""
import argparse, json, time
from collections import defaultdict
import numpy as np
import torch
from tqdm import tqdm

from src.gait import pose_gait as pg
from src.gait.track_crops import read_crops
from src.index import vector_db
from src.index.sql_db import connect

COLL = "person_gait"


def existing_ids():
    c, have, off = vector_db.client(), set(), None
    while True:
        pts, off = c.scroll(COLL, limit=5000, offset=off, with_payload=False, with_vectors=False)
        have.update(p.id for p in pts)
        if off is None:
            return have


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--recreate", action="store_true")
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--part", default="0/1", help="i/N: this worker's share of the chunks")
    ap.add_argument("--K", type=int, default=3, help="windows of 64 consecutive frames per track")
    ap.add_argument("--fmt", default="cxcyhw", help="path_json box layout: [frame, cx, cy, h, w]")
    a = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = pg.load_encoder(dev)
    if model is None:
        raise SystemExit("no trained skeleton gait weights; run training/gait/train_pose.py first")
    if a.recreate:
        vector_db.recreate(COLL)
    db = connect()
    rows = [dict(r) for r in db.execute(
        "SELECT t.id, t.chunk_id, t.fps, t.path_json, t.attrs_json, t.start_s, t.end_s, ch.camera, ch.path AS vpath "
        "FROM tracks t JOIN chunks ch ON ch.id=t.chunk_id WHERE t.kind='person' ORDER BY t.chunk_id, t.id")]
    by_chunk = defaultdict(list)
    for r in rows:
        by_chunk[r["chunk_id"]].append(r)
    pi, pn = (int(x) for x in a.part.split("/"))
    chunks = sorted(by_chunk)[pi::pn]
    have = existing_ids() if a.skip_existing and not a.recreate else set()
    todo = [(c, [r for r in by_chunk[c] if r["id"] not in have]) for c in chunks]
    todo = [(c, rs) for c, rs in todo if rs]
    if a.limit:
        todo = todo[:a.limit]
    print(f"{sum(len(rs) for _, rs in todo)} person tracks in {len(todo)} chunks | device {dev}", flush=True)

    n_ok = n_skip = 0
    t0 = time.time()
    for cid, rs in tqdm(todo, desc="gait"):
        pjs = [json.loads(r["path_json"] or "[]") for r in rs]
        try:
            crops = read_crops(rs[0]["vpath"], pjs, K=a.K, fmt=a.fmt)
        except Exception as e:
            print(f"chunk {cid}: cannot read video ({e})", flush=True); n_skip += len(rs); continue
        ids, vecs, pays = [], [], []
        for ti, r in enumerate(rs):
            wins = [crops[(ti, wi)] for wi in range(a.K) if len(crops.get((ti, wi), [])) >= pg.MIN_FRAMES]
            if not wins:
                n_skip += 1; continue
            kp = [pg.extract_keypoints_from_images(w, device=dev) for w in wins]
            es = [e for e in (pg.embed_keypoints(k, model=model) for k in kp) if e is not None]
            if not es:
                n_skip += 1; continue
            e = np.mean(es, 0); e /= np.linalg.norm(e)
            kin = pg.compute_kinematic_gait_metrics(max(kp, key=len), fps=float(r["fps"] or 25.0))
            attrs = json.loads(r["attrs_json"] or "{}")
            attrs.update({k: kin[k] for k in ("cadence_bpm", "gait_symmetry", "step_length_norm")})
            db.execute("UPDATE tracks SET attrs_json=? WHERE id=?", (json.dumps(attrs), r["id"]))
            ids.append(r["id"]); vecs.append(e.tolist())
            pays.append(dict(track_row=r["id"], chunk_id=cid, camera=r["camera"], start_s=r["start_s"], end_s=r["end_s"],
                             motion=attrs.get("motion"), **kin))
        if ids:
            vector_db.upsert(COLL, ids, vecs, pays); db.commit(); n_ok += len(ids)
    print(f"indexed {n_ok} tracks into '{COLL}' ({n_skip} skipped: too short / unreadable) in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
