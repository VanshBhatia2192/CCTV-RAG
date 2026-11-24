"""Same person on other cameras: fuse appearance (person_ft, cosine) and gait (person_gait, Euclidean) the way
eval/gait/eval_mevid.py scored them: z-score each modality over the whole gallery (all tracks except the query),
then w*appearance + (1-w)*gait, ranked over tracks from other cameras."""
import argparse, random
from pathlib import Path
import numpy as np
from qdrant_client import models
from src.index import vector_db

# weight on appearance. 'any': 0.8 (MEVID cross-camera R1/mAP 46.1/17.4 vs 41.3/16.1 at 0.6; the weight the
# cross-validated eval in eval/gait/eval_mevid.py picks), 'cloth_change': 0.0 (gait only, best on the cloth-changing
# split). Cadence / aspect-ratio gating was tested and does not help: eval/gait/eval_reid_cues.py.
MODES = {"any": 0.8, "cloth_change": 0.0}
FIELDS = ["camera", "chunk_id", "start_s", "end_s", "motion"]
WALK = {"walking", "running"}   # gait is most informative for people who are moving


def _vec(coll, tid):
    r = vector_db.client().retrieve(coll, ids=[tid], with_vectors=True)
    return r[0].vector if r else None


def _all_scores(coll, vec, limit=600):
    c = vector_db.client()
    n = c.count(coll, exact=False).count
    pts = c.query_points(coll, query=vec, limit=min(n, limit), with_payload=FIELDS).points
    return {p.id: (float(p.score), p.payload) for p in pts}


def _z(d):
    v = np.array(list(d.values()), dtype=np.float64)
    mu, sd = v.mean(), v.std() + 1e-9
    return {i: (s - mu) / sd for i, s in d.items()}


def rank(track_row, k=20, mode="any", w_app=None, same_camera=False):
    w = MODES[mode] if w_app is None else float(w_app)
    qa = _vec("person_ft", track_row)
    if qa is None:
        raise KeyError(f"track {track_row} has no appearance vector")
    qg, note = _vec("person_gait", track_row), None
    A = _all_scores("person_ft", qa)
    cam_q = A[track_row][1]["camera"]
    q_moving = A[track_row][1].get("motion") in WALK
    G = _all_scores("person_gait", qg) if qg is not None else {}
    G = {i: v for i, v in G.items() if i == track_row or v[1].get("motion") in WALK or not q_moving}
    if qg is None:
        w, note = 1.0, "query track has no gait vector (fewer than 12 clean frames): appearance only"
    elif not q_moving:
        note = "query person speed is low; gait biometrics active but silhouette dynamics may be reduced"
    za = _z({i: s for i, (s, _) in A.items() if i != track_row})
    zg = {}
    if G:
        others = {i: s for i, (s, _) in G.items() if i != track_row}
        if others:
            self_score = G[track_row][0] if track_row in G else 0.0
            med = float(np.median(list(others.values())))
            # Distance.EUCLID: closer match has lower distance, so self_score (0.0) < med.
            # Invert distance so higher is better for similarity z-scores.
            sign = 1.0 if self_score >= med else -1.0
            zg = _z({i: sign * s for i, s in others.items()})
    rows = []
    for i in set(za) | set(zg):
        pay = (A.get(i) or G.get(i))[1]
        if not same_camera and pay["camera"] == cam_q:
            continue
        a, g = za.get(i), zg.get(i)
        if w < 1.0 and g is None and w == 0.0:
            continue                                   # gait-only mode: skip tracks that have no gait vector
        if w < 1.0 and g is None:
            # Dual biometric mode, candidate has no gait vector: apply slight penalty so unconfirmed tracks don't outrank confirmed gait matches
            g = -0.5
        score = w * (a if a is not None else 0.0) + (1 - w) * (g if g is not None else 0.0)
        rows.append(dict(track_row=int(i), camera=pay["camera"], chunk_id=pay["chunk_id"], start_s=pay["start_s"],
                         end_s=pay["end_s"], score=round(float(score), 3), appearance_z=None if a is None else round(float(a), 2),
                         gait_z=None if i not in zg else round(float(zg[i]), 2)))
    rows.sort(key=lambda r: -r["score"])
    return dict(query=dict(track_row=track_row, camera=cam_q, has_gait=qg is not None), mode=mode, w_appearance=w, note=note, results=rows[:k])


def enrich(res):
    """Add thumbnail and video path from the SQLite index (needs the project's sql_db)."""
    from src.index.sql_db import connect
    db = connect()
    def info(tid):
        r = db.execute("select t.crops_dir, t.caption, ch.path as video from tracks t join chunks ch on ch.id=t.chunk_id where t.id=?", (tid,)).fetchone()
        files = sorted(Path(r["crops_dir"]).glob("*.jpg")) if r else []
        return dict(thumb=str(files[len(files) // 2]) if files else None, caption=r["caption"] if r else None, video=r["video"] if r else None)
    res["query"].update(info(res["query"]["track_row"]))
    for r in res["results"]:
        r.update(info(r["track_row"]))
    return res


def sheet(queries, out, k=6, mode="any", h=150):
    import cv2
    rows = []
    for q in queries:
        res = enrich(rank(q, k=k, mode=mode))
        tiles = []
        for lab, d in [("QUERY " + res["query"]["camera"], res["query"])] + [(f'{r["camera"]} {r["score"]:+.1f}', r) for r in res["results"]]:
            im = cv2.imread(d["thumb"]) if d.get("thumb") else None
            im = np.zeros((h, h // 2, 3), np.uint8) if im is None else cv2.resize(im, (max(1, int(im.shape[1] * h / im.shape[0])), h))
            im = cv2.copyMakeBorder(im, 18, 0, 2, 2, cv2.BORDER_CONSTANT, value=(40, 40, 40))
            cv2.putText(im, lab[:18], (2, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            tiles.append(im)
        rows.append(np.hstack(tiles))
    W = max(r.shape[1] for r in rows)
    cv2.imwrite(out, np.vstack([cv2.copyMakeBorder(r, 0, 4, 0, W - r.shape[1], cv2.BORDER_CONSTANT) for r in rows]))
    print("wrote", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--track", type=int, nargs="*"); ap.add_argument("--random", type=int, default=0)
    ap.add_argument("--k", type=int, default=6); ap.add_argument("--mode", default="any", choices=list(MODES))
    ap.add_argument("--out", default="outputs/reid_sheet.png"); ap.add_argument("--any-motion", action="store_true", help="--random may also pick standing people")
    a = ap.parse_args()
    qs = list(a.track or [])
    if a.random:
        pts, _ = vector_db.client().scroll("person_gait", limit=20000, with_payload=["motion"])
        pts = [p for p in pts if a.any_motion or (p.payload or {}).get("motion") in WALK]
        random.seed(0); qs += [p.id for p in random.sample(pts, min(a.random, len(pts)))]
    sheet(qs, a.out, a.k, a.mode)
