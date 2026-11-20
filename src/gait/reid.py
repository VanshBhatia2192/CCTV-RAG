"""Same person on other cameras: clothing-invariant Re-ID fusing appearance (person_ft, Cosine)
and 2D skeleton pose-guided gait biometrics (person_gait, Cosine).

Modes:
- 'any': Appearance-heavy (0.80) + Skeleton Gait Biometric refinement (0.20)
- 'cloth_change': 100% 2D Skeleton Gait Biometrics (completely clothing-invariant, zero appearance reliance)
- 'appearance_only': 100% Visual Appearance
"""

import argparse
import random
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
from qdrant_client import models
from src.index import vector_db

MODES = {"any": 0.80, "cloth_change": 0.0, "appearance_only": 1.0}
FIELDS = ["camera", "chunk_id", "start_s", "end_s", "motion", "cadence_bpm", "gait_symmetry", "step_length_norm"]
WALK = {"walking", "running"}


def _vec(coll: str, tid: int):
    try:
        r = vector_db.client().retrieve(coll, ids=[tid], with_vectors=True)
        return r[0].vector if r else None
    except Exception:
        return None


def _all_scores(coll: str, vec):
    if vec is None:
        return {}
    c = vector_db.client()
    try:
        n = c.count(coll, exact=True).count
        if n == 0:
            return {}
        pts = c.query_points(
            coll,
            query=vec,
            limit=n,
            with_payload=FIELDS,
            search_params=models.SearchParams(exact=True),
        ).points
        return {p.id: (float(p.score), p.payload) for p in pts}
    except Exception:
        return {}


def _z(d: Dict[int, float]) -> Dict[int, float]:
    if not d:
        return {}
    v = np.array(list(d.values()), dtype=np.float64)
    mu, sd = v.mean(), v.std() + 1e-9
    return {i: (s - mu) / sd for i, s in d.items()}


def rank(track_row: int, k: int = 20, mode: str = "any", w_app: Optional[float] = None, same_camera: bool = False):
    w = MODES.get(mode, 0.80) if w_app is None else float(w_app)
    qa = _vec("person_ft", track_row) or _vec("person", track_row)
    if qa is None:
        raise KeyError(f"track {track_row} has no appearance vector")

    A = _all_scores("person_ft", qa) or _all_scores("person", qa)
    if not A or track_row not in A:
        raise KeyError(f"track {track_row} not found in visual index")

    cam_q = A[track_row][1]["camera"]
    q_moving = A[track_row][1].get("motion") in WALK

    # 2D Skeleton ST-GCN gait vector
    qg = _vec("person_gait", track_row) or _vec("person_pose_gait", track_row)
    G = {}
    if qg is not None and q_moving:
        G = _all_scores("person_gait", qg) or _all_scores("person_pose_gait", qg)
        G = {i: v for i, v in G.items() if i == track_row or v[1].get("motion") in WALK}

    note = None
    if qg is None:
        w, note = 1.0, "Query track has no skeleton gait vector: visual appearance only"
    elif not q_moving:
        qg, w = None, 1.0
        note = "Query person is stationary; gait biometrics only apply when walking or running"

    za = _z({i: s for i, (s, _) in A.items() if i != track_row})
    zg = {}
    if G and track_row in G:
        others_g = [s for i, (s, _) in G.items() if i != track_row]
        if others_g:
            zg = _z({i: s for i, (s, _) in G.items() if i != track_row})

    rows = []
    cand_ids = set(za) | set(zg)
    for i in cand_ids:
        pay = (A.get(i) or G.get(i))[1]
        if not same_camera and pay["camera"] == cam_q:
            continue
        a = za.get(i)
        g = zg.get(i)

        if w == 0.0 and g is None:
            continue

        score = w * (a if a is not None else 0.0) + (1.0 - w) * (g if g is not None else 0.0)
        rows.append(dict(
            track_row=int(i),
            camera=pay["camera"],
            chunk_id=pay["chunk_id"],
            start_s=pay["start_s"],
            end_s=pay["end_s"],
            score=round(float(score), 3),
            appearance_z=None if a is None else round(float(a), 2),
            gait_z=None if g is None else round(float(g), 2),
            pose_z=None if g is None else round(float(g), 2),
            cadence_bpm=pay.get("cadence_bpm"),
            gait_symmetry=pay.get("gait_symmetry"),
            step_length_norm=pay.get("step_length_norm"),
        ))

    rows.sort(key=lambda r: -r["score"])
    return dict(
        query=dict(
            track_row=track_row,
            camera=cam_q,
            has_skeleton_gait=qg is not None,
        ),
        mode=mode,
        w_appearance=w,
        note=note,
        results=rows[:k],
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("track_id", type=int, help="Track ID to search")
    parser.add_argument("--mode", default="cloth_change", choices=list(MODES))
    parser.add_argument("--k", type=int, default=10)
    args = parser.parse_args()

    r = rank(args.track_id, k=args.k, mode=args.mode)
    print(f"Re-ID Mode: {r['mode']} (Appearance weight: {r['w_appearance']})")
    if r["note"]:
        print(f"Note: {r['note']}")
    for hit in r["results"]:
        print(f"Track {hit['track_row']:5d} | Cam: {hit['camera']:10s} | Score: {hit['score']:.3f} | Gait Z: {hit['gait_z']} | Cadence: {hit['cadence_bpm']} BPM")
