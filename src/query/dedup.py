"""Collapse repeated sightings of the same person / vehicle into one result.

The tracker restarts at every 10-second chunk boundary, so someone who stays in view for three minutes is ~18 tracks,
and overlapping MEVA cameras film the same moment twice. Results are grouped (union-find over the result list):
  * same camera: not visible at the same time, within SAME_CAM_S, the later track starts where the earlier one was
    (spatial continuity), and appearance similarity >= THR[kind]["same"]
  * different cameras (people only): within CROSS_CAM_S and similarity >= THR["person"]["cross"]
Groups never cross kind or dataset. The best-ranked card represents the group; the rest go into card["sightings"].

Why the extra cues: appearance alone does not separate identities well enough. Measured on the live index (same track
split at a chunk boundary = same identity, 1,500 pairs; two tracks visible at the same moment in one chunk = different,
20,000 pairs): people at cosine 0.88 keep 86% of true repeats but also merge 7% of different-person pairs; vehicles at
0.94 merge 15% of different-vehicle pairs (parked rows, same models). Two tracks visible at the same time are never one
identity, and a lingering person / parked car stays in place, so those two checks carry the same-camera case, and
vehicles are not folded across cameras at all.
"""
import datetime as dt
import numpy as np

SAME_CAM_S = 900.0          # lingering: someone sat at one table for 9+ minutes on school.G509
CROSS_CAM_S = 15.0
NEAR = 1.0               # same camera: centres at most this many box heights apart
THR = {"person": {"same": 0.85, "cross": 0.94}, "vehicle": {"same": 0.95, "cross": None}}
COLL = {"person": "person_ft", "vehicle": "vehicle_ft"}
KEEP = ("id", "camera", "recorded_at", "start_s", "end_s", "clip", "clip_t0", "clip_t1", "thumb", "caption", "is_full_video",
        "kind", "dataset", "attrs")


def _vectors(kind, ids):
    from src.index import vector_db
    out = {}
    ids = list(ids)
    for coll in (COLL.get(kind), kind):
        if not coll or not ids:
            continue
        try:
            for s in range(0, len(ids), 500):
                for p in vector_db.client().retrieve(coll, ids=ids[s:s + 500], with_vectors=True):
                    if p.vector and p.id not in out:
                        v = np.asarray(p.vector, np.float32)
                        out[p.id] = v / (np.linalg.norm(v) + 1e-9)
        except Exception:
            pass
        ids = [i for i in ids if i not in out]
    return out


def _clock(card, sources):
    """(time-domain key, seconds) so sightings are only compared on one clock: wall clock when the file name has a
    timestamp (MEVA, comparable across cameras), otherwise position inside the source video (comparable within it)."""
    if card.get("recorded_at"):
        t = dt.datetime.fromisoformat(card["recorded_at"]).timestamp()
        return ("wall", card.get("dataset")), t
    return ("video", sources.get(card["id"])), float(card.get("start_s") or 0.0)


def _near(ma, mb):
    """Spatial continuity on one camera: the later track begins where the earlier one ended (box centres within NEAR
    box heights). Unknown boxes do not block a merge."""
    if not ma or not mb or not ma.get("last") or not mb.get("first"):
        return True
    (ax, ay, ah), (bx, by, bh) = ma["last"], mb["first"]
    return np.hypot(ax - bx, ay - by) <= NEAR * max(ah, bh, 1.0)


def group(cards, meta=None, thr=None):
    """Return the cards with duplicates folded in: each kept card gets "sightings" (list of the others, best first)
    and "n_sightings" (group size). meta: {track id: {"source", "first", "last"}} from meta_of(). Incident cards and
    kinds without vectors pass through unchanged."""
    meta = meta or {}
    sources = {i: m.get("source") for i, m in meta.items()}
    kinds = {c.get("kind") for c in cards} & set(COLL)
    vec = {}
    for k in kinds:
        vec.update(_vectors(k, [c["id"] for c in cards if c.get("kind") == k]))
    parent = list(range(len(cards)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    clocks = [_clock(c, sources) for c in cards]
    for i in range(len(cards)):
        a = cards[i]
        if a.get("kind") not in COLL or a["id"] not in vec:
            continue
        th = (thr or THR)[a["kind"]]
        for j in range(i + 1, len(cards)):
            b = cards[j]
            if b.get("kind") != a.get("kind") or b.get("dataset") != a.get("dataset") or b["id"] not in vec:
                continue
            (da, ta), (db_, tb) = clocks[i], clocks[j]
            if da != db_:
                continue
            same_cam = a.get("camera") == b.get("camera")
            dur_a = max(0.0, float(a.get("end_s") or 0) - float(a.get("start_s") or 0))
            dur_b = max(0.0, float(b.get("end_s") or 0) - float(b.get("start_s") or 0))
            if same_cam:
                first, second = ((a, ta, dur_a), (b, tb, dur_b)) if ta <= tb else ((b, tb, dur_b), (a, ta, dur_a))
                gap = second[1] - (first[1] + first[2])
                if gap < -0.5 or gap > SAME_CAM_S:       # visible at the same time -> two identities; or too far apart
                    continue
                if not _near(meta.get(first[0]["id"]), meta.get(second[0]["id"])):
                    continue
                need = th["same"]
            else:
                if th["cross"] is None or abs(ta - tb) > CROSS_CAM_S:
                    continue
                need = th["cross"]
            if float(vec[a["id"]] @ vec[b["id"]]) >= need:
                parent[find(j)] = find(i)
    out, members = [], {}
    for i, c in enumerate(cards):
        members.setdefault(find(i), []).append(c)
    for i, c in enumerate(cards):
        if find(i) != i:
            continue
        rest = members[i][1:]
        c = dict(c)
        c["sightings"] = [{k: s.get(k) for k in KEEP} for s in rest]
        c["n_sightings"] = 1 + len(rest)
        out.append(c)
    return out


def meta_of(db, cards):
    """Per track card: source video (the clock for footage without timestamps) and first / last box centre and height
    (path_json rows are [frame, cx, cy, h, w])."""
    import json
    ids = [c["id"] for c in cards if c.get("kind") in COLL]
    if not ids:
        return {}
    q = ("SELECT t.id, ch.source, t.path_json FROM tracks t JOIN chunks ch ON ch.id=t.chunk_id WHERE t.id IN (%s)"
         % ",".join("?" * len(ids)))
    out = {}
    for tid, src, pj in db.execute(q, ids):
        path = json.loads(pj or "[]")
        box = lambda r: (float(r[1]), float(r[2]), float(r[3])) if r and len(r) >= 4 else None
        out[tid] = {"source": src, "first": box(path[0]) if path else None, "last": box(path[-1]) if path else None}
    return out
