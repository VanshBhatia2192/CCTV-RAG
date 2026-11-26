"""Vehicle re-identification on the live AICITY index, scored against CityFlow (AI City 2022 Track 1) ground truth.

Each indexed AICITY vehicle track is matched to the CityFlow MTMC ground truth (gt/gt.txt: frame, global vehicle id,
box) frame by frame (IoU >= 0.5); a track gets the majority id when it matches on at least 3 frames and half of them.
Queries are labelled tracks whose vehicle also appears on another camera; the gallery is every labelled AICITY track
on other cameras. R@1 / R@5 / mAP per variant:

  api            src/vehicle/reid.rank() as the UI calls it (+-60 s window on the synchronised clock); all S02 queries
                 and a sample of the other scenes
  api_color_gate / api_no_time_window   the same with the colour gate on / without the time window
  cos_ft         fine-tuned vehicle CLIP (vehicle_ft) cosine, whole gallery
  cos_ft_scene   ... only the query's recording session (CityFlow scene)
  cos_ft_win{T}  ... same scene and within T seconds on the synchronised clock (cam_timestamp offsets)
  cos_base_*     zero-shot CLIP (vehicle collection) for comparison

vehicle_ft was fine-tuned on CityFlow-NL with scene S02 held out, so S02 is the clean test; other scenes are reported
separately (the model may have seen those vehicles, though with a text-image loss, not an identity loss).

    QDRANT_URL=http://localhost:6335 python -m eval.vehicle.eval_reid_cityflow [--api-queries 150]
"""
import argparse, json, random, re, time
from collections import Counter, defaultdict
from pathlib import Path
import cv2
import numpy as np
from src.index.sql_db import connect
from src.index import vector_db

ROOT = Path("data/raw/cityflow_nl/AICity22_Track1_MTMC_Tracking")


def iou(a, b):
    ax0, ay0, ax1, ay1 = a; bx0, by0, bx1, by1 = b
    iw, ih = max(0, min(ax1, bx1) - max(ax0, bx0)), max(0, min(ay1, by1) - max(ay0, by0))
    inter = iw * ih
    return inter / max(1e-9, (ax1 - ax0) * (ay1 - ay0) + (bx1 - bx0) * (by1 - by0) - inter)


def dims(path):
    cap = cv2.VideoCapture(str(path)); w, h = cap.get(3), cap.get(4); cap.release()
    return w, h


def label_tracks(db):
    rows = [dict(r) for r in db.execute(
        "SELECT t.id, t.path_json, t.fps, t.start_s, t.end_s, ch.camera, ch.source, ch.path AS chunk_path, ch.start_s AS cs "
        "FROM tracks t JOIN chunks ch ON ch.id=t.chunk_id WHERE t.kind='vehicle' AND ch.dataset='aicity'")]
    by_src = defaultdict(list)
    for r in rows:
        by_src[r["source"]].append(r)
    offsets = {}
    for f in (ROOT / "cam_timestamp").glob("*.txt"):
        for line in open(f):
            cam, off = line.split()
            offsets[(f.stem, cam)] = float(off)
    lab, meta = {}, {}
    for src, rs in by_src.items():
        m = re.search(r"(S\d\d)/(c\d\d\d)/vdo", src)
        scene, cam = m.group(1), m.group(2)
        gt = defaultdict(list)
        gtf = Path(src).parent / "gt" / "gt.txt"
        if gtf.exists():
            for line in open(gtf):
                fr, gid, l, t, w, h = map(float, line.split(",")[:6])
                gt[int(fr)].append((int(gid), (l, t, l + w, t + h)))
        ow, oh = dims(src); cw, chh = dims(rs[0]["chunk_path"])
        sx, sy = ow / cw, oh / chh
        for r in rs:
            meta[r["id"]] = dict(scene=scene, cam=cam, t0=r["start_s"] + offsets.get((scene, cam), 0.0),
                                 t1=r["end_s"] + offsets.get((scene, cam), 0.0))
            if not gt:
                continue
            votes, n = Counter(), 0
            for f, cx, cy, bh, bw in json.loads(r["path_json"] or "[]"):
                t = r["cs"] + f / (r["fps"] or 10.0)
                gf = int(round(t * 10.0)) + 1
                box = ((cx - bw / 2) * sx, (cy - bh / 2) * sy, (cx + bw / 2) * sx, (cy + bh / 2) * sy)
                best = max(((iou(box, b), gid) for gid, b in gt.get(gf, [])), default=(0, None))
                n += 1
                if best[0] >= 0.5:
                    votes[best[1]] += 1
            if votes:
                gid, c = votes.most_common(1)[0]
                if c >= 3 and c >= 0.5 * sum(votes.values()):
                    lab[r["id"]] = (scene, gid)          # CityFlow ids are global across a scene's cameras
    return lab, meta


def vectors(coll, ids):
    out = {}
    ids = list(ids)
    for s in range(0, len(ids), 1000):
        for p in vector_db.client().retrieve(coll, ids=ids[s:s + 1000], with_vectors=True):
            v = np.asarray(p.vector, np.float32); out[p.id] = v / (np.linalg.norm(v) + 1e-9)
    return out


def metrics(ranked_lists):
    """ranked_lists: [(ranked candidate ids, set of positives)] -> R1, R5, mAP (positives not returned count as misses)."""
    r1 = r5 = ap = 0.0; n = 0
    for ranked, pos in ranked_lists:
        if not pos:
            continue
        n += 1
        hits = [i for i, c in enumerate(ranked) if c in pos]
        r1 += bool(hits and hits[0] < 1); r5 += bool(hits and hits[0] < 5)
        ap += sum((k + 1) / (h + 1) for k, h in enumerate(hits)) / len(pos)
    return dict(n=n, R1=round(100 * r1 / max(1, n), 1), R5=round(100 * r5 / max(1, n), 1), mAP=round(100 * ap / max(1, n), 1))


def main(a):
    db = connect()
    t = time.time()
    lab, meta = label_tracks(db)
    print(f"labelled {len(lab)} of {len(meta)} AICITY vehicle tracks in {time.time() - t:.0f}s", flush=True)
    ids = sorted(lab)
    pos = {i: {j for j in ids if j != i and lab[j] == lab[i] and meta[j]["cam"] != meta[i]["cam"]} for i in ids}
    queries = [i for i in ids if pos[i]]
    print(f"queries with the same vehicle on another camera: {len(queries)}", flush=True)
    V = {"ft": vectors("vehicle_ft", ids), "base": vectors("vehicle", ids)}
    rep = {"labelled_tracks": len(lab), "queries": len(queries), "variants": {}}

    def run(name, score_fn, keep_fn):
        out = {}
        for split, qs in (("S02_clean", [q for q in queries if meta[q]["scene"] == "S02"]),
                          ("other_scenes", [q for q in queries if meta[q]["scene"] != "S02"])):
            lists = []
            for q in qs:
                cand = [j for j in ids if meta[j]["cam"] != meta[q]["cam"] and keep_fn(q, j)]
                sc = score_fn(q, cand)
                ranked = [c for _, c in sorted(zip(sc, cand), key=lambda x: -x[0])]
                lists.append((ranked, pos[q]))
            out[split] = metrics(lists)
        rep["variants"][name] = out
        print(f"  {name:18s} " + " | ".join(f"{k}: R1 {v['R1']:5.1f} R5 {v['R5']:5.1f} mAP {v['mAP']:5.1f} (n={v['n']})" for k, v in out.items()), flush=True)

    for vk in ("ft", "base"):
        M = V[vk]
        cos = lambda q, cand, M=M: [float(M[q] @ M[c]) if q in M and c in M else -1.0 for c in cand]
        run(f"cos_{vk}", cos, lambda q, j: True)
        run(f"cos_{vk}_scene", cos, lambda q, j: meta[j]["scene"] == meta[q]["scene"])
        if vk == "ft":
            for T in a.windows:
                def win(q, j, T=T):
                    if meta[j]["scene"] != meta[q]["scene"]:
                        return False
                    gap = max(meta[j]["t0"] - meta[q]["t1"], meta[q]["t0"] - meta[j]["t1"], 0.0)
                    return gap <= T
                run(f"cos_ft_win{T}", cos, win)

    from src.vehicle import reid as vreid
    info = {r[0]: (json.loads(r[1] or "{}"), r[2]) for r in db.execute(
        "SELECT id, attrs_json, caption FROM tracks WHERE id IN (%s)" % ",".join(map(str, ids)))}
    col = {i: vreid.extract_color(*info[i]) for i in ids}
    typ = {i: vreid.extract_type(*info[i]) for i in ids}
    M = V["ft"]
    cos = lambda q, cand: [float(M[q] @ M[c]) if q in M and c in M else -1.0 for c in cand]
    W = a.gate_window

    def win(q, j):
        if meta[j]["scene"] != meta[q]["scene"]:
            return False
        return max(meta[j]["t0"] - meta[q]["t1"], meta[q]["t0"] - meta[j]["t1"], 0.0) <= W
    run(f"win{W}+color_gate", cos, lambda q, j: win(q, j) and not (col[q] and col[j] and col[q] != col[j]))
    run(f"win{W}+type_gate", cos, lambda q, j: win(q, j) and not (typ[q] and typ[j] and typ[q] != typ[j]))
    run(f"win{W}+color_penalty", lambda q, cand: [s - (0.1 if col[q] and col[c] and col[q] != col[c] else 0.0)
                                                  for s, c in zip(cos(q, cand), cand)], win)
    if a.skip_api:
        Path(a.report).write_text(json.dumps(rep, indent=1)); print("wrote", a.report); return
    random.seed(0)
    s02 = [q for q in queries if meta[q]["scene"] == "S02"]           # the clean scene in full, a sample of the rest
    rest = [q for q in queries if meta[q]["scene"] != "S02"]
    sample = s02 + random.sample(rest, min(a.api_queries, len(rest)))
    for name, kw in (("api", {}), ("api_color_gate", dict(use_color_gate=True)), ("api_no_time_window", dict(use_st_filter=False))):
        out, lat = {}, []
        for split in ("S02_clean", "other_scenes"):
            lists = []
            for q in [q for q in sample if (meta[q]["scene"] == "S02") == (split == "S02_clean")]:
                t = time.time()
                res = vreid.rank(q, k=500, **kw)["results"]
                lat.append(time.time() - t)
                ranked = [r["track_row"] for r in res if r["track_row"] in lab]
                lists.append((ranked, pos[q]))
            out[split] = metrics(lists)
        out["median_latency_s"] = round(float(np.median(lat)), 2) if lat else None
        rep["variants"][name] = out
        print(f"  {name:18s} " + " | ".join(f"{k}: R1 {v['R1']:5.1f} R5 {v['R5']:5.1f} mAP {v['mAP']:5.1f} (n={v['n']})"
                                            for k, v in out.items() if isinstance(v, dict)) + f"  {out['median_latency_s']}s", flush=True)
    Path(a.report).write_text(json.dumps(rep, indent=1)); print("wrote", a.report)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--api-queries", type=int, default=150)
    ap.add_argument("--windows", type=int, nargs="+", default=[60, 180, 600])
    ap.add_argument("--gate-window", type=int, default=60)
    ap.add_argument("--skip-api", action="store_true")
    ap.add_argument("--report", default="outputs/reports/eval_vehicle_reid_cityflow.json")
    main(ap.parse_args())
