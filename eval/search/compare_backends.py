"""Qdrant vs ResonanceDB on the same embeddings.

A. MEVID ground truth (identity labels): cross-camera person re-ID with the fine-tuned (app_ft) and zero-shot
   (app_zs) CLIP appearance vectors, eval_mevid protocol (same-id same-camera gallery items are junk; the cc split
   also drops same-outfit). Reference = exact cosine ranking (what Qdrant approximates); ResonanceDB ranks the
   gallery from a temporary corpus.
B. Live index agreement: top-10 overlap between the two backends for stored vectors used as queries, plus latency.
C. End to end: contextual.search on the 24-query set (full@10 by the constraint checker) with each backend.

    QDRANT_URL=http://localhost:6335 RESONANCE_URL=http://localhost:31415 DEMO_DB=/tmp/meta_demo_$USER.db \
        python -m eval.search.compare_backends
"""
import argparse, json, os, time
from pathlib import Path
import numpy as np
from src.common.config import PATHS
if os.environ.get("DEMO_DB"):
    PATHS["sqlite"] = os.environ["DEMO_DB"]
from src.index import vector_db, resonance


def mevid_rank(Sm, z, qi, gi, mode):
    pid, cam, outfit = z["pid"], z["cam"], z["outfit"]
    res = []
    for j, q in enumerate(qi):
        same = pid[gi] == pid[q]
        junk = same & (cam[gi] == cam[q])
        if mode == "cc":
            junk = junk | (same & (outfit[gi] == outfit[q]))
        pos = same & ~junk
        if not pos.any():
            continue
        keep = ~junk
        hit = pos[keep][np.argsort(-Sm[j][keep], kind="stable")]
        r = np.where(hit)[0]
        res.append([r[0] < 1, r[0] < 5, np.mean((np.arange(len(r)) + 1) / (r + 1))])
    r = np.array(res, float)
    return dict(n=len(r), R1=round(100 * r[:, 0].mean(), 1), R5=round(100 * r[:, 1].mean(), 1), mAP=round(100 * r[:, 2].mean(), 1))


def part_a(cache):
    z = np.load(cache)
    qi, gi = np.where(z["is_query"])[0], np.where(~z["is_query"])[0]
    out = {}
    for key in ("app_ft", "app_zs"):
        e = z[key].astype(np.float64)
        e /= np.linalg.norm(e, axis=1, keepdims=True)
        S_cos = e[qi] @ e[gi].T
        coll = f"eval_mevid_{key}"
        resonance.ROOT = Path("/tmp") / f"resonance_eval_{os.getuid()}"
        resonance._MAPS.pop(coll, None)
        if resonance.count(coll) != len(gi):
            resonance._MAPS[coll] = {"ids": {}, "rev": {}, "payload": {}}
            t = time.time()
            resonance.upsert(coll, [int(g) for g in gi], e[gi].tolist(), [{} for _ in gi])
            resonance.save(coll)
            print(f"  inserted {len(gi)} gallery vectors into {coll} in {time.time() - t:.0f}s", flush=True)
        # ResonanceDB returns every match with its full 1536-value pattern, so rank only the top `top` of the
        # gallery with it; the rest is ordered after them. Cosine is cut the same way for a like-for-like mAP.
        top = min(len(gi), 300)
        cut = np.argsort(-S_cos, axis=1)[:, top:]
        S_cos = S_cos.copy(); np.put_along_axis(S_cos, cut, -1e9, axis=1)
        S_res = np.full_like(S_cos, -1e9)
        col = {int(g): c for c, g in enumerate(gi)}
        lat = []
        for j, q in enumerate(qi):
            t = time.time()
            hits = resonance.search(coll, e[q], k=top)
            lat.append(time.time() - t)
            for rank, h in enumerate(hits):
                S_res[j, col[h.id]] = -rank           # rank order is what matters for the metrics
        for mode in ("all", "cc"):
            out[f"{key}/{mode}/cosine"] = mevid_rank(S_cos, z, qi, gi, mode)
            out[f"{key}/{mode}/resonance"] = mevid_rank(S_res, z, qi, gi, mode)
            print(f"  {key:6s} {mode:3s}  cosine {out[f'{key}/{mode}/cosine']}  resonance {out[f'{key}/{mode}/resonance']}", flush=True)
        out[f"{key}/resonance_median_latency_ms"] = round(1000 * float(np.median(lat)), 1)
    return out


def part_b(colls, n=200, k=10):
    c = vector_db.client()
    out = {}
    for coll in colls:
        if not resonance.count(coll):
            print(f"  {coll}: not synced, skipped"); continue
        pts, _ = c.scroll(coll, limit=5000, with_vectors=True, with_payload=False)
        rng = np.random.RandomState(0)
        sample = [pts[i] for i in rng.choice(len(pts), min(n, len(pts)), replace=False)]
        ov, tq, tr = [], [], []
        for p in sample:
            t = time.time(); a = [h.id for h in c.query_points(coll, query=p.vector, limit=k + 1).points if h.id != p.id][:k]; tq.append(time.time() - t)
            t = time.time(); b = [h.id for h in resonance.search(coll, p.vector, k + 1) if h.id != p.id][:k]; tr.append(time.time() - t)
            ov.append(len(set(a) & set(b)) / k)
        out[coll] = dict(top10_overlap=round(float(np.mean(ov)), 3), qdrant_ms=round(1000 * float(np.median(tq)), 1),
                         resonance_ms=round(1000 * float(np.median(tr)), 1), n=len(sample))
        print(f"  {coll:14s} {out[coll]}", flush=True)
    return out


def part_c(k=10):
    from src.query import contextual
    from eval.search.compare_queries import QUERIES
    out = {}
    for backend in ("qdrant", "resonance"):
        vector_db.BACKEND = backend
        full, lat, per = 0, [], {}
        for q in QUERIES:
            t = time.time(); r = contextual.search(q, n=k); lat.append(time.time() - t)
            f = sum(h["match"] == "full" for h in r["hits"][:k]); full += f; per[q] = f
        out[backend] = dict(full_at_10=full, median_latency_s=round(float(np.median(lat)), 2), per_query=per)
        print(f"  {backend:9s} full@10 {full} over {len(QUERIES)} queries, median {out[backend]['median_latency_s']}s", flush=True)
    vector_db.BACKEND = os.getenv("VECTOR_BACKEND", "qdrant")
    return out


def main(a):
    if not resonance.health():
        raise SystemExit(f"ResonanceDB is not answering at {resonance.URL}")
    rep = {}
    live_root = resonance.ROOT
    if "a" in a.parts:
        print("A. MEVID re-ID ground truth"); rep["mevid"] = part_a(a.cache)
    resonance.ROOT = live_root; resonance._MAPS.clear()
    if "b" in a.parts:
        print("B. live index agreement"); rep["agreement"] = part_b(a.colls)
    if "c" in a.parts:
        print("C. end-to-end query set"); rep["queries"] = part_c()
    Path(a.report).parent.mkdir(parents=True, exist_ok=True)
    Path(a.report).write_text(json.dumps(rep, indent=1)); print("wrote", a.report)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts", default="abc")
    ap.add_argument("--cache", default="outputs/gait_prep/emb_test_ft.npz")
    ap.add_argument("--colls", nargs="+", default=["person_ft", "vehicle_ft", "text_person", "text_vehicle", "incident"])
    ap.add_argument("--report", default="outputs/reports/compare_backends.json")
    main(ap.parse_args())
