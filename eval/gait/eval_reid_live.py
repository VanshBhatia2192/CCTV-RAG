"""Run two re-ID implementations on the live CCTV index and compare them.

Ground truth on real footage: a person who walks across a chunk boundary is split into two tracks of the same
camera, one ending where the next begins (same place, ~0 s gap). Those pairs are the same person for certain.
For each such query we find where its continuation ranks (same_camera allowed). This only checks same-camera,
same-clothes matching, but it is real footage with certain labels; cross-camera accuracy is measured on MEVID.

Also reports latency, how many candidates each version returns, and top-10 overlap between the two.

    python eval/gait/eval_reid_live.py --branch ~/cctv-search-gait --main ~/cctv-search-ctx --n 60
"""
import argparse, importlib.util, json, sqlite3, sys, time
from pathlib import Path
import numpy as np


def load_module(root, name="src.gait.reid"):
    """Import src/gait/reid.py from a given checkout (its own src package on sys.path)."""
    for k in [m for m in sys.modules if m == "src" or m.startswith("src.")]:
        del sys.modules[k]
    sys.path.insert(0, str(root))
    import importlib
    mod = importlib.import_module(name)
    sys.path.pop(0)
    return mod


def continuation_pairs(db, n, seed=0):
    rows = db.execute("""SELECT t.id, t.chunk_id, t.start_s, t.end_s, t.path_json, t.attrs_json, ch.source, ch.start_s cs, ch.end_s ce
                         FROM tracks t JOIN chunks ch ON ch.id=t.chunk_id WHERE t.kind='person'""").fetchall()
    by_src = {}
    for r in rows:
        by_src.setdefault(r["source"], []).append(r)
    pairs = []
    for src, rs in by_src.items():
        ends = [r for r in rs if r["ce"] - r["end_s"] < 0.25]
        starts = [r for r in rs if r["start_s"] - r["cs"] < 0.25]
        for a in ends:
            pa = json.loads(a["path_json"] or "[]")
            if len(pa) < 15:
                continue
            la = pa[-1]
            best = None
            for b in starts:
                if abs(b["cs"] - a["ce"]) > 0.25:
                    continue
                pb = json.loads(b["path_json"] or "[]")
                if len(pb) < 15:
                    continue
                fb = pb[0]
                d = np.hypot(la[1] - fb[1], la[2] - fb[2])
                if d < 0.35 * la[3] and abs(la[3] - fb[3]) < 0.3 * la[3] and (best is None or d < best[0]):
                    best = (d, b["id"])
            if best:
                pairs.append((a["id"], best[1]))
    rng = np.random.RandomState(seed)
    rng.shuffle(pairs)
    return pairs[:n]


def main(a):

    db = sqlite3.connect(a.db); db.row_factory = sqlite3.Row
    pairs = continuation_pairs(db, a.n)
    print(f"{len(pairs)} continuation pairs (same person split at a chunk boundary)")
    impls = {}
    for name, root in (("main", a.main), ("branch", a.branch)):
        mod = load_module(root)
        cfg = sys.modules["src.common.config"]; cfg.PATHS["sqlite"] = a.db
        impls[name] = mod
    report = {"pairs": len(pairs)}
    tops = {}
    for name, mod in impls.items():
        for mode in a.modes:
            if mode not in mod.MODES:
                continue
            ranks, lat, nres, errs = [], [], [], 0
            for q, pos in pairs:
                t = time.time()
                try:
                    kw = dict(k=a.k, mode=mode, same_camera=True)
                    r = mod.rank(q, **kw)
                except Exception as e:
                    errs += 1; ranks.append(0); continue
                lat.append(time.time() - t)
                ids = [x["track_row"] for x in r["results"]]
                nres.append(len(ids))
                ranks.append(ids.index(pos) + 1 if pos in ids else 0)
                tops[(name, mode, q)] = ids[:10]
            rk = np.array(ranks)
            m = {"R@1": round(float(np.mean((rk > 0) & (rk <= 1))) * 100, 1), "R@5": round(float(np.mean((rk > 0) & (rk <= 5))) * 100, 1),
                 "R@20": round(float(np.mean((rk > 0) & (rk <= 20))) * 100, 1), "MRR": round(float(np.mean([1 / x if x else 0 for x in rk])), 3),
                 "found_in_top_k": round(float(np.mean(rk > 0)) * 100, 1), "median_latency_s": round(float(np.median(lat)), 2) if lat else None,
                 "median_results": int(np.median(nres)) if nres else 0, "errors": errs}
            report[f"{name}/{mode}"] = m
            print(f"{name:7s} {mode:16s} {m}", flush=True)
    for mode in a.modes:
        ov = [len(set(tops[("main", mode, q)]) & set(tops[("branch", mode, q)])) / 10
              for q, _ in pairs if ("main", mode, q) in tops and ("branch", mode, q) in tops]
        if ov:
            report[f"top10_overlap/{mode}"] = round(float(np.mean(ov)), 2)
            print(f"top-10 overlap main vs branch ({mode}): {np.mean(ov):.2f}")
    # cross-camera behaviour of the branch's gates on real footage (default: other cameras only)
    mod = impls["branch"]; drop = []
    for q, _ in pairs[:20]:
        full = mod.rank(q, k=10000, mode="cloth_change", use_st=False, use_kinematics=False)["results"]
        gated = mod.rank(q, k=10000, mode="cloth_change")["results"]
        if full:
            drop.append(1 - len(gated) / len(full))
    if drop:
        report["branch_cloth_change_gates_drop_fraction"] = round(float(np.mean(drop)), 3)
        print(f"branch cloth-change: topology + kinematic gates remove {np.mean(drop):.0%} of cross-camera candidates")
    Path(a.report).write_text(json.dumps(report, indent=1)); print("wrote", a.report)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--branch", required=True); ap.add_argument("--main", required=True)
    ap.add_argument("--db", default="/tmp/meta_demo_vansh.122433.db")
    ap.add_argument("--n", type=int, default=60); ap.add_argument("--k", type=int, default=200)
    ap.add_argument("--modes", nargs="+", default=["any", "cloth_change", "appearance_only"])
    ap.add_argument("--report", default="outputs/reports/eval_reid_live.json")
    main(ap.parse_args())
