import argparse, json, random, re
from pathlib import Path
import numpy as np
from src.index.sql_db import connect
from src.captions import hybrid

def main(n, seed):
    db = connect()
    rows = db.execute("SELECT id, caption FROM tracks WHERE kind='person' AND caption IS NOT NULL").fetchall()
    random.Random(seed).shuffle(rows)
    qs = [(r["id"], re.sub(r"\s*Motion:.*$", "", r["caption"]).strip()) for r in rows[:n]]
    print(len(rows), "captioned person tracks,", len(qs), "pseudo-queries; e.g.:", qs[0][1])
    res = {"n_queries": len(qs), "gallery": len(rows)}
    for mode in ("visual", "visual_ft"):
        rk = []
        for tid, q in qs:
            ids = [h["id"] for h in hybrid.retrieve("person", q, 50, mode)]
            rk.append(ids.index(tid) + 1 if tid in ids else 0)
        r = np.array(rk)
        res[mode] = {"R@1": float(np.mean((r > 0) & (r <= 1))), "R@5": float(np.mean((r > 0) & (r <= 5))),
                     "R@10": float(np.mean((r > 0) & (r <= 10))), "MRR": float(np.mean([1 / x if x else 0 for x in r]))}
        print(mode, res[mode], flush=True)
    Path("outputs/reports").mkdir(parents=True, exist_ok=True)
    json.dump(res, open("outputs/reports/eval_pseudo_person.json", "w"), indent=1)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(); main(a.n, a.seed)
