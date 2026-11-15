import itertools, json
from pathlib import Path
import numpy as np
from src.index.sql_db import connect
from src.captions import hybrid
from eval.e2e.eval_activities import NAT, activity

def score(queries, label, cv, w):
    rs = []
    for a, q in queries:
        seen = []
        for h in hybrid.retrieve("incident", q, 300, "hybrid", w=w):
            v = cv.get(h["chunk_id"])
            if v and v not in seen:
                seen.append(v)
        rs.append(next((i + 1 for i, v in enumerate(seen) if label.get(v) == a), 0))
    r = np.array(rs)
    return {"MRR": float(np.mean([1 / x if x else 0 for x in r])), "R@5": float(np.mean((r > 0) & (r <= 5))),
            "R@10": float(np.mean((r > 0) & (r <= 10)))}

def main():
    db = connect()
    cv = {r[0]: r[1] for r in db.execute("SELECT id, source FROM chunks")}
    label = {v: activity(v) for v in set(cv.values()) if activity(v)}
    classes = sorted(set(label.values()))
    qs = [(a, NAT.get(a, a.replace("-", " "))) for a in classes]
    A, B = qs[::2], qs[1::2]
    grid = [(wv, 1.0, wb) for wv, wb in itertools.product([0.0, 0.25, 0.5, 1.0], [0.0, 0.5, 1.0])]
    res = {w: score(A, label, cv, w) for w in grid}
    best = max(res, key=lambda w: res[w]["MRR"])
    rep = {"tuned_on": len(A), "held_out": len(B), "best_weights": best,
           "equal_weights_heldout": score(B, label, cv, (1.0, 1.0, 1.0)),
           "best_weights_heldout": score(B, label, cv, best)}
    print(json.dumps(rep, indent=1))
    Path("configs/fusion.json").write_text(json.dumps({"incident": list(best)}))
    Path("outputs/reports").mkdir(parents=True, exist_ok=True)
    json.dump(rep, open("outputs/reports/tune_fusion.json", "w"), indent=1)

if __name__ == "__main__":
    main()
