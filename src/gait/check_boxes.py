"""Which box format does path_json use? Compare saved crops with crops cut from the chunk video under each format."""
import json, sqlite3, sys
from collections import Counter
from pathlib import Path
import cv2, numpy as np
from src.gait.track_crops import xyxy

FMTS = ["cxcywh", "tlwh", "cxcyhw", "tlhw"]

def corr(a, b):
    a = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY).astype(np.float32)
    b = cv2.cvtColor(cv2.resize(b, (a.shape[1], a.shape[0])), cv2.COLOR_BGR2GRAY).astype(np.float32)
    a -= a.mean(); b -= b.mean(); d = np.linalg.norm(a) * np.linalg.norm(b)
    return float((a * b).sum() / d) if d > 0 else 0.0

def main(n=10, npts=12):
    db = sqlite3.connect("data/indexes/sqlite/meta.db"); db.row_factory = sqlite3.Row
    rows = db.execute("select t.id, t.path_json, t.crops_dir, ch.path vpath from tracks t join chunks ch on ch.id=t.chunk_id "
                      "where t.kind='person' order by random() limit ?", (n,)).fetchall()
    votes, gaps = Counter(), []
    for r in rows:
        pj = json.loads(r["path_json"]); files = sorted(Path(r["crops_dir"]).glob("*.jpg"))
        saved = cv2.imread(str(files[0])); cap = cv2.VideoCapture(r["vpath"])
        Wv, Hv = int(cap.get(3)), int(cap.get(4)); frames = {}
        need = {int(p[0]) for p in pj[:npts]}
        for f in range(max(need) + 1):
            ok, fr = cap.read()
            if not ok: break
            if f in need: frames[f] = fr
        best = {}
        for fmt in FMTS:
            sc = []
            for k, p in enumerate(pj[:npts]):
                x1, y1, x2, y2 = xyxy(p, fmt, Wv, Hv)
                if x2 - x1 >= 8 and y2 - y1 >= 16 and int(p[0]) in frames:
                    sc.append((corr(saved, frames[int(p[0])][y1:y2, x1:x2]), k))
            best[fmt] = max(sc) if sc else (-1, -1)
        order = sorted(best, key=lambda f: -best[f][0])
        votes[order[0]] += 1; gaps.append(best[order[0]][0] - best[order[1]][0])
        print(f'track {r["id"]}: saved {saved.shape[1]}x{saved.shape[0]} ({files[0].name}) | best {order[0]} corr {best[order[0]][0]:.2f} at path point {best[order[0]][1]} '
              f'| runner-up {order[1]} {best[order[1]][0]:.2f}')
    print("\nvotes:", dict(votes), "| median margin over runner-up: %.2f" % float(np.median(gaps)))

if __name__ == "__main__":
    main()
