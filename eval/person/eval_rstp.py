import argparse, json
from pathlib import Path
import numpy as np
from src.common import clip

def main(root, split, limit):
    root = Path(root)
    js = next(root.rglob("data_captions.json"), None)
    if js is None:
        raise SystemExit(f"data_captions.json not found under {root}")
    allrows = json.load(open(js))
    data = [d for d in allrows if d.get("split") == split]
    if not data:
        raise SystemExit(f"no rows for split={split}; splits: {sorted({d.get('split') for d in allrows}, key=str)}; keys: {list(allrows[0])}")
    if limit:
        data = data[:limit]
    byname = {}
    for f in root.rglob("*"):
        if f.suffix.lower() in (".jpg", ".jpeg", ".png"):
            byname.setdefault(f.name, f)
    print(len(byname), "image files under", root)
    paths, gp, texts, qp = [], [], [], []
    for d in data:
        p = js.parent / d["img_path"]
        if not p.exists():
            p = root / d["img_path"]
        if not p.exists():
            p = byname.get(Path(d["img_path"]).name)
        if p is None or not p.exists():
            continue
        paths.append(str(p)); gp.append(d["id"])
        for c in d["captions"]:
            texts.append(c); qp.append(d["id"])
    if not paths:
        raise SystemExit(f"no images resolved; first img_path = {data[0]['img_path']!r}, json at {js}")
    print(len(paths), "images,", len(texts), "text queries")
    G = clip.embed_images(paths)
    T = np.concatenate([clip.embed_text(texts[i:i + 256]) for i in range(0, len(texts), 256)])
    gp, qp = np.array(gp), np.array(qp)
    order = np.argsort(-(T @ G.T), axis=1)
    r1 = r5 = r10 = 0; aps = []
    for i in range(len(qp)):
        m = gp[order[i]] == qp[i]
        if not m.any():
            continue
        first = int(np.argmax(m)); hits = np.where(m)[0]
        r1 += first < 1; r5 += first < 5; r10 += first < 10
        aps.append(float(np.mean((np.arange(len(hits)) + 1) / (hits + 1))))
    n = len(qp)
    res = {"split": split, "queries": n, "R1": r1 / n, "R5": r5 / n, "R10": r10 / n, "mAP": float(np.mean(aps))}
    print(json.dumps(res, indent=1))
    Path("outputs/reports").mkdir(parents=True, exist_ok=True)
    json.dump(res, open("outputs/reports/baseline_rstp.json", "w"), indent=1)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/raw/rstpreid")
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    main(a.root, a.split, a.limit)
