import json, sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import cv2, numpy as np
from ultralytics import YOLO
from src.gait.silhouettes import norm64

A = Path("data/raw/mevid/mevid-v1-annotation-data")
ROOT = Path("data/processed/mevid_gait")
split = sys.argv[1]
limit = int(sys.argv[2]) if len(sys.argv) > 2 else 0
man = json.load(open(f"outputs/gait_prep/{split}_manifest.json"))
names = (A / f"{split}_name.txt").read_text().split()
out = ROOT / f"{split}_sil"; out.mkdir(parents=True, exist_ok=True)
model = YOLO("models/pretrained/yolov8m-seg.pt")
pool = ThreadPoolExecutor(8)

def masks(ims):
    res = model.predict(ims, imgsz=320, classes=[0], retina_masks=True, half=True, verbose=False)
    o = []
    for r in res:
        if r.masks is None or len(r.masks.data) == 0:
            o.append(None); continue
        d = r.masks.data
        k = int(d.flatten(1).sum(1).argmax())
        o.append((d[k] > 0.5).byte().mul(255).cpu().numpy())
    return o

todo = [t for t in man["tracklets"] if t["win"]]
if limit:
    todo = todo[:limit]
t0 = time.time(); nf = 0; nvalid = 0
for n_done, t in enumerate(todo):
    f = out / f'{t["i"]}.npz'
    if f.exists():
        continue
    idx = sorted({j for lo, hi in t["win"] for j in range(lo, hi)})
    paths = [str(ROOT / split / t["pid4"] / names[t["s"] + j]) for j in idx]
    sil = np.zeros((len(idx), 64, 64), np.uint8)
    asp = np.zeros(len(idx), np.float16); trunc = np.zeros(len(idx), bool)
    for b in range(0, len(idx), 64):
        ims = list(pool.map(cv2.imread, paths[b:b + 64]))
        keep = [k for k, im in enumerate(ims) if im is not None]
        ms = masks([ims[k] for k in keep]) if keep else []
        for k, m in zip(keep, ms):
            if m is None:
                continue
            s = norm64(m)
            if s is None:
                continue
            ys = np.nonzero(m.any(1))[0]; xs = np.nonzero(m.any(0))[0]
            sil[b + k] = s
            asp[b + k] = (ys[-1] - ys[0] + 1) / (xs[-1] - xs[0] + 1)
            trunc[b + k] = bool(ys[0] == 0 or ys[-1] == m.shape[0] - 1)
    np.savez_compressed(f, sil=sil, idx=np.array(idx), asp=asp, trunc=trunc)
    nf += len(idx); nvalid += int(sil.any((1, 2)).sum())
    if (n_done + 1) % 25 == 0:
        print(f"{n_done + 1}/{len(todo)} tracklets | {nf / (time.time() - t0):.0f} frames/s | valid {nvalid / nf:.2f}", flush=True)
print(f"done {len(todo)} tracklets, {nf} frames, valid frac {nvalid / max(nf, 1):.3f}, {time.time() - t0:.0f}s", flush=True)
