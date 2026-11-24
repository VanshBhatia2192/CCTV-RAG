"""Extract 17-keypoint 2D skeletons (ViTPose, top-down on each crop) for the MEVID gait windows.

Uses the same tracklets/windows as the silhouette prep (outputs/gait_prep/{split}_manifest.json, frames in
data/processed/mevid_gait/{split}/<pid4>/). Writes one file per tracklet:
    data/processed/mevid_pose/{split}/{i}.npz   kpts (N,17,3) [x, y, score] in crop pixels, idx (N,) frame
                                                 index within the tracklet, wh (N,2) crop size
Resumable (existing files are skipped).

    python -m training.gait.extract_pose --split train
    python -m training.gait.extract_pose --split test --part 0/2
"""
import argparse, json, time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

SRC = Path("data/processed/mevid_gait")
ANN = Path("data/raw/mevid/mevid-v1-annotation-data")
GP = Path("outputs/gait_prep")
OUT = Path("data/processed/mevid_pose")


class Frames(Dataset):
    def __init__(self, items, model_name):
        self.items, self.model_name, self.proc = items, model_name, None

    def __len__(self):
        return len(self.items)

    def __getitem__(self, k):
        from PIL import Image
        if self.proc is None:
            from transformers import AutoProcessor
            self.proc = AutoProcessor.from_pretrained(self.model_name)
        i, j, path = self.items[k]
        try:
            im = Image.open(path).convert("RGB")
        except Exception:
            return None
        w, h = im.size
        box = [0.0, 0.0, float(w), float(h)]
        pv = self.proc([im], boxes=[[box]], return_tensors="pt")["pixel_values"][0]
        return i, j, w, h, pv


def collate(b):
    b = [x for x in b if x is not None]
    if not b:
        return None
    i, j, w, h, pv = zip(*b)
    return list(i), list(j), list(w), list(h), torch.stack(pv)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, choices=["train", "test"])
    ap.add_argument("--part", default="0/1")
    ap.add_argument("--model", default="usyd-community/vitpose-base-simple")
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0, help="first N tracklets only (smoke test)")
    a = ap.parse_args()

    from transformers import AutoProcessor, VitPoseForPoseEstimation
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    out = OUT / a.split; out.mkdir(parents=True, exist_ok=True)
    man = json.load(open(GP / f"{a.split}_manifest.json"))
    names = (ANN / f"{a.split}_name.txt").read_text().split()
    pi, pn = (int(x) for x in a.part.split("/"))
    trs = [t for t in man["tracklets"] if t["win"]][pi::pn]
    if a.limit:
        trs = trs[:a.limit]
    todo = [t for t in trs if not (out / f'{t["i"]}.npz').exists()]
    items, want = [], {}
    for t in todo:
        js = sorted({j for lo, hi in t["win"] for j in range(lo, hi)})
        for j in js:
            p = SRC / a.split / t["pid4"] / names[t["s"] + j]
            if p.exists():
                items.append((t["i"], j, str(p)))
                want[t["i"]] = want.get(t["i"], 0) + 1
    print(f"{a.split} part {a.part}: {len(trs)} tracklets, {len(todo)} to do, {len(items)} frames | device {dev}", flush=True)
    if not items:
        return

    proc = AutoProcessor.from_pretrained(a.model)
    model = VitPoseForPoseEstimation.from_pretrained(a.model).to(dev).eval()
    if dev == "cuda":
        model = model.half()
    dl = DataLoader(Frames(items, a.model), batch_size=a.bs, num_workers=a.workers, collate_fn=collate,
                    shuffle=False, pin_memory=(dev == "cuda"), persistent_workers=a.workers > 0, prefetch_factor=4 if a.workers else None)
    acc, done, t0, nf = {}, 0, time.time(), 0

    def flush(i):
        nonlocal done
        r = acc.pop(i)
        o = np.argsort(r["idx"])
        np.savez(out / f"{i}.npz", kpts=np.asarray(r["k"], np.float32)[o], idx=np.asarray(r["idx"], np.int32)[o],
                 wh=np.asarray(r["wh"], np.int16)[o])
        done += 1

    with torch.no_grad():
        for b in dl:
            if b is None:
                continue
            I, J, Wd, Hd, pv = b
            pv = pv.to(dev, non_blocking=True)
            if dev == "cuda":
                pv = pv.half()
            boxes = [[[0.0, 0.0, float(w), float(h)]] for w, h in zip(Wd, Hd)]
            o = model(pixel_values=pv); o.heatmaps = o.heatmaps.float()   # post-processing (scipy) needs float32
            res = proc.post_process_pose_estimation(o, boxes=boxes)
            for i, j, w, h, r in zip(I, J, Wd, Hd, res):
                k = np.concatenate([r[0]["keypoints"].float().cpu().numpy(), r[0]["scores"].float().cpu().numpy()[:, None]], 1)
                d = acc.setdefault(i, dict(k=[], idx=[], wh=[]))
                d["k"].append(k); d["idx"].append(j); d["wh"].append((w, h))
                if len(d["idx"]) == want[i]:
                    flush(i)
            nf += len(I)
            if (nf // a.bs) % 50 == 0:
                el = time.time() - t0
                print(f"{nf}/{len(items)} frames | {nf / el:.0f} fps | {done} tracklets written | eta {(len(items) - nf) / max(nf / el, 1e-6) / 60:.1f} min", flush=True)
    for i in list(acc):      # tracklets with unreadable frames
        flush(i)
    print(f"done: {done} tracklets, {nf} frames in {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
