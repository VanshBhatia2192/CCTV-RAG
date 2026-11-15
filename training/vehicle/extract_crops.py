"""CityFlow-NL: cut a few vehicle crops per training track out of the vdo.avi files (the img1/*.jpg frames are not on disk).
Frame names are 1-based (000001.jpg = first frame); boxes are [x, y, w, h] with a top-left origin."""
import argparse, json, os, re, sys
from collections import Counter, defaultdict
from multiprocessing import Pool
from pathlib import Path
import cv2, numpy as np

RX = re.compile(r"^\./?(?P<split>[^/]+)/(?P<scene>S\d+)/(?P<cam>c\d+)/img1/(?P<f>\d+)\.jpg$")


def plan(tracks, root, per_track, margin):
    jobs = defaultdict(list)          # video -> [(track_key, frame_no, box)]
    meta, missing = {}, Counter()
    for key, t in tracks.items():
        m = RX.match(t["frames"][0])
        if not m:
            missing["unparsed frame path"] += 1; continue
        video = Path(root) / m["split"] / m["scene"] / m["cam"] / "vdo.avi"
        if not video.exists():
            missing[f"no video {m['split']}/{m['scene']}/{m['cam']}"] += 1; continue
        n = len(t["frames"]); idx = np.unique(np.linspace(0, n - 1, min(per_track, n)).round().astype(int))
        meta[key] = dict(scene=m["scene"], cam=m["cam"], nl=t["nl"], nl_other=t.get("nl_other_views", []), crops=[])
        for i in idx:
            fn = int(re.search(r"(\d+)\.jpg$", t["frames"][i])[1])
            jobs[str(video)].append((key, fn, t["boxes"][i]))
    return jobs, meta, missing


def run_video(args):
    video, items, out, margin, maxside = args
    need = defaultdict(list)
    for key, fn, box in items:
        need[fn].append((key, box))
    cap = cv2.VideoCapture(video); done, f = [], 0
    last = max(need)
    while f < last:
        f += 1                                                    # 1-based frame number
        if f in need:
            ok, fr = cap.read()
            if not ok:
                break
            H, W = fr.shape[:2]
            for key, (x, y, w, h) in need[f]:
                mx, my = margin * w, margin * h
                x1, y1, x2, y2 = max(0, int(x - mx)), max(0, int(y - my)), min(W, int(x + w + mx)), min(H, int(y + h + my))
                if x2 - x1 < 12 or y2 - y1 < 12:
                    continue
                c = fr[y1:y2, x1:x2]; s = maxside / max(c.shape[:2])
                if s < 1: c = cv2.resize(c, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
                p = Path(out) / key / f"{f:06d}.jpg"; p.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(p), c); done.append((key, str(p)))
        elif not cap.grab():
            break
    cap.release()
    return video, done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/raw/cityflow_nl"); ap.add_argument("--videos", default="data/raw/cityflow_nl/AICity22_Track1_MTMC_Tracking")
    ap.add_argument("--out", default="data/processed/cityflow_crops"); ap.add_argument("--per-track", type=int, default=6)
    ap.add_argument("--margin", type=float, default=0.08); ap.add_argument("--maxside", type=int, default=256)
    ap.add_argument("--workers", type=int, default=6); ap.add_argument("--limit-videos", type=int, default=0)
    a = ap.parse_args()
    tracks = json.load(open(Path(a.root) / "train-tracks.json"))
    jobs, meta, missing = plan(tracks, a.videos, a.per_track, a.margin)
    print(f"tracks {len(tracks)} | planned on {len(jobs)} videos | skipped: {dict(missing)}", flush=True)
    vids = sorted(jobs)[:a.limit_videos or None]
    with Pool(a.workers) as p:
        for k, (video, done) in enumerate(p.imap_unordered(run_video, [(v, jobs[v], a.out, a.margin, a.maxside) for v in vids])):
            for key, path in done:
                meta[key]["crops"].append(path)
            print(f"[{k + 1}/{len(vids)}] {Path(video).parent.parent.name}/{Path(video).parent.name}: {len(done)} crops", flush=True)
    meta = {k: v for k, v in meta.items() if v["crops"]}
    json.dump(meta, open(Path(a.out) / "manifest.json", "w")); 
    sc = Counter(v["scene"] for v in meta.values())
    print(f"tracks with crops: {len(meta)} | crops: {sum(len(v['crops']) for v in meta.values())} | tracks per scene: {dict(sc)}", flush=True)
    rng = np.random.RandomState(0); keys = rng.choice(list(meta), min(12, len(meta)), replace=False); tiles = []
    for k in keys:
        im = cv2.imread(meta[k]["crops"][0]); s = 150 / im.shape[0]; im = cv2.resize(im, (max(1, int(im.shape[1] * s)), 150))
        im = cv2.copyMakeBorder(im, 0, 34, 0, 0, cv2.BORDER_CONSTANT, value=(30, 30, 30)); w = im.shape[1]
        cv2.putText(im, meta[k]["nl"][0][:int(w / 6.2)], (2, 163), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (255, 255, 255), 1)
        cv2.putText(im, f'{meta[k]["scene"]}/{meta[k]["cam"]}', (2, 178), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (160, 200, 255), 1); tiles.append(im)
    Path("outputs").mkdir(exist_ok=True)
    if not tiles:
        print("no crops written"); return
    rows = [np.hstack(tiles[i:i + 4]) for i in range(0, len(tiles), 4)]
    W = max(r.shape[1] for r in rows)
    cv2.imwrite("outputs/cityflow_sheet.png", np.vstack([cv2.copyMakeBorder(r, 0, 4, 0, W - r.shape[1], cv2.BORDER_CONSTANT) for r in rows])); print("wrote outputs/cityflow_sheet.png")


if __name__ == "__main__":
    main()
