"""Build BasicVSR++ fine-tuning pairs from raw MEVA videos.

For each picked video, cut `--seqs-per-video` sequences of `--frames` consecutive frames:
  gt/  clean target: frame downscaled to `--gt-width` (default 960, i.e. 1080p/2, which also averages out
       some of MEVA's own compression noise), cropped to a multiple of 8
  lq/  input: gt blurred, downscaled 4x, noised, then H.264-compressed as a clip at a random CRF,
       so the model learns to undo CCTV-style degradation rather than clean bicubic downscaling
Whole cameras are held out for validation, so val measures generalisation to unseen views.

    python -m training.sr.prepare_meva /path/to/meva --out data/processed/sr_meva
Use the raw MEVA files, not data/processed/chunks (those are already downscaled and re-encoded at CRF 28).
"""
import argparse, json, random, re, subprocess, tempfile
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import cv2, numpy as np

SCALE = 4
VIDEO_EXT = {".avi", ".mp4", ".mkv", ".mov"}


def cam_name(v):
    """Same rule as src.ingest.chunk.cam_name: MEVA names look like 2018-03-07.11-00-00.11-05-00.school.G330.r13.avi."""
    m = re.match(r"\d{4}-\d{2}-\d{2}\.[\d-]+\.[\d-]+\.([^.]+)\.([^.]+)\.", Path(v).name)
    return f"{m.group(1)}.{m.group(2)}" if m else Path(v).parent.name


def probe(v):
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height:format=duration",
                          "-of", "json", str(v)], capture_output=True, text=True)
    try:
        j = json.loads(out.stdout)
        return float(j["format"]["duration"]), int(j["streams"][0]["width"]), int(j["streams"][0]["height"])
    except (KeyError, IndexError, ValueError, TypeError):
        return 0.0, 0, 0


def read_frames(v, start, n, src_wh, width):
    """n consecutive RGB frames from `start` seconds, scaled to `width` (area filter); None if the video runs out."""
    w, h = width, int(round(src_wh[1] * width / src_wh[0] / 2)) * 2
    cmd = ["ffmpeg", "-v", "error", "-ss", f"{start:.3f}", "-i", str(v), "-frames:v", str(n),
           "-vf", f"scale={w}:{h}:flags=area", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    raw = subprocess.run(cmd, capture_output=True).stdout
    return np.frombuffer(raw, np.uint8).reshape(n, h, w, 3) if len(raw) == n * w * h * 3 else None


def compress(frames, crf, tmp):
    """Encode the RGB clip with libx264 at `crf` (yuv420p, like CCTV) and decode it back."""
    k, h, w, _ = frames.shape
    path = Path(tmp) / "c.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", "25", "-i", "-",
                    "-c:v", "libx264", "-preset", "medium", "-crf", str(crf), "-pix_fmt", "yuv420p", str(path)],
                   input=frames.tobytes(), check=True)
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                         capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.uint8).reshape(-1, h, w, 3)[:k]


def degrade(gt, rng):
    """gt (k,h,w,3) uint8 RGB -> lq (k,h/4,w/4,3) uint8 RGB, plus the parameters used."""
    p = dict(blur=round(rng.uniform(0.2, 1.5), 3), noise=round(rng.uniform(0, 10), 2), crf=rng.randint(23, 35))
    h, w = gt.shape[1] // SCALE, gt.shape[2] // SCALE
    lq = []
    for f in gt:
        f = cv2.GaussianBlur(f, (0, 0), p["blur"])
        lq.append(cv2.resize(f, (w, h), interpolation=cv2.INTER_CUBIC))
    lq = np.stack(lq).astype(np.float32)
    lq += np.random.default_rng(rng.randint(0, 2**31)).normal(0, p["noise"], lq.shape).astype(np.float32)
    lq = np.clip(lq, 0, 255).round().astype(np.uint8)
    with tempfile.TemporaryDirectory() as tmp:
        lq = compress(lq, p["crf"], tmp)
    return lq, p


def make_seq(job):
    v, cam, split, start, n, src_wh, gt_width, out, sid, seed = job
    rng = random.Random(seed)
    frames = read_frames(v, start, n, src_wh, gt_width)
    if frames is None:
        return None
    h8, w8 = frames.shape[1] // 8 * 8, frames.shape[2] // 8 * 8          # lq stays even for yuv420p
    if h8 < 64 * SCALE or w8 < 64 * SCALE:
        return None
    y0, x0 = (frames.shape[1] - h8) // 2, (frames.shape[2] - w8) // 2
    gt = np.ascontiguousarray(frames[:, y0:y0 + h8, x0:x0 + w8])
    lq, p = degrade(gt, rng)
    for sub, arr in (("gt", gt), ("lq", lq)):
        d = Path(out) / split / sub / sid
        d.mkdir(parents=True, exist_ok=True)
        for i, f in enumerate(arr):
            cv2.imwrite(str(d / f"{i:05d}.png"), cv2.cvtColor(f, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_PNG_COMPRESSION, 1])
    return dict(id=sid, split=split, camera=cam, source=str(v), start_s=round(start, 2), n=len(gt),
                gt_size=[w8, h8], lq_size=[w8 // SCALE, h8 // SCALE], **p)


def main(a):
    rng = random.Random(a.seed)
    vids = sorted(p for p in Path(a.src).rglob("*") if p.suffix.lower() in VIDEO_EXT)
    by_cam = defaultdict(list)
    for v in vids:
        by_cam[cam_name(v)].append(v)
    cams = sorted(by_cam)
    print(f"{len(vids)} videos across {len(cams)} cameras", flush=True)
    assert len(cams) > a.val_cams, "need more cameras than --val-cams"
    val = set(a.val_cam.split(",")) if a.val_cam else set(rng.sample(cams, a.val_cams))
    jobs = []
    for cam in cams:
        split = "val" if cam in val else "train"
        pool = by_cam[cam][:]
        rng.shuffle(pool)
        per_cam = a.val_videos_per_cam if split == "val" else a.videos_per_cam
        for v in pool[:per_cam]:
            dur, w, h = probe(v)
            span = a.frames / 25.0 + 1
            if dur < span or w < a.gt_width:
                continue
            k = 1 if split == "val" else a.seqs_per_video
            gaps = (dur - span) / k
            for j in range(k):                     # spread sequences over the video, no overlap
                start = j * gaps + rng.uniform(0, max(gaps - span, 0.1))
                sid = f"{cam}__{v.stem}__{j}".replace("/", "_")
                jobs.append((v, cam, split, start, a.frames, (w, h), a.gt_width, a.out, sid, rng.randint(0, 2**31)))
    print(f"{len(jobs)} sequences to build ({sum(j[2] == 'val' for j in jobs)} val), val cameras: {sorted(val)}", flush=True)
    meta, done = [], 0
    with ProcessPoolExecutor(a.workers) as ex:
        for f in as_completed([ex.submit(make_seq, j) for j in jobs]):
            r = f.result()
            done += 1
            if r:
                meta.append(r)
            if done % 20 == 0:
                print(f"{done}/{len(jobs)}", flush=True)
    meta.sort(key=lambda r: (r["split"], r["id"]))
    Path(a.out).mkdir(parents=True, exist_ok=True)
    json.dump(dict(scale=SCALE, frames=a.frames, gt_width=a.gt_width, val_cameras=sorted(val), sequences=meta),
              open(Path(a.out) / "meta.json", "w"), indent=1)
    ntr = sum(r["split"] == "train" for r in meta)
    print(f"done: {ntr} train / {len(meta) - ntr} val sequences, {sum(r['n'] for r in meta)} frames -> {a.out}/meta.json", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("src", help="folder of raw MEVA videos (searched recursively)")
    ap.add_argument("--out", default="data/processed/sr_meva")
    ap.add_argument("--frames", type=int, default=50, help="frames per sequence")
    ap.add_argument("--videos-per-cam", type=int, default=5)
    ap.add_argument("--seqs-per-video", type=int, default=2)
    ap.add_argument("--val-cams", type=int, default=3, help="cameras held out for validation (random)")
    ap.add_argument("--val-cam", default="", help="comma-separated camera names to hold out instead")
    ap.add_argument("--val-videos-per-cam", type=int, default=7)
    ap.add_argument("--gt-width", type=int, default=960)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    main(ap.parse_args())
