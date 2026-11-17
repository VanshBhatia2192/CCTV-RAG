"""Download a camera-balanced MEVA subset from the public bucket (no AWS account needed).

    python -m training.sr.download_meva --out data/raw/meva

Picks `--per-cam` random videos from each training camera and `--val-per-cam` from each held-out camera
(seeded, so re-running picks the same files and only fetches what is missing). Default: 25 x 5 + 3 x 7 = 146
videos, about 17.5 GB. The held-out cameras must match `--val-cam` given to training.sr.prepare_meva.
"""
import argparse, json, random, re, urllib.parse, urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

BUCKET = "https://mevadata-public-01.s3.amazonaws.com/"
VAL_CAMS = "school.G424,bus.G509,hospital.G479"          # one per site: school, bus, hospital


def list_videos(prefix):
    out, tok = [], None
    while True:
        q = {"list-type": "2", "prefix": prefix}
        if tok:
            q["continuation-token"] = tok
        x = urllib.request.urlopen(BUCKET + "?" + urllib.parse.urlencode(q), timeout=60).read().decode()
        out += [(k, int(s)) for k, s in re.findall(r"<Key>([^<]+)</Key>.*?<Size>(\d+)</Size>", x, re.S)
                if k.lower().endswith((".avi", ".mp4"))]
        m = re.search(r"<NextContinuationToken>([^<]+)</NextContinuationToken>", x)
        if not m:
            return out
        tok = m.group(1)


def cam_of(key):
    m = re.match(r"\d{4}-\d{2}-\d{2}\.[\d-]+\.[\d-]+\.([^.]+)\.([^.]+)\.", key.split("/")[-1])
    return f"{m.group(1)}.{m.group(2)}" if m else None


def fetch(job):
    key, size, out = job
    dst = Path(out) / key
    if dst.exists() and dst.stat().st_size == size:
        return 0
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".part")
    with urllib.request.urlopen(BUCKET + urllib.parse.quote(key), timeout=120) as r, open(tmp, "wb") as f:
        while chunk := r.read(1 << 20):
            f.write(chunk)
    assert tmp.stat().st_size == size, f"size mismatch for {key}"
    tmp.replace(dst)
    return size


def main(a):
    vids = list_videos(a.prefix)
    by_cam = defaultdict(list)
    for k, s in vids:
        if cam_of(k):
            by_cam[cam_of(k)].append((k, s))
    val = set(a.val_cam.split(","))
    missing = val - set(by_cam)
    assert not missing, f"val cameras not in bucket: {missing}"
    rng = random.Random(a.seed)
    picks = []
    for cam in sorted(by_cam):
        pool = sorted(by_cam[cam])
        rng.shuffle(pool)
        picks += pool[:a.val_per_cam if cam in val else a.per_cam]
    total = sum(s for _, s in picks)
    print(f"{len(vids)} videos / {len(by_cam)} cameras in bucket; picking {len(picks)} videos, {total / 1e9:.1f} GB", flush=True)
    if a.dry_run:
        return
    Path(a.out).mkdir(parents=True, exist_ok=True)
    json.dump(dict(val_cameras=sorted(val), videos=picks), open(Path(a.out) / "subset.json", "w"), indent=1)
    done = 0
    with ThreadPoolExecutor(a.threads) as ex:
        for i, n in enumerate(ex.map(fetch, [(k, s, a.out) for k, s in picks]), 1):
            done += n
            if i % 10 == 0 or i == len(picks):
                print(f"{i}/{len(picks)} files, {done / 1e9:.1f} GB downloaded this run", flush=True)
    print("done ->", a.out, flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/raw/meva")
    ap.add_argument("--prefix", default="drops-123-r13/")
    ap.add_argument("--per-cam", type=int, default=5)
    ap.add_argument("--val-per-cam", type=int, default=7)
    ap.add_argument("--val-cam", default=VAL_CAMS)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true")
    main(ap.parse_args())
