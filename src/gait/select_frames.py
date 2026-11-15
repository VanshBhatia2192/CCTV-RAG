import json, re, sys
from pathlib import Path
import numpy as np

A = Path("data/raw/mevid/mevid-v1-annotation-data")
OUT = Path("outputs/gait_prep"); OUT.mkdir(parents=True, exist_ok=True)
RX = re.compile(r"^(\d{4})([A-Za-z0-9]{4})C(\d+)T(\d+)$")
W, MIN_N = 64, 30
split = sys.argv[1]
K = int(sys.argv[2]) if len(sys.argv) > 2 else 3

def windows(n):
    if n < MIN_N:
        return []
    if n <= W:
        return [[0, n]]
    k = min(K, n // W)
    return [[int(s), int(s) + W] for s in np.linspace(0, n - W, k).round()]

info = np.rint(np.loadtxt(A / f"track_{split}_info.txt")).astype(int)
names = (A / f"{split}_name.txt").read_text().split()
print(f"[{split}] K={K} info rows {len(info)}, name lines {len(names)}")
contig = bool(info[0, 0] == 0 and (info[1:, 0] == info[:-1, 1] + 1).all() and info[-1, 1] == len(names) - 1)
print("ranges contiguous, end inclusive, cover all names:", contig)
if not contig:
    sys.exit("range convention differs; paste this output")

tr, bad, gaps, cam_mis, of_mis, ex = [], 0, 0, 0, 0, None
empty = 0
for i, row in enumerate(info):
    s, e, pid, outfit, cam = [int(x) for x in row]
    ch = names[s:e + 1]
    if not ch:
        empty += 1; continue
    pre = {x[:x.rindex("F")] for x in ch}
    m = RX.match(next(iter(pre))) if len(pre) == 1 else None
    if m is None:
        bad += 1; ex = ex or (len(pre), ch[0]); continue
    fn = np.array([int(x[x.rindex("F") + 1:-4]) for x in ch])
    gaps += int((np.diff(fn) != 1).any())
    cam_mis += int(int(m.group(3)) != cam)
    digs = re.sub(r"\D", "", m.group(2))
    of_mis += int(int(digs or -1) != outfit)
    tr.append(dict(i=i, s=s, n=len(ch), pid=pid, pid4=m.group(1), outfit=outfit, cam=cam,
                   T=int(m.group(4)), win=windows(len(ch))))
print(f"empty ranges skipped: {empty}")
print(f"parsed {len(tr)}/{len(info)} | unparsed {bad} (first: {ex}) | tracklets with frame gaps {gaps} | "
      f"info-cam != filename-cam {cam_mis} | info-outfit != filename {of_mis}")
a, b = {}, {}
for r in tr:
    a.setdefault(r["pid"], set()).add(r["pid4"]); b.setdefault(r["pid4"], set()).add(r["pid"])
print("id<->dir bijection:", all(len(v) == 1 for v in a.values()) and all(len(v) == 1 for v in b.values()), "| ids:", len(a))
if not tr:
    sys.exit("nothing parsed")

ok = [r for r in tr if r["win"]]
print(f"pass >= {MIN_N} frames: {len(ok)}/{len(tr)}")
files = []
for r in ok:
    idx = sorted({j for lo, hi in r["win"] for j in range(lo, hi)})
    files += [f'{r["pid4"]}/{names[r["s"] + j]}' for j in idx]
print("frames to extract:", len(files))
(OUT / f"{split}_extract.txt").write_text("\n".join(files) + "\n")

man = dict(split=split, W=W, K=K, min_n=MIN_N, tracklets=tr)
if split == "test":
    qs = set(int(x) for x in np.rint(np.loadtxt(A / "query_IDX.txt")).astype(int))
    byi = {r["i"]: r for r in ok}
    Q = [byi[i] for i in qs if i in byi]
    G = [r for i, r in byi.items() if i not in qs]
    pos = [[g for g in G if g["pid"] == x["pid"] and g["cam"] != x["cam"]] for x in Q]
    posx = [[g for g in p if g["outfit"] != x["outfit"]] for p, x in zip(pos, Q)]
    print(f"queries {len(qs)} (pass gate {len(Q)}), gallery {len(G)} | queries with >=1 cross-camera positive: "
          f"{sum(map(bool, pos))} | ...cross-camera AND different outfit: {sum(map(bool, posx))}")
    man["query_idx"] = sorted(qs)
(OUT / f"{split}_manifest.json").write_text(json.dumps(man))
