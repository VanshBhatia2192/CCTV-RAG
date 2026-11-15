import sys, tarfile
from pathlib import Path
split = sys.argv[1]
want = {f"bbox_{split}/{l.strip()}" for l in open(f"outputs/gait_prep/{split}_extract.txt") if l.strip()}
out = Path(f"data/processed/mevid_gait/{split}")
tf = tarfile.open(f"data/raw/mevid/mevid-v1-bbox-{split}.tgz", "r|gz")
n = seen = 0
for m in tf:
    seen += 1
    if m.isfile() and m.name in want:
        dst = out / m.name.split("/", 1)[1]
        if not dst.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_bytes(tf.extractfile(m).read())
        n += 1
        if n == len(want):
            break
    if seen % 100000 == 0:
        print(seen, n, flush=True)
print("done", seen, n, len(want), flush=True)
