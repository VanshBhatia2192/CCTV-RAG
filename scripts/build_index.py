import argparse
from src.ingest import chunk
from src.person import detect_track, embed as pembed
from src.incident import windows, embed as iembed, caption

ap = argparse.ArgumentParser()
ap.add_argument("--src", help="video file/folder to ingest (skip if already chunked)")
ap.add_argument("--camera", default=None)
ap.add_argument("--chunks", type=int, default=20, help="max chunks per kind to track")
ap.add_argument("--captions", action="store_true", help="also run Qwen2.5-VL captions (slow)")
a = ap.parse_args()
BIG = 10**6
if a.src:
    chunk.main(a.src, a.camera)
for kind in ("person", "vehicle"):
    detect_track.main(kind, a.chunks)
    pembed.main(kind, BIG)
windows.main()
iembed.main(BIG)
if a.captions:
    caption.main(BIG)
