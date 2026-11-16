"""Enhance a video clip with BasicVSR++ (4x). Used on final search-result clips when pipeline.yaml sr.enabled is true.

    python -m src.sr.enhance in.mp4 [out.mp4]

The input is first scaled down to at most sr.in_width (4x output stays manageable: 480 -> 1920 px wide),
then run in overlapping windows of sr.window frames so long clips don't run out of GPU memory.
"""
import json, subprocess, sys, threading
from pathlib import Path
import numpy as np, torch
from src.common.config import MODELS, PIPE
from src.sr.basicvsrpp import build

_MODEL, _LOCK = None, threading.Lock()


def cfg():
    return {"in_width": 480, "window": 30, "overlap": 4, "crf": 20, **(PIPE.get("sr") or {})}


def _model():
    global _MODEL
    if _MODEL is None:
        m = MODELS["sr"]
        w = m["weights"] if Path(m["weights"]).exists() else m.get("pretrained")
        if not w or not Path(w).exists():
            raise FileNotFoundError(f"no BasicVSR++ weights at {m['weights']} or {m.get('pretrained')}")
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        _MODEL = build(w, dev)
        print(f"[sr] loaded {w} on {dev}", flush=True)
    return _MODEL


def _probe(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height,r_frame_rate",
                          "-of", "json", str(path)], capture_output=True, text=True, check=True)
    s = json.loads(out.stdout)["streams"][0]
    num, den = s["r_frame_rate"].split("/")
    return int(s["width"]), int(s["height"]), float(num) / max(float(den), 1)


def _read(path, max_w):
    w, h, fps = _probe(path)
    if w > max_w:
        w, h = max_w, int(round(h * max_w / w / 2)) * 2
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-vf", f"scale={w}:{h}:flags=area", "-f", "rawvideo",
                          "-pix_fmt", "rgb24", "-"], capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.uint8).reshape(-1, h, w, 3), fps


@torch.no_grad()
def upscale(frames, window=30, overlap=4):
    """(t,h,w,3) uint8 RGB -> (t,4h,4w,3) uint8 RGB."""
    model = _model()
    dev = next(model.parameters()).device
    t = len(frames)
    if t < 2:
        frames = np.concatenate([frames, frames])
    T, half = len(frames), overlap // 2
    out, kept, start = [], 0, 0
    while True:
        end = min(start + window, T)
        s0 = max(0, end - window)                          # a short last window is extended backwards
        x = torch.from_numpy(frames[s0:end]).to(dev).permute(0, 3, 1, 2).float().div_(255)[None]
        with torch.autocast(device_type=dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda"):
            y = model(x).float()[0]
        y = (y.clamp_(0, 1) * 255).round_().byte().permute(0, 2, 3, 1).cpu().numpy()
        hi = T if end == T else end - (overlap - half)     # neighbours split the overlap; edges are dropped
        out.extend(y[kept - s0:hi - s0]); kept = hi
        if end == T:
            break
        start = end - overlap
    return np.stack(out[:t])


def enhance(src, out=None):
    """Write the 4x-enhanced clip next to src (or to out) and return its path. Cached by output path."""
    c = cfg()
    out = Path(out) if out else Path(src).with_name(Path(src).stem + "_sr.mp4")
    if out.exists():
        return str(out)
    frames, fps = _read(src, c["in_width"])
    if frames.shape[1] < 64 or frames.shape[2] < 64:
        return str(src)                                    # too small for BasicVSR++
    with _LOCK:                                            # one clip at a time on the GPU
        hr = upscale(frames, c["window"], c["overlap"])
    t, h, w, _ = hr.shape
    tmp = out.with_suffix(".tmp.mp4")
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", f"{fps:.3f}",
                    "-i", "-", "-c:v", "libx264", "-preset", "veryfast", "-crf", str(c["crf"]), "-pix_fmt", "yuv420p",
                    "-movflags", "+faststart", str(tmp)], input=hr.tobytes(), check=True)
    tmp.replace(out)
    return str(out)


if __name__ == "__main__":
    print(enhance(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None))
