import hashlib, shutil, subprocess
from pathlib import Path
from src.common.config import PATHS

def trim(src, a, b, tag="", dur=None):
    key = f"{src}{a:.2f}{b:.2f}{tag}v3"
    out = Path(PATHS["outputs"]) / "clips" / (hashlib.md5(key.encode()).hexdigest()[:10] + ".mp4")
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        return str(out)
    if dur is not None and a <= 0.7 and b >= dur - 0.7:      # nearly the whole chunk: copy, no re-encode
        shutil.copyfile(src, out)
        return str(out)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-ss", f"{a:.2f}", "-i", str(src), "-t", f"{max(b - a, 1):.2f}",
                    "-vf", "scale='min(640,iw)':-2", "-c:v", "libx264", "-preset", "ultrafast", "-crf", "30",
                    "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-threads", "2", "-an", str(out)], check=True)
    return str(out)
