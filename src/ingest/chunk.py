import argparse, re, subprocess
from pathlib import Path
import yaml
from src.index.sql_db import connect
from src.ingest.motion_gate import has_motion

def cam_name(v):
    m = re.match(r"\d{4}-\d{2}-\d{2}\.[\d-]+\.[\d-]+\.([^.]+)\.([^.]+)\.", Path(v).name)
    return f"{m.group(1)}.{m.group(2)}" if m else Path(v).parent.name

def duration(p):
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=nw=1:nk=1", str(p)],
                         capture_output=True, text=True)
    return float(out.stdout.strip() or 0)

def split(video, out_dir, sec):
    out_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(video), "-vf", "scale='min(1280,iw)':-2", "-c:v", "libx264", "-preset", "ultrafast",
                    "-crf", "28", "-threads", "2", "-an", "-force_key_frames", f"expr:gte(t,n_forced*{sec})",
                    "-f", "segment", "-segment_time", str(sec), "-reset_timestamps", "1", str(out_dir / "c%04d.mp4")], check=True)
    return sorted(out_dir.glob("c*.mp4"))

def main(src, camera):
    P = yaml.safe_load(open("configs/paths.yaml")); C = yaml.safe_load(open("configs/pipeline.yaml"))["ingest"]
    db = connect()
    vids = [src] if Path(src).is_file() else sorted(Path(src).rglob("*.[mM][pP]4")) + sorted(Path(src).rglob("*.avi"))
    for v in vids:
        v = Path(v)
        if db.execute("SELECT 1 FROM chunks WHERE source=? LIMIT 1", (str(v),)).fetchone():
            continue
        cam = camera or cam_name(v)
        out = Path(P["processed"]) / "chunks" / cam / v.stem
        try:
            parts = split(v, out, C["chunk_sec"])
        except subprocess.CalledProcessError:
            print("SKIP bad video:", v, flush=True); continue
        t = 0.0
        for c in parts:
            d = duration(c)
            keep, score = has_motion(c, C["motion_threshold"])
            db.execute("INSERT OR IGNORE INTO chunks(camera,source,path,start_s,end_s,motion,kept) VALUES(?,?,?,?,?,?,?)",
                       (cam, str(v), str(c), t, t + d, score, int(keep)))
            t += d
        db.commit()
        print("chunked", v.name, cam, len(parts), "chunks", flush=True)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("src"); ap.add_argument("--camera", default=None)
    a = ap.parse_args(); main(a.src, a.camera)
