import argparse, json
from pathlib import Path
import cv2
from tqdm import tqdm
from ultralytics import YOLO
from src.common.config import PATHS, MODELS, SHARD, NSHARDS
from src.index.sql_db import connect

CLASSES = {"person": [0], "vehicle": [2, 3, 5, 7]}
CROP_EVERY, MAX_CROPS, MIN_SIDE = 5, 12, 32
MIN_FRAMES, MAX_TRACKS = 12, 15

def model_path():
    p = Path(PATHS["models"]) / "pretrained" / MODELS["detector"]
    return str(p if p.exists() else MODELS["detector"])

def process_chunk(model, chunk, kind, db):
    cap = cv2.VideoCapture(chunk["path"]); fps = cap.get(cv2.CAP_PROP_FPS) or 25; cap.release()
    tracks = {}
    gen = model.track(chunk["path"], tracker="bytetrack.yaml", classes=CLASSES[kind], stream=True, verbose=False, persist=False)
    for fi, r in enumerate(gen):
        if r.boxes is None or r.boxes.id is None:
            continue
        for box, tid in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.id.int().cpu().tolist()):
            x1, y1, x2, y2 = [int(v) for v in box]
            t = tracks.setdefault(tid, {"path": [], "crops": 0, "dir": None})
            t["path"].append([fi, (x1 + x2) / 2, (y1 + y2) / 2, y2 - y1, x2 - x1])
            if fi % CROP_EVERY == 0 and t["crops"] < MAX_CROPS and min(x2 - x1, y2 - y1) >= MIN_SIDE:
                if t["dir"] is None:
                    t["dir"] = Path(PATHS["processed"]) / "crops" / kind / str(chunk["id"]) / str(tid)
                    t["dir"].mkdir(parents=True, exist_ok=True)
                h, w = r.orig_shape
                crop = r.orig_img[max(y1, 0):min(y2, h), max(x1, 0):min(x2, w)]
                cv2.imwrite(str(t["dir"] / f"{fi:05d}.jpg"), crop)
                t["crops"] += 1
    best = sorted(tracks.items(), key=lambda kv: -len(kv[1]["path"]))[:MAX_TRACKS]
    for tid, t in best:
        if len(t["path"]) < MIN_FRAMES or t["dir"] is None:
            continue
        a, b = t["path"][0][0], t["path"][-1][0]
        db.execute("INSERT OR IGNORE INTO tracks(chunk_id,kind,track_id,start_s,end_s,n_frames,fps,crops_dir,path_json) "
                   "VALUES(?,?,?,?,?,?,?,?,?)",
                   (chunk["id"], kind, tid, chunk["start_s"] + a / fps, chunk["start_s"] + b / fps,
                    len(t["path"]), fps, str(t["dir"]), json.dumps(t["path"])))
    db.execute("INSERT OR IGNORE INTO processed VALUES(?,?)", (chunk["id"], kind))
    db.commit()

def main(kind, limit):
    db, model = connect(), YOLO(model_path())
    rows = db.execute("SELECT * FROM chunks WHERE kept=1 AND id NOT IN (SELECT chunk_id FROM processed WHERE kind=?) "
                      "AND id % ? = ? LIMIT ?", (kind, NSHARDS, SHARD, limit)).fetchall()
    print(len(rows), kind, "chunks to track (shard", SHARD, ")", flush=True)
    for c in tqdm(rows, desc=f"{kind} tracking", mininterval=30):
        try:
            process_chunk(model, dict(c), kind, db)
        except Exception as e:
            print("SKIP chunk", c["id"], repr(e), flush=True)
    print("tracks:", db.execute("SELECT COUNT(*) FROM tracks WHERE kind=?", (kind,)).fetchone()[0])

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", choices=["person", "vehicle"], default="person")
    ap.add_argument("--limit", type=int, default=20)
    a = ap.parse_args(); main(a.kind, a.limit)
