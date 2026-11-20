"""Demo API: text search over person / vehicle / incident indexes + gait/appearance re-id. Run from the project root."""
import hashlib, json, re, shutil, threading, time, uuid
from pathlib import Path
import numpy as np
from fastapi import BackgroundTasks, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, StreamingResponse

import os
if os.environ.get("DEMO_DB"):   # use a node-local copy of the SQLite index (WAL mode is unreliable on the shared filesystem)
    from src.common.config import PATHS
    PATHS["sqlite"] = os.environ["DEMO_DB"]

app = FastAPI(title="CCTV search demo")
STATIC = Path(__file__).parent / "static"
MODES = {"person": "hybrid_ft" if Path("models/finetuned/text2person/clip_t2p.pt").exists() else "hybrid", "vehicle": "hybrid", "incident": "hybrid"}
WARM = {"done": False, "error": None, "secs": None}   # model warm-up status, shown by the page


def _fusion_weights(kind):
    """Same tuned (visual, text, bm25) weights src/query/search.py uses (configs/fusion.json)."""
    p = Path("configs/fusion.json"); w = json.load(open(p)).get(kind) if p.exists() else None
    return tuple(w) if w else (1.0, 1.0, 1.0)


def _db():
    from src.index.sql_db import connect
    return connect()


def _row(db, sql, *args):
    r = db.execute(sql, args).fetchone()
    return dict(r) if r is not None else None


def _track(db, tid):
    return _row(db, "select t.id, t.chunk_id, t.kind, t.start_s, t.end_s, t.crops_dir, t.caption, t.attrs_json, ch.camera, "
                    "ch.start_s as chunk_start from tracks t join chunks ch on ch.id=t.chunk_id where t.id=?", tid)


def _clip_times(start, end, chunk_start):
    t0 = max(0.0, float(start) - float(chunk_start)); t1 = max(t0 + 1.0, float(end) - float(chunk_start))
    return round(t0, 1), round(min(t1, 10.0), 1)


def _card_track(t, score=None):
    t0, t1 = _clip_times(t["start_s"], t["end_s"], t["chunk_start"])
    return dict(id=t["id"], kind=t["kind"], camera=t["camera"], start_s=round(t["start_s"], 1), end_s=round(t["end_s"], 1),
                caption=t["caption"], attrs=json.loads(t.get("attrs_json") or "{}"), thumb=f"/api/thumb/{t['id']}",
                clip=f"/api/clip/{t['chunk_id']}", clip_t0=t0, clip_t1=t1, score=None if score is None else round(float(score), 4))


def _card_window(db, hit):
    w = None
    try:
        w = _row(db, "select w.id, w.chunk_id, w.start_s, w.end_s, w.rel_start, w.rel_end, w.caption, ch.camera, ch.start_s as chunk_start "
                     "from windows w join chunks ch on ch.id=w.chunk_id where w.id=?", hit["id"])
    except Exception as e:
        raise HTTPException(500, f"incident lookup failed ({type(e).__name__}: {e}); check the windows table columns")
    if w is None:
        return None
    t0 = round(max(0.0, float(w["rel_start"])), 1); t1 = round(min(max(float(w["rel_end"]), t0 + 1.0), 10.0), 1)   # chunk-relative
    return dict(id=w["id"], kind="incident", camera=w["camera"], start_s=round(w["start_s"], 1), end_s=round(w["end_s"], 1),
                caption=w["caption"], attrs={}, thumb=None, clip=f"/api/clip/{w['chunk_id']}", clip_t0=t0, clip_t1=t1,
                score=round(float(hit["score"]), 4))


def drop_closeups(db, rows, min_aspect=1.3):
    """Keep full-body-looking tracks: median box height/width (path_json = [frame, cx, cy, h, w]) >= min_aspect."""
    keep = []
    for r in rows:
        x = _row(db, "select path_json from tracks where id=?", r["track_row"])
        pj = json.loads(x["path_json"] or "[]") if x else []
        if pj and float(np.median([p[3] / max(p[4], 1.0) for p in pj])) >= min_aspect:
            keep.append(r)
    return keep


def _range_response(path, request):
    size = path.stat().st_size; hdr = request.headers.get("range")
    m = re.match(r"bytes=(\d*)-(\d*)", hdr or "")
    if not m:
        return FileResponse(path, media_type="video/mp4", headers={"Accept-Ranges": "bytes"})
    g1, g2 = m.groups()
    start, end = (max(0, size - int(g2)), size - 1) if (not g1 and g2) else (int(g1 or 0), min(int(g2), size - 1) if g2 else size - 1)
    if start >= size or start > end:
        raise HTTPException(416, "range not satisfiable")
    def body():
        with open(path, "rb") as f:
            f.seek(start); left = end - start + 1
            while left > 0:
                b = f.read(min(1 << 20, left))
                if not b:
                    break
                left -= len(b); yield b
    return StreamingResponse(body(), status_code=206, media_type="video/mp4",
                             headers={"Content-Range": f"bytes {start}-{end}/{size}", "Accept-Ranges": "bytes", "Content-Length": str(end - start + 1)})


MEDIA = {}   # token -> trimmed clip path returned by src.query.trim (only these files are ever served)


def _media(path):
    p = Path(path); tok = hashlib.sha1(str(p).encode()).hexdigest()[:16]; MEDIA[tok] = p
    return f"/api/media/{tok}"


@app.get("/api/media/{tok}")
def media(tok: str, request: Request):
    p = MEDIA.get(tok)
    if p is None or not p.exists():
        raise HTTPException(404, "no clip")
    return _range_response(p, request)


@app.get("/api/ask")
def ask(q: str, n: int = 12, use_llm: bool = False, rerank: bool = False):
    """The full existing pipeline (src/query/search.py): parse type + camera, retrieve, localize, trim."""
    if not q.strip():
        raise HTTPException(400, "empty query")
    from src.query import search as S
    try:
        res = S.search(q.strip(), rerank=rerank, use_llm=use_llm, n_clips=min(max(n, 1), 24))
    except Exception as e:
        raise HTTPException(500, f"search failed ({type(e).__name__}: {e})")
    kind = res["query"]["type"]; db = _db(); out = []
    try:
        for r in res["results"]:
            tid = r.get("id")
            t = _track(db, tid) if (kind != "incident" and tid is not None) else None
            out.append(dict(id=tid, kind=kind, camera=r["camera"], start_s=round(r["start_s"], 1), end_s=round(r["end_s"], 1),
                            score=r["score"], caption=r.get("caption"), vlm=r.get("vlm"), source=r.get("source"), clip=_media(r["clip"]),
                            thumb=f"/api/thumb/{tid}" if t else None, attrs=json.loads(t["attrs_json"] or "{}") if t else {}))
    finally:
        db.close()
    return dict(query=res["query"], kind=kind, mode=res["mode"], latency_s=res["latency_s"], timings=res["timings"], n=len(out), results=out)


@app.get("/api/health")
def health():
    return {"ok": True, **WARM}


@app.get("/api/search")
def search(kind: str, q: str, k: int = 20, mode: str | None = None):
    if kind not in MODES:
        raise HTTPException(400, f"kind must be one of {list(MODES)}")
    if not q.strip():
        raise HTTPException(400, "empty query")
    from src.captions import hybrid
    t0 = time.time(); mode = mode or MODES[kind]
    hits = hybrid.retrieve(kind, q.strip(), k=min(max(k, 1), 60), mode=mode, w=_fusion_weights(kind))
    db = _db(); out = []
    try:
        for h in hits:
            card = _card_window(db, h) if kind == "incident" else (lambda t: _card_track(t, h["score"]) if t else None)(_track(db, h["id"]))
            if card:
                out.append(card)
    finally:
        db.close()
    return dict(kind=kind, q=q, mode=mode, took_s=round(time.time() - t0, 2), n=len(out), results=out)


@app.get("/api/thumb/{tid}")
def thumb(tid: int):
    db = _db()
    try:
        t = _row(db, "select crops_dir from tracks where id=?", tid)
    finally:
        db.close()
    files = sorted(Path(t["crops_dir"]).glob("*.jpg")) if t else []
    if not files:
        raise HTTPException(404, "no crop")
    return FileResponse(files[len(files) // 2], media_type="image/jpeg", headers={"Cache-Control": "max-age=86400"})


@app.get("/api/clip/{chunk_id}")
def clip(chunk_id: int, request: Request):
    db = _db()
    try:
        c = _row(db, "select path from chunks where id=?", chunk_id)
    finally:
        db.close()
    if not c or not Path(c["path"]).exists():
        raise HTTPException(404, "no video")
    return _range_response(Path(c["path"]), request)


@app.get("/api/reid/vehicle/{tid}")
def reid_vehicle_api(tid: int, k: int = 12, use_st: bool = True, use_color: bool = True):
    """Multi-camera vehicle tracking endpoint using visual Re-ID, strict color gating, and spatio-temporal physics."""
    from src.vehicle import reid as vreid
    k = min(max(k, 1), 40)
    try:
        res = vreid.rank(tid, k=k, use_st_filter=use_st, use_color_gate=use_color)
    except KeyError as e:
        raise HTTPException(404, str(e))
    except Exception as e:
        raise HTTPException(500, f"Vehicle tracking failed: {e}")

    db = _db()
    try:
        cards = []
        for r in res.get("results", []):
            t = _track(db, r["track_row"])
            if t:
                c = _card_track(t, r["score"])
                c.update(
                    appearance_sim=r.get("appearance_sim"),
                    st_valid=r.get("st_valid"),
                    color_match=r.get("color_match"),
                )
                cards.append(c)
            else:
                cards.append(r)
        q = _track(db, tid)
        query = _card_track(q) if q else res.get("query")
    finally:
        db.close()
    return dict(
        query=query,
        kind="vehicle",
        use_st=use_st,
        use_color=use_color,
        results=cards,
    )


@app.get("/api/reid/{tid}")
def reid_api(tid: int, mode: str = "any", k: int = 12, full_body: bool = True, use_st: bool = True, use_color: bool = True):
    db = _db()
    try:
        t = _track(db, tid)
    finally:
        db.close()

    # Automatically route vehicles to the multi-camera vehicle tracking engine
    if t and t.get("kind") == "vehicle":
        return reid_vehicle_api(tid, k=k, use_st=use_st, use_color=use_color)

    from src.gait import reid
    if mode not in reid.MODES:
        raise HTTPException(400, f"mode must be one of {list(reid.MODES)}")
    k = min(max(k, 1), 40)
    try:
        res = reid.rank(tid, k=k * 4 if full_body else k, mode=mode)
    except KeyError as e:
        raise HTTPException(404, str(e))
    db = _db()
    try:
        rows = drop_closeups(db, res["results"]) if full_body else res["results"]
        cards = []
        for r in rows[:k]:
            t = _track(db, r["track_row"])
            if t:
                c = _card_track(t, r["score"])
                c.update(
                    appearance_z=r.get("appearance_z"),
                    gait_z=r.get("gait_z"),
                    pose_z=r.get("pose_z"),
                    cadence_bpm=r.get("cadence_bpm"),
                    gait_symmetry=r.get("gait_symmetry"),
                )
                cards.append(c)
        q = _track(db, tid)
        query = _card_track(q) if q else None
    finally:
        db.close()
    return dict(query=query, mode=mode, w_appearance=res["w_appearance"], note=res["note"], full_body_filter=full_body, results=cards)


MAIN_JOBS: dict = {}


def _run_main_gait_job(job_id: str, dest_path: Path, uploads_dir: Path):
    import sys
    if sys.meta_path is None:
        return
    try:
        from demo_showcase.video_tracker import process_video_and_detect_gait_alerts
    except (ImportError, TypeError):
        return

    def update_prog(pct: int, msg: str):
        if job_id in MAIN_JOBS:
            MAIN_JOBS[job_id]["progress"] = pct
            MAIN_JOBS[job_id]["message"] = msg
    try:
        report = process_video_and_detect_gait_alerts(
            str(dest_path),
            output_dir=uploads_dir,
            progress_callback=update_prog,
        )
        MAIN_JOBS[job_id] = {"status": "completed", "progress": 100, "message": "Done", "report": report}
    except Exception as e:
        if sys.meta_path is None:
            return
        MAIN_JOBS[job_id] = {"status": "error", "progress": 0, "message": str(e), "report": None}


@app.post("/api/upload_video")
async def upload_video_api(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    """Upload any video file, track people, extract 2D skeleton gait, and alert on re-appearance."""
    uploads_dir = Path("data/uploads")
    uploads_dir.mkdir(parents=True, exist_ok=True)
    dest_path = uploads_dir / file.filename
    with open(dest_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    job_id = str(uuid.uuid4())
    MAIN_JOBS[job_id] = {"status": "processing", "progress": 5, "message": f"Processing {file.filename}...", "report": None}
    background_tasks.add_task(_run_main_gait_job, job_id, dest_path, uploads_dir)
    return {"job_id": job_id, "status": "processing"}


@app.get("/api/upload_status/{job_id}")
def get_main_upload_status(job_id: str):
    if job_id not in MAIN_JOBS:
        raise HTTPException(404, "Job not found")
    return MAIN_JOBS[job_id]


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


@app.on_event("startup")
def warm():
    def go():
        import sys
        t0 = time.time()
        try:
            if sys.meta_path is None:
                return
            from src.captions import hybrid
            from src.query import search as _S   # import the pipeline modules now, not on the first request
            for kind, mode in MODES.items():
                if sys.meta_path is None:
                    return
                hybrid.retrieve(kind, "a person" if kind == "person" else "a car" if kind == "vehicle" else "people walking", k=1, mode=mode, w=_fusion_weights(kind))
        except (ImportError, TypeError) as e:
            if "sys.meta_path" in str(e) or sys.meta_path is None:
                return
            WARM["error"] = f"{type(e).__name__}: {e}"; print("warm-up failed:", WARM["error"])
        except Exception as e:
            if sys.meta_path is None:
                return
            WARM["error"] = f"{type(e).__name__}: {e}"; print("warm-up failed:", WARM["error"])
        WARM["secs"] = round(time.time() - t0, 1); WARM["done"] = True
    threading.Thread(target=go, daemon=True).start()
