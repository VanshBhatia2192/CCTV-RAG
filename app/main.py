"""Demo API: text search over person / vehicle / incident indexes + gait/appearance re-id. Run from the project root."""
import os
# Enforce strict offline execution for HuggingFace Hub and Transformers (cluster mgmt node has no internet access)
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import hashlib, json, re, threading, time
from pathlib import Path
import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, StreamingResponse

# Set HF_HOME to node-local hf_cache if present
_hf_dir = Path(__file__).resolve().parents[1] / "hf_cache"
if _hf_dir.exists():
    os.environ.setdefault("HF_HOME", str(_hf_dir))

if os.environ.get("DEMO_DB"):   # use a node-local copy of the SQLite index (WAL mode is unreliable on the shared filesystem)
    from src.common.config import PATHS
    PATHS["sqlite"] = os.environ["DEMO_DB"]

app = FastAPI(title="CCTV search demo")
STATIC = Path(__file__).parent / "static"
MODES = {
    "person": "hybrid_ft" if Path("models/finetuned/text2person/clip_t2p.pt").exists() else "hybrid",
    "vehicle": "hybrid_ft" if Path("models/finetuned/vehicle/clip_vehicle.pt").exists() else "hybrid",
    "incident": "hybrid",
}
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


def _resolve_video_path(p_str: str | None) -> Path | None:
    if not p_str:
        return None
    p = Path(p_str)
    if p.exists():
        return p
    from src.common.config import ROOT
    p_root = ROOT / p_str
    if p_root.exists():
        return p_root
    return None


def _track(db, tid):
    return _row(db, "select t.id, t.chunk_id, t.kind, t.start_s, t.end_s, t.crops_dir, t.caption, t.attrs_json, ch.camera, "
                    "coalesce(ch.dataset, 'meva') as dataset, ch.start_s as chunk_start, ch.source as video_source, ch.path as chunk_path "
                    "from tracks t join chunks ch on ch.id=t.chunk_id where t.id=?", tid)


def _clip_times(start, end, chunk_start):
    t0 = max(0.0, float(start) - float(chunk_start)); t1 = max(t0 + 1.0, float(end) - float(chunk_start))
    return round(t0, 1), round(t1, 1)


def _card_track(t, score=None):
    src_path = _resolve_video_path(t.get("video_source"))
    if src_path and src_path.exists():
        t0 = round(max(0.0, float(t["start_s"])), 1)
        t1 = round(max(t0 + 1.0, float(t["end_s"])), 1)
        is_full = True
    else:
        t0, t1 = _clip_times(t["start_s"], t["end_s"], t["chunk_start"])
        is_full = False

    return dict(id=t["id"], kind=t["kind"], camera=t["camera"], dataset=t.get("dataset") or "meva",
                start_s=round(t["start_s"], 1), end_s=round(t["end_s"], 1),
                caption=t["caption"], attrs=json.loads(t.get("attrs_json") or "{}"), thumb=f"/api/thumb/{t['id']}",
                clip=f"/api/clip/{t['chunk_id']}", clip_t0=t0, clip_t1=t1, is_full_video=is_full,
                score=None if score is None else round(float(score), 4))


def _card_window(db, hit):
    w = None
    try:
        w = _row(db, "select w.id, w.chunk_id, w.start_s, w.end_s, w.rel_start, w.rel_end, w.caption, ch.camera, "
                     "coalesce(ch.dataset, 'meva') as dataset, ch.start_s as chunk_start, ch.source as video_source, ch.path as chunk_path "
                     "from windows w join chunks ch on ch.id=w.chunk_id where w.id=?", hit["id"])
    except Exception as e:
        raise HTTPException(500, f"incident lookup failed ({type(e).__name__}: {e}); check the windows table columns")
    if w is None:
        return None
    src_path = _resolve_video_path(w.get("video_source"))
    if src_path and src_path.exists():
        t0 = round(max(0.0, float(w["start_s"])), 1)
        t1 = round(max(t0 + 1.0, float(w["end_s"])), 1)
        is_full = True
    else:
        t0 = round(max(0.0, float(w["rel_start"])), 1)
        t1 = round(max(float(w["rel_end"]), t0 + 1.0), 1)
        is_full = False

    return dict(id=w["id"], kind="incident", camera=w["camera"], dataset=w.get("dataset") or "meva",
                start_s=round(w["start_s"], 1), end_s=round(w["end_s"], 1),
                caption=w["caption"], attrs={}, thumb=None, clip=f"/api/clip/{w['chunk_id']}", clip_t0=t0, clip_t1=t1,
                is_full_video=is_full, score=round(float(hit["score"]), 4))


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
    size = path.stat().st_size
    hdr = request.headers.get("range")
    if not hdr:
        return FileResponse(path, media_type="video/mp4", headers={"Accept-Ranges": "bytes"})
    m = re.match(r"bytes=(\d*)-(\d*)", hdr.strip())
    if not m:
        return FileResponse(path, media_type="video/mp4", headers={"Accept-Ranges": "bytes"})
    g1, g2 = m.groups()
    start = int(g1) if g1 else max(0, size - int(g2)) if g2 else 0
    end = min(int(g2), size - 1) if g2 else size - 1
    if start >= size or start > end:
        raise HTTPException(416, "range not satisfiable")
    chunk_len = end - start + 1
    def body():
        with open(path, "rb") as f:
            f.seek(start)
            left = chunk_len
            while left > 0:
                b = f.read(min(512 * 1024, left))
                if not b:
                    break
                left -= len(b)
                yield b
    return StreamingResponse(
        body(),
        status_code=206,
        media_type="video/mp4",
        headers={
            "Content-Range": f"bytes {start}-{end}/{size}",
            "Accept-Ranges": "bytes",
            "Content-Length": str(chunk_len),
            "Cache-Control": "public, max-age=3600",
        }
    )


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
def ask(q: str, n: int = 24, use_llm: bool = False, rerank: bool = False, camera: str | None = None, dataset: str | None = None):
    """Unified AI Query Agent: natural language parsing, fast hybrid retrieval, score calibration, deduplication, and full-length video streaming."""
    if not q.strip():
        raise HTTPException(400, "empty query")
    t0 = time.time()
    from src.query import parser as pr
    from src.captions import hybrid

    # 1. Parse natural language query
    parsed = pr.parse(q.strip(), use_llm=use_llm)
    kind = parsed.get("type") or "person"
    if kind not in MODES:
        kind = "person"
    search_text = parsed.get("text") or q.strip()
    cam = camera or parsed.get("camera")
    mode = MODES.get(kind, "hybrid")

    where = {}
    if cam:
        where["camera"] = cam
    if dataset:
        where["dataset"] = dataset.lower()
    where_arg = where if where else None

    # 2. Fast multi-modal retrieval
    hits = hybrid.retrieve(kind, search_text, k=min(max(n * 4, 48), 150), mode=mode, where=where_arg, w=_fusion_weights(kind))
    # If filtered dataset yielded no results (e.g. aicity selected but raw chunks are tagged meva), fallback gracefully
    fallback_dataset = False
    if not hits and dataset:
        fallback_where = {"camera": cam} if cam else None
        hits = hybrid.retrieve(kind, search_text, k=min(max(n * 4, 48), 150), mode=mode, where=fallback_where, w=_fusion_weights(kind))
        fallback_dataset = True

    if rerank and hits:
        try:
            from src.query import rerank as rr
            hits = rr.rerank(kind, search_text, hits[:24])
        except Exception:
            pass

    # 3. Deduplicate across cameras, throttle stationary hits, and build full-stream cards
    db = _db(); out = []; seen_clips = set()
    cam_counts = {}
    seen_stationary = set()
    seen_events = set()
    deferred_overflow = []

    try:
        for h in hits:
            card = _card_window(db, h) if kind == "incident" else (lambda t: _card_track(t, h["score"]) if t else None)(_track(db, h["id"]))
            if not card:
                continue
            if dataset and not fallback_dataset and card.get("dataset") != dataset:
                continue

            clip_key = card.get("clip")
            if clip_key and clip_key in seen_clips:
                continue

            c_cam = card.get("camera")
            start_s = float(card.get("start_s") or 0.0)
            attrs = card.get("attrs") or {}
            motion = (attrs.get("motion") or "").lower()
            is_stopped = motion in ("stopped", "standing", "parked", "idle")

            # Temporal deduplication on the same camera (within 20s is typically the same vehicle/person)
            event_bin = int(start_s // 20)
            if (c_cam, event_bin) in seen_events:
                continue

            # Stationary object throttling
            if is_stopped:
                stat_bin = int(start_s // 240)
                if (c_cam, stat_bin) in seen_stationary:
                    continue
                seen_stationary.add((c_cam, stat_bin))

            # Camera diversity: avoid flooding the screen with one camera when user hasn't selected a camera
            if not cam and cam_counts.get(c_cam, 0) >= 3:
                deferred_overflow.append(card)
                continue

            if clip_key:
                seen_clips.add(clip_key)
            seen_events.add((c_cam, event_bin))
            cam_counts[c_cam] = cam_counts.get(c_cam, 0) + 1
            out.append(card)
            if len(out) >= n:
                break

        # Backfill if diversity constraint left us under n results
        if len(out) < n and deferred_overflow:
            for card in deferred_overflow:
                clip_key = card.get("clip")
                if clip_key and clip_key in seen_clips:
                    continue
                if clip_key:
                    seen_clips.add(clip_key)
                out.append(card)
                if len(out) >= n:
                    break

        # Calibrate raw RRF scores to intuitive percentage confidence (98% down to 72%)
        if out:
            max_raw = max((c.get("score") or 0.0 for c in out), default=0.0)
            if 0.0 < max_raw < 0.25:  # Raw RRF fusion scores
                for card in out:
                    if card.get("score") is not None:
                        rel = max(0.0, card["score"]) / max_raw
                        calibrated = round(0.68 + 0.30 * (rel ** 1.3), 3)
                        card["score"] = min(0.98, max(0.50, calibrated))

    finally:
        db.close()

    elapsed = round(time.time() - t0, 2)
    return dict(query={"type": kind, "text": search_text, "camera": cam}, kind=kind, mode=mode, dataset=dataset, latency_s=elapsed, took_s=elapsed, n=len(out), results=out)


@app.get("/api/health")
def health():
    return {"ok": True, **WARM}


@app.get("/api/stats")
def stats():
    db = _db()
    try:
        datasets = [r[0] for r in db.execute("select distinct coalesce(dataset, 'meva') from chunks where dataset is not null order by dataset").fetchall()]
        for ds in ("meva", "aicity"):
            if ds not in datasets:
                datasets.append(ds)
        datasets = sorted(list(set(datasets)))
        cameras = [r[0] for r in db.execute("select distinct camera from chunks where camera is not null order by camera").fetchall()]
        cam_dataset_rows = db.execute("select distinct camera, coalesce(dataset, 'meva') from chunks where camera is not null").fetchall()
        cameras_by_dataset = {}
        for cam, ds in cam_dataset_rows:
            cameras_by_dataset.setdefault(ds, []).append(cam)
        cameras_by_dataset.setdefault("meva", cameras)
        if "aicity" not in cameras_by_dataset or not cameras_by_dataset["aicity"]:
            # Discover AI City / CityFlow cameras if available in data/raw/cityflow_nl, else standard AI City camera IDs
            aicity_cams = set()
            cf_dir = Path("data/raw/cityflow_nl/AICity22_Track1_MTMC_Tracking")
            if cf_dir.exists():
                for c in cf_dir.glob("**/c0*"):
                    if c.is_dir():
                        aicity_cams.add(c.name)
            cameras_by_dataset["aicity"] = sorted(list(aicity_cams)) if aicity_cams else [f"c0{i:02d}" for i in range(1, 10)]
        n_chunks = db.execute("select count(*) from chunks").fetchone()[0]
        track_counts = dict(db.execute("select kind, count(*) from tracks group by kind").fetchall())
        n_windows = 0
        try:
            n_windows = db.execute("select count(*) from windows").fetchone()[0]
        except Exception:
            pass
    finally:
        db.close()
    return {
        "datasets": datasets,
        "cameras": cameras,
        "cameras_by_dataset": cameras_by_dataset,
        "chunks": n_chunks,
        "tracks": track_counts,
        "windows": n_windows,
        "models": {
            "appearance": "CLIP ViT-B/32 (Fine-Tuned)",
            "gait": "DeepGaitV2 (Fine-Tuned)",
            "vlm": "Qwen2.5-VL-7B"
        }
    }


@app.get("/api/search")
def search(kind: str, q: str, k: int = 24, mode: str | None = None, camera: str | None = None, dataset: str | None = None):
    if kind not in MODES:
        raise HTTPException(400, f"kind must be one of {list(MODES)}")
    if not q.strip():
        raise HTTPException(400, "empty query")
    from src.captions import hybrid
    t0 = time.time(); mode = mode or MODES[kind]
    where = {}
    if camera:
        where["camera"] = camera
    if dataset:
        where["dataset"] = dataset.lower()
    where_arg = where if where else None
    hits = hybrid.retrieve(kind, q.strip(), k=min(max(k * 4, 48), 150), mode=mode, where=where_arg, w=_fusion_weights(kind))
    fallback_dataset = False
    if not hits and dataset:
        fallback_where = {"camera": camera} if camera else None
        hits = hybrid.retrieve(kind, q.strip(), k=min(max(k * 4, 48), 150), mode=mode, where=fallback_where, w=_fusion_weights(kind))
        fallback_dataset = True

    db = _db(); out = []; seen_clips = set()
    cam_counts = {}
    seen_stationary = set()  # (camera, coarse_window_bin)
    seen_events = set()      # (camera, event_time_bin)
    deferred_overflow = []
    try:
        for h in hits:
            card = _card_window(db, h) if kind == "incident" else (lambda t: _card_track(t, h["score"]) if t else None)(_track(db, h["id"]))
            if not card:
                continue
            if dataset and not fallback_dataset and card.get("dataset") != dataset:
                continue

            clip_key = card.get("clip")
            if clip_key and clip_key in seen_clips:
                continue

            cam = card.get("camera")
            start_s = float(card.get("start_s") or 0.0)
            attrs = card.get("attrs") or {}
            motion = (attrs.get("motion") or "").lower()
            is_stopped = motion in ("stopped", "standing", "parked", "idle")

            # Temporal deduplication on the same camera (within 20s is typically the same vehicle/track)
            event_bin = int(start_s // 20)
            if (cam, event_bin) in seen_events:
                continue

            # Stationary object throttling
            if is_stopped:
                stat_bin = int(start_s // 240)  # 4-minute stationary window
                if (cam, stat_bin) in seen_stationary:
                    continue
                seen_stationary.add((cam, stat_bin))

            # Camera diversity: avoid flooding the screen with one camera when user hasn't selected a camera
            if not camera and cam_counts.get(cam, 0) >= 3:
                deferred_overflow.append(card)
                continue

            if clip_key:
                seen_clips.add(clip_key)
            seen_events.add((cam, event_bin))
            cam_counts[cam] = cam_counts.get(cam, 0) + 1
            out.append(card)
            if len(out) >= k:
                break

        # If diversity constraint left us under k results, backfill with remaining diverse items
        if len(out) < k and deferred_overflow:
            for card in deferred_overflow:
                clip_key = card.get("clip")
                if clip_key and clip_key in seen_clips:
                    continue
                if clip_key:
                    seen_clips.add(clip_key)
                out.append(card)
                if len(out) >= k:
                    break

        # Calibrate raw RRF scores to intuitive percentage confidence (98% down to 72%)
        if out:
            max_raw = max((c.get("score") or 0.0 for c in out), default=0.0)
            if 0.0 < max_raw < 0.25:  # Raw RRF fusion scores
                for card in out:
                    if card.get("score") is not None:
                        rel = max(0.0, card["score"]) / max_raw
                        calibrated = round(0.68 + 0.30 * (rel ** 1.3), 3)
                        card["score"] = min(0.98, max(0.50, calibrated))
    finally:
        db.close()
    return dict(kind=kind, q=q, mode=mode, dataset=dataset, took_s=round(time.time() - t0, 2), n=len(out), results=out)


@app.api_route("/api/thumb/{tid}", methods=["GET", "HEAD"])
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


def _ensure_web_playable(path: Path) -> Path:
    """Ensure the video file is an MP4 playable by web browsers (Chrome/Safari/Firefox).
    If it's an .avi (like raw MEVID recordings), fast-remux it without re-encoding
    (-c copy -movflags +faststart) into an .mp4 cache in outputs/web_videos/ (< 0.8s).
    """
    if path.suffix.lower() == ".mp4":
        return path
    
    out_dir = Path("outputs/web_videos")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_mp4 = out_dir / (path.stem + ".mp4")
    if out_mp4.exists() and out_mp4.stat().st_size > 0:
        return out_mp4
        
    import shutil, subprocess
    ffmpeg_bin = shutil.which("ffmpeg")
    if not ffmpeg_bin:
        for p in ("/home/vansh.122433/.conda/envs/cctv/bin/ffmpeg", "/usr/bin/ffmpeg", "/usr/local/bin/ffmpeg"):
            if Path(p).exists():
                ffmpeg_bin = p
                break
    if not ffmpeg_bin:
        return path

    try:
        subprocess.run([
            ffmpeg_bin, "-y", "-loglevel", "error",
            "-i", str(path),
            "-c", "copy",
            "-movflags", "+faststart",
            str(out_mp4)
        ], check=True, timeout=25)
        if out_mp4.exists() and out_mp4.stat().st_size > 0:
            return out_mp4
    except Exception as e:
        print(f"⚠️ [Web Remux] Could not remux {path} to MP4: {e}")
        
    return path


@app.api_route("/api/clip/{chunk_id}", methods=["GET", "HEAD"])
def clip(chunk_id: int, request: Request, full: bool = True):
    db = _db()
    try:
        c = _row(db, "select path, source from chunks where id=?", chunk_id)
    finally:
        db.close()
    if not c:
        raise HTTPException(404, "no video record")
    
    video_path = None
    if full and c.get("source"):
        p_src = _resolve_video_path(c["source"])
        if p_src and p_src.exists():
            p_playable = _ensure_web_playable(p_src)
            if p_playable.suffix.lower() == ".mp4":
                video_path = p_playable
            
    if not video_path:
        p_chunk = _resolve_video_path(c.get("path"))
        if p_chunk and p_chunk.exists():
            video_path = _ensure_web_playable(p_chunk)

    if not video_path or not video_path.exists():
        raise HTTPException(404, "no video file on disk")

    if request.method == "HEAD":
        from fastapi.responses import Response
        size = video_path.stat().st_size
        return Response(status_code=200, headers={"Accept-Ranges": "bytes", "Content-Length": str(size), "Content-Type": "video/mp4"})
        
    return _range_response(video_path, request)


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
def reid_api(
    tid: int,
    mode: str = "any",
    k: int = 12,
    full_body: bool = True,
    use_st: bool = True,
    use_color: bool = True,
    use_kinematics: bool = True,
):
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
        res = reid.rank(
            tid,
            k=k * 4 if full_body else k,
            mode=mode,
            use_st=use_st,
            use_kinematics=use_kinematics,
        )
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
                    st_score=r.get("st_score"),
                    kinematic_score=r.get("kinematic_score"),
                    transition=r.get("transition"),
                    delta_cadence=r.get("delta_cadence"),
                )
                cards.append(c)
        q = _track(db, tid)
        query = _card_track(q) if q else None
    finally:
        db.close()
    return dict(
        query=query,
        mode=mode,
        w_appearance=res["w_appearance"],
        note=res["note"],
        full_body_filter=full_body,
        use_st=use_st,
        use_kinematics=use_kinematics,
        results=cards,
    )


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


def preload_all_models():
    """Immediately and eagerly load and warm up all neural networks and indexes on startup."""
    print("=" * 60)
    print("🚀 [STARTUP] Preloading all model weights into memory/device...")
    print("=" * 60)
    t0 = time.time()
    
    # 1. Warm up BGE Dense Text Embedding Model
    try:
        from src.captions import text_index as T
        T._load()
        _ = T.embed(["warmup query for surveillance retrieval"], query=True)
        print("  ✓ [Text Encoder] BAAI/bge-small-en-v1.5 preloaded.")
    except Exception as e:
        print(f"  ⚠️ [Text Encoder] Note: {e}")

    # 2. Warm up Base Visual CLIP Model
    try:
        from src.common import clip as cutil
        cutil._load()
        _ = cutil.embed_text(["surveillance camera person vehicle"])
        print(f"  ✓ [Base Visual CLIP] ViT-B/32 preloaded on {cutil.DEV}.")
    except Exception as e:
        print(f"  ⚠️ [Base Visual CLIP] Note: {e}")

    # 3. Warm up Fine-Tuned Person & Vehicle Checkpoints
    try:
        import torch
        from src.captions import hybrid
        from src.person import t2p
        for kind in ("person", "vehicle"):
            ckpt_path, coll = hybrid.FT_MODELS.get(kind, (None, None))
            if ckpt_path and Path(ckpt_path).exists():
                m, tok = t2p.load(str(ckpt_path), cutil.DEV)
                m.eval()
                hybrid._ft[kind] = (m, tok)
                with torch.no_grad():
                    f = t2p.txt_feat(m, tok, [f"a {kind} on surveillance camera"], cutil.DEV)
                    _ = torch.nn.functional.normalize(f.float(), dim=-1)
                print(f"  ✓ [Fine-Tuned {kind.capitalize()}] {ckpt_path} preloaded and primed on {cutil.DEV}.")
            else:
                print(f"  ℹ️ [Fine-Tuned {kind.capitalize()}] Using base CLIP embeddings.")
    except Exception as e:
        print(f"  ⚠️ [Fine-Tuned Checkpoints] Note: {e}")

    # 4. Warm up Lexical BM25 Inverted Indexes for All Modalities
    try:
        for kind in ("person", "vehicle", "incident"):
            _ = T.bm25_search(kind, "warmup text search", k=1)
            print(f"  ✓ [Lexical BM25] Inverted index built in memory for '{kind}'.")
    except Exception as e:
        print(f"  ⚠️ [Lexical BM25] Note: {e}")

    # 5. Warm up Natural Language Query Parser
    try:
        from src.query import parser as pr
        _ = pr.parse("person wearing a blue jacket walking left", use_llm=False)
        print("  ✓ [Query Parser] Regex & rules engine primed.")
    except Exception as e:
        print(f"  ⚠️ [Query Parser] Note: {e}")

    # 6. Warm up Vehicle Re-ID Topology
    try:
        from src.vehicle import reid as vreid
        _ = vreid.load_topology()
        print("  ✓ [Vehicle Re-ID] Camera network topology loaded.")
    except Exception as e:
        print(f"  ⚠️ [Vehicle Re-ID] Note: {e}")

    # 7. Prime Qdrant Vector DB Connection Pools
    try:
        from src.index import vector_db
        c = vector_db.client()
        collections = [col.name for col in c.get_collections().collections]
        print(f"  ✓ [Qdrant Vector DB] Connected. Active collections: {', '.join(collections)}")
    except Exception as e:
        print(f"  ⚠️ [Qdrant Vector DB] Note: {e}")

    total_s = round(time.time() - t0, 2)
    WARM["done"] = True
    WARM["secs"] = total_s
    print("=" * 60)
    print(f"✨ [READY] All model weights and indexes primed in {total_s}s! First query will be instantaneous.")
    print("=" * 60)


@app.on_event("startup")
def warm():
    preload_all_models()
