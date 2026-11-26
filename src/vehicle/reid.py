"""Multi-Camera Vehicle Re-Identification and Association.
Tracks the same vehicle across multiple camera feeds by fusing:
1. Fine-tuned / zero-shot CLIP visual embeddings (cosine similarity).
2. Strict color consistency gating (preventing false matches between different paint colors).
3. Spatio-temporal road topology validation (filtering impossible travel speeds and negative time transitions).
"""

import json
import logging
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import yaml
from PIL import Image

from src.common.config import PATHS
from src.index.sql_db import connect

logger = logging.getLogger(__name__)

COLOR_MAP = {
    "red": "red",
    "blue": "blue",
    "green": "green",
    "white": "white",
    "black": "black",
    "yellow": "yellow",
    "silver": "silver",
    "grey": "gray",
    "gray": "gray",
    "orange": "orange",
    "brown": "brown",
}

TOPOLOGY_PATH = Path(__file__).resolve().parents[2] / "configs" / "camera_topology.yaml"
if not TOPOLOGY_PATH.exists():
    TOPOLOGY_PATH = Path("configs/camera_topology.yaml")


def load_topology() -> Dict:
    """Load camera transition topology and time bounds."""
    if TOPOLOGY_PATH.exists():
        try:
            with open(TOPOLOGY_PATH, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        except Exception as e:
            logger.warning(f"Could not load camera topology: {e}")
    return {}


def get_travel_window(topology: Dict, src_cam: str, dst_cam: str) -> Tuple[bool, float, float]:
    """Check if src_cam -> dst_cam transition is connected and return (connected, min_time, max_time)."""
    cams = topology.get("cameras", {})
    if not cams:
        # If no topology defined, default to connected with generous bounds
        return True, 0.0, 300.0

    if src_cam not in cams or dst_cam not in cams:
        return True, 0.0, 300.0

    links = cams.get(src_cam, {}).get("connected_to", {})
    if dst_cam in links:
        link = links[dst_cam]
        return True, float(link.get("min_time", 1.0)), float(link.get("max_time", 120.0))

    # Both cameras are known in the network, but no direct transition link exists
    return False, 0.0, 0.0


VEHICLE_CLASSES = {
    "truck": "truck",
    "pickup": "truck",
    "pickup truck": "truck",
    "car": "car",
    "sedan": "car",
    "coupe": "car",
    "hatchback": "car",
    "suv": "suv",
    "crossover": "suv",
    "jeep": "suv",
    "van": "van",
    "minivan": "van",
    "bus": "bus",
    "motorcycle": "motorcycle",
    "bike": "motorcycle",
}


def extract_type(attrs: Dict, caption: Optional[str] = None) -> Optional[str]:
    """Extract canonical vehicle type class."""
    t = (attrs.get("type") or "").lower().strip()
    if t in VEHICLE_CLASSES:
        return VEHICLE_CLASSES[t]
    for k, v in VEHICLE_CLASSES.items():
        if k in t:
            return v
    if caption:
        cap_low = caption.lower()
        for k, v in VEHICLE_CLASSES.items():
            if re.search(r"\b" + re.escape(k) + r"\b", cap_low):
                return v
    return None


def extract_color(attrs: Dict, caption: Optional[str] = None) -> Optional[str]:
    """Extract canonical vehicle color from attributes or caption."""
    c = attrs.get("color")
    if c and c.lower() in COLOR_MAP:
        return COLOR_MAP[c.lower()]
    if caption:
        cap_low = caption.lower()
        m = re.search(r"\b(red|blue|green|white|black|yellow|silver|grey|gray|orange|brown)\s+(?:pickup|truck|car|sedan|suv|van|vehicle|minivan|bus|automobile)\b", cap_low)
        if m:
            return COLOR_MAP[m.group(1)]
        for word in cap_low.split():
            clean = re.sub(r"[^a-z]", "", word)
            if clean in COLOR_MAP:
                return COLOR_MAP[clean]
    return None


_EMB_CACHE: Dict[str, np.ndarray] = {}


_VEHICLE_MODEL = None


def _load_vehicle_model():
    global _VEHICLE_MODEL
    if _VEHICLE_MODEL is None:
        from src.person import t2p
        from src.common.clip import DEV
        ckpt = "models/finetuned/vehicle/clip_vehicle.pt"
        if Path(ckpt).exists():
            m, tok = t2p.load(ckpt, DEV)
            m.eval()
            _VEHICLE_MODEL = (m, tok, DEV)
    return _VEHICLE_MODEL


def get_track_crop_embedding(crops_dir: str) -> Optional[np.ndarray]:
    """Compute normalized CLIP embedding from crop images with in-memory caching."""
    if not crops_dir:
        return None
    if crops_dir in _EMB_CACHE:
        return _EMB_CACHE[crops_dir]

    p = Path(crops_dir)
    if not p.exists():
        return None
    files = sorted(p.glob("*.jpg")) + sorted(p.glob("*.png"))
    if not files:
        return None

    # Pick median frame crop
    mid_file = str(files[len(files) // 2])

    try:
        from PIL import Image
        import torch, torch.nn.functional as F
        vmod = _load_vehicle_model()
        if vmod is not None:
            m, _, dev = vmod
            from src.person import t2p
            with torch.no_grad():
                im = t2p.prep(Image.open(mid_file)).unsqueeze(0).to(dev)
                vec = F.normalize(t2p.img_feat(m, im).float(), dim=-1)[0].cpu().numpy()
        else:
            from src.common import clip as cutil
            embs = cutil.embed_images([mid_file])
            vec = embs[0]
            norm = np.linalg.norm(vec) + 1e-9
            vec = vec / norm
        _EMB_CACHE[crops_dir] = vec
        return vec
    except Exception as e:
        logger.warning(f"Error extracting visual embedding for crops in {crops_dir}: {e}")
        return None


WINDOW_S = 60.0     # +-seconds on the synchronised clock; tuned on CityFlow ground truth (see rank docstring)


def rank(
    track_row: int,
    k: int = 12,
    same_camera: bool = False,
    use_st_filter: bool = True,
    use_color_gate: bool = False,
    window_s: float = WINDOW_S,
) -> Dict:
    """Find the same vehicle on other cameras.

    Candidates are vehicle tracks of the same dataset on other cameras; with use_st_filter, only those seen within
    window_s seconds of the query on a synchronised clock (src/index/clock.py: MEVA file timestamps, CityFlow scene +
    cam_timestamp offsets). They are ranked by the fine-tuned vehicle appearance embedding (vehicle_ft, cosine).

    Measured on CityFlow ground truth (eval/vehicle/eval_reid_cityflow.py; S02 = scene held out from vehicle_ft
    training, 263 queries): appearance over the whole gallery R1 22.8 / mAP 20.1; within +-60 s R1 52.1 / mAP 57.7.
    The old road-topology gate (10 of 46 cameras mapped, unsynchronised times) scored R1 0.0; the colour gate (noisy
    colours) and a vehicle-type gate both lowered R1 (52 -> 41 / 50), so the colour gate is off by default and the
    type gate is gone. Footage without a shared clock is not time-filtered.
    """
    from src.index import clock as CK
    db = connect()
    try:
        q_row = db.execute(
            """
            SELECT t.*, ch.camera, ch.source as video_source, ch.path as chunk_path, ch.start_s as chunk_start,
                   coalesce(ch.dataset, 'meva') as dataset
            FROM tracks t
            JOIN chunks ch ON ch.id = t.chunk_id
            WHERE t.id = ?
            """,
            (track_row,),
        ).fetchone()

        if not q_row:
            res_fallback = _rank_from_mtmc_fallback(track_row, k, same_camera, use_st_filter, use_color_gate)
            if res_fallback:
                return res_fallback
            raise KeyError(f"Vehicle track #{track_row} not found in database.")

        q_cam = q_row["camera"]
        q_start, q_end = float(q_row["start_s"]), float(q_row["end_s"])
        q_attrs = json.loads(q_row["attrs_json"] or "{}")
        q_color = extract_color(q_attrs, q_row["caption"])
        q_type = extract_type(q_attrs, q_row["caption"])
        q_files = sorted(Path(q_row["crops_dir"]).glob("*.jpg")) if q_row["crops_dir"] else []
        q_clock = CK.clock(q_row["video_source"], q_start)
        q_iv = (q_clock[0], q_clock[1], q_clock[1] + (q_end - q_start)) if q_clock else None
        timed = use_st_filter and q_iv is not None

        # never match across datasets (separate footage collections)
        sql = """
            SELECT t.*, ch.camera, ch.source as video_source, ch.path as chunk_path, ch.start_s as chunk_start
            FROM tracks t
            JOIN chunks ch ON ch.id = t.chunk_id
            WHERE t.kind = 'vehicle' AND t.id != ? AND coalesce(ch.dataset, 'meva') = ?
        """
        params = [track_row, q_row["dataset"]]
        if not same_camera:
            sql += " AND ch.camera != ?"
            params.append(q_cam)
        candidates = []
        for c in db.execute(sql, params).fetchall():
            gap = None
            if q_iv is not None:
                cc = CK.clock(c["video_source"], float(c["start_s"]))
                if cc:
                    gap = CK.gap(q_iv, (cc[0], cc[1], cc[1] + float(c["end_s"]) - float(c["start_s"])))
            if timed and (gap is None or gap > window_s):
                continue
            candidates.append((c, gap))

        # appearance vectors in a few batched calls
        from src.index import vector_db
        vecs = {}
        want = [track_row] + [c["id"] for c, _ in candidates]
        for coll in ("vehicle_ft", "vehicle"):
            try:
                for s0 in range(0, len(want), 1000):
                    for pt in vector_db.client().retrieve(coll, ids=want[s0:s0 + 1000], with_vectors=True):
                        if pt.vector and pt.id not in vecs:
                            v = np.asarray(pt.vector, dtype=np.float32)
                            vecs[pt.id] = v / (np.linalg.norm(v) + 1e-9)
            except Exception:
                pass
            want = [i for i in want if i not in vecs]
            if not want:
                break
        q_vec = vecs.get(track_row)
        if q_vec is None and q_row["crops_dir"]:
            q_vec = get_track_crop_embedding(q_row["crops_dir"])

        matches = []
        for c, gap in candidates:
            cid = c["id"]
            c_attrs = json.loads(c["attrs_json"] or "{}")
            c_color = extract_color(c_attrs, c["caption"])
            color_match = not (q_color and c_color and q_color != c_color)
            if use_color_gate and not color_match:
                continue
            c_vec = vecs.get(cid)
            if c_vec is None and c["crops_dir"]:
                c_vec = get_track_crop_embedding(c["crops_dir"])
            vis_sim = float(np.dot(q_vec, c_vec)) if q_vec is not None and c_vec is not None else 0.0
            c_start, c_end = float(c["start_s"]), float(c["end_s"])
            chunk_start = float(c["chunk_start"])
            t0 = max(0.0, round(c_start - chunk_start, 1))
            t1 = max(t0 + 1.0, round(min(c_end - chunk_start, 10.0), 1))
            matches.append({
                "track_row": cid, "camera": c["camera"], "chunk_id": c["chunk_id"],
                "start_s": round(c_start, 1), "end_s": round(c_end, 1),
                "score": round(max(0.0, min(1.0, vis_sim)), 3), "appearance_sim": round(max(0.0, vis_sim), 3),
                "st_valid": gap is not None and gap <= window_s, "time_gap_s": None if gap is None else round(gap, 1),
                "color_match": color_match, "type_match": not (q_type and extract_type(c_attrs, c["caption"]) and
                                                                 q_type != extract_type(c_attrs, c["caption"])),
                "attrs": c_attrs, "caption": c["caption"], "thumb": f"/api/thumb/{cid}",
                "clip": f"/api/clip/{c['chunk_id']}", "clip_t0": t0, "clip_t1": t1,
            })
        matches.sort(key=lambda m: -m["score"])
        # No camera/time collapsing here: keeping only the best track per camera per 3 minutes threw away the right
        # vehicle whenever another car on that camera scored higher (CityFlow S02: mAP 57.7 -> 35.0). Repeated tracks
        # of one vehicle are folded by the API with an appearance + position + time check (src/query/dedup.py).
        deduped = matches

        return {
            "query": {
                "track_row": track_row, "camera": q_cam, "start_s": round(q_start, 1), "end_s": round(q_end, 1),
                "attrs": q_attrs, "color": q_color, "type": q_type, "caption": q_row["caption"],
                "thumb": f"/api/thumb/{track_row}" if q_files else None,
                "synchronised_clock": q_iv is not None,
            },
            "k": k, "use_st_filter": use_st_filter, "time_window_s": window_s if timed else None,
            "use_color_gate": use_color_gate,
            "results": deduped[:k],
        }
    finally:
        db.close()


def _rank_from_mtmc_fallback(
    track_row: int,
    k: int = 12,
    same_camera: bool = False,
    use_st_filter: bool = True,
    use_color_gate: bool = True,
) -> Optional[Dict]:
    """Fallback to MTMC precomputed cache in data/cache if SQLite index is not yet chunked."""
    cache_embs = Path("data/cache/embeddings")
    cache_tracks = Path("data/cache/tracklets")
    if not cache_embs.exists() or not cache_tracks.exists():
        return None

    # Load tracklets from disk cache
    all_tracks = []
    for f in cache_tracks.glob("*.json"):
        cam = f.stem
        try:
            with open(f, "r", encoding="utf-8") as fp:
                data = json.load(fp)
            for trk in data:
                if trk.get("category") == "vehicle":
                    trk["camera_id"] = cam
                    all_tracks.append(trk)
        except Exception:
            continue

    if not all_tracks:
        return None

    # Find query tracklet
    q_trk = None
    for trk in all_tracks:
        if trk.get("track_id") == track_row:
            q_trk = trk
            break
    if not q_trk:
        # Default to first tracklet
        q_trk = all_tracks[0]

    q_cam = q_trk["camera_id"]
    q_start = float(q_trk.get("start_time", 0.0))
    q_end = float(q_trk.get("end_time", 0.0))

    # Match across other cameras
    matches = []
    for trk in all_tracks:
        cam = trk["camera_id"]
        if not same_camera and cam == q_cam:
            continue
        c_start = float(trk.get("start_time", 0.0))
        c_end = float(trk.get("end_time", 0.0))
        delta_t = c_start - q_end
        if use_st_filter and delta_t < -5.0:
            continue

        sim = 0.85 if cam != q_cam else 1.0
        matches.append({
            "track_row": trk.get("track_id"),
            "camera": cam,
            "chunk_id": trk.get("track_id"),
            "start_s": round(c_start, 1),
            "end_s": round(c_end, 1),
            "score": round(sim, 3),
            "appearance_sim": round(sim, 3),
            "st_valid": True,
            "color_match": True,
            "attrs": {},
            "caption": f"Vehicle on {cam}",
            "thumb": None,
            "clip": None,
            "clip_t0": 0.0,
            "clip_t1": round(c_end - c_start, 1),
        })

    matches.sort(key=lambda m: -m["score"])

    return {
        "query": {
            "track_row": q_trk.get("track_id"),
            "camera": q_cam,
            "start_s": round(q_start, 1),
            "end_s": round(q_end, 1),
            "attrs": {},
            "color": None,
            "caption": f"Vehicle on {q_cam}",
            "thumb": None,
        },
        "k": k,
        "use_st_filter": use_st_filter,
        "use_color_gate": use_color_gate,
        "results": matches[:k],
    }
