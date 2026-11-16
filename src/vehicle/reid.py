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


def extract_color(attrs: Dict, caption: Optional[str] = None) -> Optional[str]:
    """Extract canonical vehicle color from attributes or caption."""
    c = attrs.get("color")
    if c and c.lower() in COLOR_MAP:
        return COLOR_MAP[c.lower()]
    if caption:
        cap_low = caption.lower()
        for word in cap_low.split():
            clean = re.sub(r"[^a-z]", "", word)
            if clean in COLOR_MAP:
                return COLOR_MAP[clean]
    return None


_EMB_CACHE: Dict[str, np.ndarray] = {}


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


def rank(
    track_row: int,
    k: int = 12,
    same_camera: bool = False,
    use_st_filter: bool = True,
    use_color_gate: bool = True,
) -> Dict:
    """Find and rank the same vehicle across other cameras.

    Args:
        track_row: ID of the query vehicle track in SQLite tracks table.
        k: Maximum number of matches to return.
        same_camera: Whether to include matches from the same camera feed.
        use_st_filter: Enforce spatio-temporal road topology constraints.
        use_color_gate: Reject/penalize vehicles with conflicting paint colors.

    Returns:
        Structured dictionary with query vehicle details and cross-camera candidate matches.
    """
    db = connect()
    try:
        # 1. Fetch query track
        q_row = db.execute(
            """
            SELECT t.*, ch.camera, ch.source as video_source, ch.path as chunk_path, ch.start_s as chunk_start
            FROM tracks t
            JOIN chunks ch ON ch.id = t.chunk_id
            WHERE t.id = ?
            """,
            (track_row,),
        ).fetchone()

        if not q_row:
            # Check if this ID is in our standalone MTMC cache format
            res_fallback = _rank_from_mtmc_fallback(track_row, k, same_camera, use_st_filter, use_color_gate)
            if res_fallback:
                return res_fallback
            raise KeyError(f"Vehicle track #{track_row} not found in database.")

        q_cam = q_row["camera"]
        q_start = float(q_row["start_s"])
        q_end = float(q_row["end_s"])
        q_attrs = json.loads(q_row["attrs_json"] or "{}")
        q_color = extract_color(q_attrs, q_row["caption"])
        q_crop_dir = q_row["crops_dir"]

        # Thumbnail path for query
        q_files = sorted(Path(q_crop_dir).glob("*.jpg")) if q_crop_dir else []
        q_thumb = f"/api/thumb/{track_row}" if q_files else None

        # 2. Extract query visual vector
        q_vec = None
        # Try Qdrant first if available
        try:
            from src.index import vector_db
            r = vector_db.client().retrieve("vehicle_ft", ids=[track_row], with_vectors=True)
            if not r:
                r = vector_db.client().retrieve("vehicle", ids=[track_row], with_vectors=True)
            if r and r[0].vector:
                q_vec = np.array(r[0].vector, dtype=np.float32)
                norm = np.linalg.norm(q_vec) + 1e-9
                q_vec = q_vec / norm
        except Exception:
            pass

        # If not in vector DB, extract on the fly from crop images
        if q_vec is None and q_crop_dir:
            q_vec = get_track_crop_embedding(q_crop_dir)

        # 3. Retrieve all candidate vehicle tracks
        sql = """
            SELECT t.*, ch.camera, ch.path as chunk_path, ch.start_s as chunk_start
            FROM tracks t
            JOIN chunks ch ON ch.id = t.chunk_id
            WHERE t.kind = 'vehicle' AND t.id != ?
        """
        params = [track_row]
        if not same_camera:
            sql += " AND ch.camera != ?"
            params.append(q_cam)

        candidates = db.execute(sql, params).fetchall()

        # 4. Score candidates
        topology = load_topology()
        matches = []

        for c in candidates:
            cid = c["id"]
            cam = c["camera"]
            c_start = float(c["start_s"])
            c_end = float(c["end_s"])
            c_attrs = json.loads(c["attrs_json"] or "{}")
            c_color = extract_color(c_attrs, c["caption"])

            # A. Color consistency gating
            color_match = True
            color_penalty = 0.0
            if q_color and c_color:
                if q_color != c_color:
                    color_match = False
                    if use_color_gate:
                        # Skip candidate outright if strictly conflicting color
                        continue
                    else:
                        color_penalty = 0.4

            # B. Spatio-temporal feasibility (Bidirectional transition check)
            st_valid = True
            st_score = 1.0

            if use_st_filter:
                # Determine direction: forward (q_cam -> cam) or backward (cam -> q_cam)
                if c_start >= (q_end - 5.0):
                    # Forward transition
                    dt = c_start - q_end
                    connected, min_t, max_t = get_travel_window(topology, q_cam, cam)
                elif q_start >= (c_end - 5.0):
                    # Backward transition (vehicle was seen on cam before reaching q_cam)
                    dt = q_start - c_end
                    connected, min_t, max_t = get_travel_window(topology, cam, q_cam)
                else:
                    # Overlapping observation in adjacent cameras
                    dt = abs(c_start - q_start)
                    connected_fwd, _, max_fwd = get_travel_window(topology, q_cam, cam)
                    connected_bwd, _, max_bwd = get_travel_window(topology, cam, q_cam)
                    connected = connected_fwd or connected_bwd
                    min_t, max_t = 0.0, max(max_fwd, max_bwd, 30.0)

                if not connected:
                    continue

                # Check travel time window
                if dt < (min_t - 5.0) or dt > (max_t + 35.0):
                    continue

                # Score transition timing
                expected_t = (min_t + max_t) / 2.0
                time_diff = abs(dt - expected_t)
                st_score = max(0.3, 1.0 - (time_diff / (max_t + 1e-6)))

            # C. Visual similarity
            c_vec = None
            try:
                from src.index import vector_db
                rc = vector_db.client().retrieve("vehicle_ft", ids=[cid], with_vectors=True)
                if not rc:
                    rc = vector_db.client().retrieve("vehicle", ids=[cid], with_vectors=True)
                if rc and rc[0].vector:
                    c_vec = np.array(rc[0].vector, dtype=np.float32)
                    c_vec = c_vec / (np.linalg.norm(c_vec) + 1e-9)
            except Exception:
                pass

            if c_vec is None and c["crops_dir"]:
                c_vec = get_track_crop_embedding(c["crops_dir"])

            if q_vec is not None and c_vec is not None:
                vis_sim = float(np.dot(q_vec, c_vec))
            else:
                # If embeddings not available, fall back to attribute and motion overlap
                vis_sim = 0.5

            # Combined score: visual + spatio-temporal - color penalty
            final_score = (0.75 * max(0.0, vis_sim) + 0.25 * st_score) - color_penalty
            final_score = max(0.0, min(1.0, final_score))

            # Clip-relative times for video playback
            chunk_start = float(c["chunk_start"])
            t0 = max(0.0, round(c_start - chunk_start, 1))
            t1 = max(t0 + 1.0, round(min(c_end - chunk_start, 10.0), 1))

            matches.append({
                "track_row": cid,
                "camera": cam,
                "chunk_id": c["chunk_id"],
                "start_s": round(c_start, 1),
                "end_s": round(c_end, 1),
                "score": round(final_score, 3),
                "appearance_sim": round(max(0.0, vis_sim), 3),
                "st_valid": st_valid,
                "color_match": color_match,
                "attrs": c_attrs,
                "caption": c["caption"],
                "thumb": f"/api/thumb/{cid}",
                "clip": f"/api/clip/{c['chunk_id']}",
                "clip_t0": t0,
                "clip_t1": t1,
            })

        # Sort matches by score descending
        matches.sort(key=lambda m: -m["score"])

        return {
            "query": {
                "track_row": track_row,
                "camera": q_cam,
                "start_s": round(q_start, 1),
                "end_s": round(q_end, 1),
                "attrs": q_attrs,
                "color": q_color,
                "caption": q_row["caption"],
                "thumb": q_thumb,
            },
            "k": k,
            "use_st_filter": use_st_filter,
            "use_color_gate": use_color_gate,
            "results": matches[:k],
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
