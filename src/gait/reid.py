"""Same person on other cameras: clothing-invariant Re-ID fusing appearance (person_ft, Cosine),
2D skeleton pose-guided gait biometrics (person_gait, Cosine), anthropometric kinematics,
and spatio-temporal camera network topology.

Modes:
- 'any': Appearance-heavy (0.75) + Skeleton Gait Biometric refinement (0.25)
- 'cloth_change': Multi-cue CC-ReID fusing 2D skeleton gait biometrics, anthropometric stature &
  cadence tolerance gating, and spatio-temporal camera topology transition windows.
- 'appearance_only': 100% Visual Appearance + Spatio-Temporal validation.
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import yaml
from qdrant_client import models

from src.index import vector_db

logger = logging.getLogger(__name__)

MODES = {"any": 0.75, "cloth_change": 0.20, "appearance_only": 1.0}
FIELDS = [
    "camera",
    "chunk_id",
    "start_s",
    "end_s",
    "motion",
    "cadence_bpm",
    "gait_symmetry",
    "step_length_norm",
    "body_aspect_ratio",
    "aspect_ratio",
]
WALK = {"walking", "running"}

TOPOLOGY_PATH = Path(__file__).resolve().parents[2] / "configs" / "camera_topology.yaml"
if not TOPOLOGY_PATH.exists():
    TOPOLOGY_PATH = Path("configs/camera_topology.yaml")


def load_topology() -> Dict:
    """Load multi-camera transition topology and time bounds."""
    if TOPOLOGY_PATH.exists():
        try:
            with open(TOPOLOGY_PATH, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        except Exception as e:
            logger.warning(f"Could not load camera topology: {e}")
    return {}


def get_pedestrian_travel_window(topology: Dict, src_cam: str, dst_cam: str) -> Tuple[bool, float, float]:
    """Check if src_cam -> dst_cam pedestrian transition is connected and return (connected, min_time, max_time).

    Pedestrians typically walk at 1.0 - 1.8 m/s (jogging up to 3.5 m/s), so travel times are
    calibrated to human walking speeds rather than motor vehicle speeds.
    """
    cams = topology.get("cameras", {})
    if not cams:
        # Default generous bounds if no topology configured
        return True, 0.0, 600.0

    if src_cam not in cams or dst_cam not in cams:
        # Unknown camera in topology: allow with generous default window
        def_tw = topology.get("default_travel_window", {})
        min_t = float(def_tw.get("min_time", 1.0))
        max_t = float(def_tw.get("max_time", 120.0)) * 2.5
        return True, min_t, max(max_t, 300.0)

    links = cams.get(src_cam, {}).get("connected_to", {})
    if dst_cam in links:
        link = links[dst_cam]
        if "distance_meters" in link:
            dist = float(link["distance_meters"])
            # Human walking/running speed bounds
            min_walk = max(1.0, dist / 3.5)              # Running speed ~3.5 m/s
            max_walk = max(45.0, (dist / 0.8) + 60.0)    # Leisurely walk ~0.8 m/s + pause
            return True, round(min_walk, 1), round(max_walk, 1)

        # Scale vehicle bounds for pedestrians
        min_t = float(link.get("min_time", 2.0)) * 2.0
        max_t = float(link.get("max_time", 60.0)) * 3.5 + 45.0
        return True, min_t, max_t

    # Both cameras are known in the network topology graph, but have no direct pedestrian connection
    return False, 0.0, 0.0


def validate_spatio_temporal(
    topology: Dict,
    src_cam: str,
    dst_cam: str,
    q_start: float,
    q_end: float,
    c_start: float,
    c_end: float,
) -> Tuple[bool, float, str]:
    """Validates pedestrian transition between cameras and timestamps.

    Returns:
        (valid, st_score, transition_description)
    """
    if c_start >= (q_end - 5.0):
        # Forward transition (query -> candidate)
        dt = c_start - q_end
        connected, min_t, max_t = get_pedestrian_travel_window(topology, src_cam, dst_cam)
        direction = f"{src_cam} → {dst_cam}"
    elif q_start >= (c_end - 5.0):
        # Backward transition (candidate -> query)
        dt = q_start - c_end
        connected, min_t, max_t = get_pedestrian_travel_window(topology, dst_cam, src_cam)
        direction = f"{dst_cam} → {src_cam}"
    else:
        # Overlapping or simultaneous observation
        dt = abs(c_start - q_start)
        c1, _, max1 = get_pedestrian_travel_window(topology, src_cam, dst_cam)
        c2, _, max2 = get_pedestrian_travel_window(topology, dst_cam, src_cam)
        connected = c1 or c2
        min_t, max_t = 0.0, max(max1, max2, 60.0)
        direction = f"{src_cam} ⟷ {dst_cam}"

    if not connected:
        return False, 0.0, f"{direction} (no topological connection)"

    if dt < (min_t - 8.0):
        return False, 0.0, f"{direction} (impossible speed: {round(dt, 1)}s < min {round(min_t, 1)}s)"
    if dt > (max_t + 60.0):
        return False, 0.0, f"{direction} (expired travel window: {round(dt, 1)}s > max {round(max_t, 1)}s)"

    t_mid = (min_t + max_t) / 2.0
    st_score = max(0.35, 1.0 - abs(dt - t_mid) / (max_t + 1e-5))
    return True, st_score, f"{direction} (+{round(dt, 1)}s, feasible)"


def validate_kinematics(
    q_cadence: Optional[float],
    c_cadence: Optional[float],
    q_aspect: Optional[float],
    c_aspect: Optional[float],
    strict_cloth_change: bool = False,
) -> Tuple[bool, float, Optional[float]]:
    """Evaluates physical kinematic consistency: cadence and body aspect ratio.

    Returns:
        (valid, kinematic_score, delta_cadence)
    """
    penalty = 0.0
    delta_cad = None

    if q_cadence is not None and c_cadence is not None and q_cadence > 0 and c_cadence > 0:
        delta_cad = round(abs(float(q_cadence) - float(c_cadence)), 1)
        if delta_cad > 18.0:
            penalty += min(0.35, (delta_cad - 18.0) / 40.0)
        if strict_cloth_change and delta_cad > 35.0:
            return False, 0.0, delta_cad

    if q_aspect is not None and c_aspect is not None and q_aspect > 0 and c_aspect > 0:
        delta_ar = abs(float(q_aspect) - float(c_aspect))
        if delta_ar > 0.5:
            penalty += min(0.30, (delta_ar - 0.5) / 1.5)
        if strict_cloth_change and delta_ar > 1.2:
            return False, 0.0, delta_cad

    kin_score = max(0.20, 1.0 - penalty)
    return True, kin_score, delta_cad


def _vec(coll: str, tid: int):
    try:
        r = vector_db.client().retrieve(coll, ids=[tid], with_vectors=True)
        return r[0].vector if r else None
    except Exception:
        return None


def _all_scores(coll: str, vec):
    if vec is None:
        return {}
    c = vector_db.client()
    try:
        n = c.count(coll, exact=True).count
        if n == 0:
            return {}
        pts = c.query_points(
            coll,
            query=vec,
            limit=n,
            with_payload=FIELDS,
            search_params=models.SearchParams(exact=True),
        ).points
        return {p.id: (float(p.score), p.payload) for p in pts}
    except Exception:
        return {}


def _track_info_sql(track_id: int) -> Optional[Dict]:
    """Retrieve detailed track metadata from SQLite database if available."""
    try:
        from src.index.sql_db import connect
        db = connect()
        row = db.execute(
            """
            SELECT t.*, ch.camera, ch.start_s as chunk_start
            FROM tracks t
            JOIN chunks ch ON ch.id = t.chunk_id
            WHERE t.id = ?
            """,
            (track_id,),
        ).fetchone()
        if not row:
            return None
        attrs = json.loads(row["attrs_json"] or "{}")
        pj = json.loads(row["path_json"] or "[]")
        aspect = None
        if pj:
            aspect = float(np.median([p[3] / max(p[4], 1.0) for p in pj]))
        return {
            "camera": row["camera"],
            "start_s": float(row["start_s"]),
            "end_s": float(row["end_s"]),
            "attrs": attrs,
            "cadence_bpm": attrs.get("cadence_bpm"),
            "aspect_ratio": aspect or attrs.get("body_aspect_ratio"),
            "motion": attrs.get("motion"),
            "crops_dir": row["crops_dir"],
        }
    except Exception:
        return None


def _z(d: Dict[int, float]) -> Dict[int, float]:
    if not d:
        return {}
    v = np.array(list(d.values()), dtype=np.float64)
    mu, sd = v.mean(), v.std() + 1e-9
    return {i: (s - mu) / sd for i, s in d.items()}


def rank(
    track_row: int,
    k: int = 20,
    mode: str = "any",
    w_app: Optional[float] = None,
    same_camera: bool = False,
    use_st: bool = True,
    use_kinematics: bool = True,
) -> Dict:
    """Find and rank cross-camera matches for a person track.

    Supports multi-cue fusion:
    1. Appearance features (person_ft / person).
    2. 2D Skeleton ST-GCN / DeepGaitV2 gait biometrics (person_gait).
    3. Anthropometric physical consistency (cadence BPM and body aspect ratio).
    4. Spatio-temporal camera network topology bounds (travel time feasibility).
    """
    qa = _vec("person_ft", track_row) or _vec("person", track_row)
    sql_q = _track_info_sql(track_row)

    if qa is None and sql_q is None:
        raise KeyError(f"track {track_row} not found in database or visual index")

    A = _all_scores("person_ft", qa) or _all_scores("person", qa)
    if not A or track_row not in A:
        if sql_q:
            # Fallback for synthetic/test environments where vectors are mocked or DB is populated
            A = A or {}
            A[track_row] = (1.0, {
                "camera": sql_q["camera"],
                "start_s": sql_q["start_s"],
                "end_s": sql_q["end_s"],
                "motion": sql_q.get("motion"),
                "cadence_bpm": sql_q.get("cadence_bpm"),
                "aspect_ratio": sql_q.get("aspect_ratio"),
            })
        else:
            raise KeyError(f"track {track_row} not found in visual index")

    q_pay = A[track_row][1]
    cam_q = sql_q["camera"] if sql_q else q_pay["camera"]
    q_start = sql_q["start_s"] if sql_q else float(q_pay.get("start_s", 0.0))
    q_end = sql_q["end_s"] if sql_q else float(q_pay.get("end_s", 0.0))
    q_cadence = (sql_q.get("cadence_bpm") if sql_q else None) or q_pay.get("cadence_bpm")
    q_aspect = (sql_q.get("aspect_ratio") if sql_q else None) or q_pay.get("aspect_ratio") or q_pay.get("body_aspect_ratio")
    q_motion = (sql_q.get("motion") if sql_q else None) or q_pay.get("motion")
    q_moving = q_motion in WALK

    # 2D Skeleton ST-GCN gait vector
    qg = _vec("person_gait", track_row)
    G = {}
    if qg is not None and q_moving:
        G = _all_scores("person_gait", qg)
        G = {i: v for i, v in G.items() if i == track_row or v[1].get("motion") in WALK}

    note = None
    if mode == "cloth_change":
        w = 0.20 if w_app is None else float(w_app)
        if qg is None:
            w = 0.55
            note = "Query track has no skeleton gait vector: using clothing-invariant soft biometrics & spatio-temporal gating"
        elif not q_moving:
            w = 0.55
            note = "Query person is stationary: using clothing-invariant soft biometrics & spatio-temporal gating"
    elif mode == "appearance_only":
        w = 1.0
    else:  # "any"
        w = MODES.get(mode, 0.75) if w_app is None else float(w_app)
        if qg is None:
            w = 1.0
            note = "Query track has no skeleton gait vector: visual appearance only"
        elif not q_moving:
            w = 1.0
            note = "Query person is stationary; gait biometrics only apply when walking or running"

    za = _z({i: s for i, (s, _) in A.items() if i != track_row})
    zg = {}
    if G and track_row in G:
        others_g = [s for i, (s, _) in G.items() if i != track_row]
        if others_g:
            zg = _z({i: s for i, (s, _) in G.items() if i != track_row})

    topology = load_topology()
    rows = []
    cand_ids = set(za) | set(zg)

    for i in cand_ids:
        pay = (A.get(i) or G.get(i))[1]
        c_cam = pay["camera"]
        if not same_camera and c_cam == cam_q:
            continue

        c_start = float(pay.get("start_s", 0.0))
        c_end = float(pay.get("end_s", c_start))
        c_cadence = pay.get("cadence_bpm")
        c_aspect = pay.get("aspect_ratio") or pay.get("body_aspect_ratio")

        # Optional SQLite metadata enrichment for aspect ratio
        if c_aspect is None:
            c_sql = _track_info_sql(int(i))
            if c_sql:
                c_aspect = c_sql.get("aspect_ratio")
                if c_cadence is None:
                    c_cadence = c_sql.get("cadence_bpm")

        # A. Spatio-temporal road/pedestrian topology gating
        st_score = 1.0
        transition_desc = f"{cam_q} → {c_cam}"
        if use_st:
            st_valid, st_score, transition_desc = validate_spatio_temporal(
                topology, cam_q, c_cam, q_start, q_end, c_start, c_end
            )
            if not st_valid:
                continue

        # B. Anthropometric and kinematic consistency
        kin_score = 1.0
        delta_cad = None
        if use_kinematics:
            kin_valid, kin_score, delta_cad = validate_kinematics(
                q_cadence, c_cadence, q_aspect, c_aspect,
                strict_cloth_change=(mode == "cloth_change"),
            )
            if not kin_valid:
                continue

        a = za.get(i)
        g = zg.get(i)

        # C. Multi-cue fusion scoring
        if mode == "cloth_change":
            if g is not None:
                # Fused CC-ReID score: gait biometrics + invariant appearance + kinematics + topology
                score = (
                    0.50 * g
                    + 0.20 * (a if a is not None else 0.0)
                    + 0.15 * ((kin_score - 0.5) * 4.0)
                    + 0.15 * ((st_score - 0.5) * 4.0)
                )
            else:
                score = (
                    0.45 * (a if a is not None else 0.0)
                    + 0.30 * ((kin_score - 0.5) * 4.0)
                    + 0.25 * ((st_score - 0.5) * 4.0)
                )
        elif mode == "any":
            base = w * (a if a is not None else 0.0) + (1.0 - w) * (g if g is not None else 0.0)
            score = base + 0.10 * ((kin_score - 0.5) * 4.0) + 0.10 * ((st_score - 0.5) * 4.0)
        else:  # appearance_only
            base = (a if a is not None else 0.0)
            score = base + 0.10 * ((st_score - 0.5) * 4.0)

        rows.append(dict(
            track_row=int(i),
            camera=c_cam,
            chunk_id=pay.get("chunk_id"),
            start_s=c_start,
            end_s=c_end,
            score=round(float(score), 3),
            appearance_z=None if a is None else round(float(a), 2),
            gait_z=None if g is None else round(float(g), 2),
            pose_z=None if g is None else round(float(g), 2),
            st_score=round(float(st_score), 2),
            kinematic_score=round(float(kin_score), 2),
            cadence_bpm=c_cadence,
            delta_cadence=delta_cad,
            aspect_ratio=round(float(c_aspect), 2) if c_aspect else None,
            transition=transition_desc,
            gait_symmetry=pay.get("gait_symmetry"),
            step_length_norm=pay.get("step_length_norm"),
        ))

    rows.sort(key=lambda r: -r["score"])
    return dict(
        query=dict(
            track_row=track_row,
            camera=cam_q,
            start_s=q_start,
            end_s=q_end,
            cadence_bpm=q_cadence,
            aspect_ratio=round(float(q_aspect), 2) if q_aspect else None,
            has_skeleton_gait=qg is not None,
        ),
        mode=mode,
        w_appearance=round(w, 2),
        use_st=use_st,
        use_kinematics=use_kinematics,
        note=note,
        results=rows[:k],
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("track_id", type=int, help="Track ID to search")
    parser.add_argument("--mode", default="cloth_change", choices=list(MODES))
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--no-st", dest="use_st", action="store_false", help="Disable spatio-temporal filter")
    parser.add_argument("--no-kin", dest="use_kin", action="store_false", help="Disable kinematic filter")
    args = parser.parse_args()

    r = rank(args.track_id, k=args.k, mode=args.mode, use_st=args.use_st, use_kinematics=args.use_kin)
    print(f"Re-ID Mode: {r['mode']} (Appearance weight: {r['w_appearance']})")
    if r["note"]:
        print(f"Note: {r['note']}")
    for hit in r["results"]:
        print(
            f"Track {hit['track_row']:5d} | Cam: {hit['camera']:10s} | Score: {hit['score']:+.3f} | "
            f"Gait Z: {hit['gait_z']} | ST: {hit['st_score']} | Trans: {hit['transition']}"
        )
