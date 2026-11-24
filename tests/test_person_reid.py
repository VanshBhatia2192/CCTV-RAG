"""Unit and integration tests for multi-cue Person and Cloth-Changing Re-ID (CC-ReID)."""

import json
import os
import sqlite3
import tempfile
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

from src.gait import reid
from app.main import app


@pytest.fixture
def mock_pedestrian_env(monkeypatch):
    """Sets up a temporary SQLite database and mock Qdrant vectors for person Re-ID testing."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.executescript("""
        CREATE TABLE chunks(
          id INTEGER PRIMARY KEY, camera TEXT, source TEXT, path TEXT UNIQUE,
          start_s REAL, end_s REAL, motion REAL, kept INTEGER, dataset TEXT DEFAULT 'meva');
        CREATE TABLE tracks(
          id INTEGER PRIMARY KEY, chunk_id INTEGER, kind TEXT, track_id INTEGER,
          start_s REAL, end_s REAL, n_frames INTEGER, fps REAL, crops_dir TEXT,
          path_json TEXT, attrs_json TEXT, caption TEXT, UNIQUE(chunk_id, kind, track_id));
    """)

    # Chunks across CityFlow/Synthetic network: c041, c042, c043, c046
    conn.execute("INSERT INTO chunks VALUES(1, 'c041', 'c041/vdo.avi', 'chunks/c041_1.mp4', 0.0, 60.0, 0.5, 1, 'meva')")
    conn.execute("INSERT INTO chunks VALUES(2, 'c042', 'c042/vdo.avi', 'chunks/c042_1.mp4', 0.0, 60.0, 0.5, 1, 'meva')")
    conn.execute("INSERT INTO chunks VALUES(3, 'c043', 'c043/vdo.avi', 'chunks/c043_1.mp4', 0.0, 60.0, 0.5, 1, 'meva')")
    conn.execute("INSERT INTO chunks VALUES(4, 'c046', 'c046/vdo.avi', 'chunks/c046_1.mp4', 0.0, 60.0, 0.5, 1, 'meva')")

    # Path json representing normal aspect ratio ~2.5 (h=100, w=40)
    normal_boxes = json.dumps([[0, 50, 50, 100, 40], [10, 55, 55, 100, 40]])
    squat_boxes = json.dumps([[0, 50, 50, 60, 50], [10, 55, 55, 60, 50]])  # aspect ratio 1.2

    # Track 201 on c041: Query Subject (red jacket, 10s-18s, cadence 112 BPM, walking)
    conn.execute(
        "INSERT INTO tracks VALUES(201, 1, 'person', 1, 10.0, 18.0, 80, 10.0, '', ?, ?, 'person in red jacket walking')",
        (normal_boxes, json.dumps({"motion": "walking", "cadence_bpm": 112, "gait_symmetry": 0.90})),
    )

    # Track 202 on c042: Ground-truth match who changed into black coat (30s-38s, dt = 12s, c041->c042 window 2-35s)
    # Cadence 114 BPM (close to 112), aspect ratio 2.5
    conn.execute(
        "INSERT INTO tracks VALUES(202, 2, 'person', 2, 30.0, 38.0, 80, 10.0, '', ?, ?, 'person in dark coat walking')",
        (normal_boxes, json.dumps({"motion": "walking", "cadence_bpm": 114, "gait_symmetry": 0.89})),
    )

    # Track 203 on c042: Distractor stranger with completely different cadence (30s-38s, cadence 165 BPM, squat aspect)
    conn.execute(
        "INSERT INTO tracks VALUES(203, 2, 'person', 3, 30.0, 38.0, 80, 10.0, '', ?, ?, 'person jogging')",
        (squat_boxes, json.dumps({"motion": "walking", "cadence_bpm": 165, "gait_symmetry": 0.60})),
    )

    # Track 204 on c042: Candidate observed 800 seconds later (dt = 782s, exceeds max travel window)
    conn.execute(
        "INSERT INTO tracks VALUES(204, 2, 'person', 4, 800.0, 808.0, 80, 10.0, '', ?, ?, 'person walking late')",
        (normal_boxes, json.dumps({"motion": "walking", "cadence_bpm": 112, "gait_symmetry": 0.90})),
    )

    # Track 205 on c046: Candidate on disconnected camera from c041 (no path c041 -> c046 in topology)
    conn.execute(
        "INSERT INTO tracks VALUES(205, 4, 'person', 5, 25.0, 33.0, 80, 10.0, '', ?, ?, 'person on avenue')",
        (normal_boxes, json.dumps({"motion": "walking", "cadence_bpm": 113, "gait_symmetry": 0.91})),
    )

    conn.commit()
    conn.close()

    # Monkeypatch PATHS['sqlite'] to temporary database
    from src.common.config import PATHS
    orig_path = PATHS["sqlite"]
    PATHS["sqlite"] = db_path

    # Monkeypatch app database connection
    def get_test_conn():
        c = sqlite3.connect(db_path, timeout=60)
        c.row_factory = sqlite3.Row
        return c

    monkeypatch.setattr("app.main._db", get_test_conn)

    # Mock Qdrant _vec and _all_scores in src.gait.reid
    mock_vectors = {
        "person_ft": {
            201: [0.1] * 512,
            202: [0.08] * 512,  # Moderate visual similarity (cloth changed)
            203: [0.01] * 512,  # Low visual similarity
            204: [0.09] * 512,
            205: [0.08] * 512,
        },
        "person_gait": {
            201: [0.2] * 256,
            202: [0.19] * 256,  # High gait similarity
            203: [-0.1] * 256,  # Low gait similarity
            204: [0.19] * 256,  # High gait similarity (distractor)
            205: [0.19] * 256,  # High gait similarity (distractor on wrong camera)
        },
    }

    def mock_vec(coll, tid):
        return mock_vectors.get(coll, {}).get(tid)

    def mock_all_scores(coll, vec):
        if coll not in mock_vectors:
            return {}
        res = {}
        vec_arr = [float(x) for x in vec]
        db = get_test_conn()
        try:
            for tid, v in mock_vectors[coll].items():
                sim = float(sum(a * b for a, b in zip(vec_arr, v)))
                r = db.execute("SELECT t.*, ch.camera FROM tracks t JOIN chunks ch ON ch.id=t.chunk_id WHERE t.id=?", (tid,)).fetchone()
                attrs = json.loads(r["attrs_json"] or "{}")
                pj = json.loads(r["path_json"] or "[]")
                aspect = float(pj[0][3] / max(pj[0][4], 1.0)) if pj else 2.5
                payload = {
                    "track_row": tid,
                    "chunk_id": r["chunk_id"],
                    "camera": r["camera"],
                    "start_s": float(r["start_s"]),
                    "end_s": float(r["end_s"]),
                    "motion": attrs.get("motion"),
                    "cadence_bpm": attrs.get("cadence_bpm"),
                    "gait_symmetry": attrs.get("gait_symmetry"),
                    "aspect_ratio": aspect,
                }
                res[tid] = (sim, payload)
        finally:
            db.close()
        return res

    monkeypatch.setattr(reid, "_vec", mock_vec)
    monkeypatch.setattr(reid, "_all_scores", mock_all_scores)

    yield db_path

    PATHS["sqlite"] = orig_path
    try:
        if os.path.exists(db_path):
            os.remove(db_path)
    except OSError:
        pass


def test_pedestrian_travel_window():
    """Verify camera topology travel bounds for pedestrians."""
    topo = reid.load_topology()
    assert "cameras" in topo

    # c041 -> c042: connected with distance 60m
    connected, min_t, max_t = reid.get_pedestrian_travel_window(topo, "c041", "c042")
    assert connected is True
    assert min_t > 0
    assert max_t > min_t

    # c041 -> c046: directly disconnected in topology
    connected, _, _ = reid.get_pedestrian_travel_window(topo, "c041", "c046")
    assert connected is False


def test_spatio_temporal_validation():
    """Verify forward, backward, expired, and disconnected transitions."""
    topo = reid.load_topology()

    # Valid forward transition: c041 (10s-18s) to c042 (30s-38s), dt = 12s
    valid, score, desc = reid.validate_spatio_temporal(topo, "c041", "c042", 10.0, 18.0, 30.0, 38.0)
    assert valid is True
    assert score >= 0.5
    assert "feasible" in desc

    # Expired transition: c041 (10s-18s) to c042 (800s-808s), dt = 782s
    valid, _, desc = reid.validate_spatio_temporal(topo, "c041", "c042", 10.0, 18.0, 800.0, 808.0)
    assert valid is False
    assert "expired" in desc

    # Disconnected transition: c041 to c046
    valid, _, desc = reid.validate_spatio_temporal(topo, "c041", "c046", 10.0, 18.0, 25.0, 33.0)
    assert valid is False
    assert "no topological connection" in desc


def test_kinematic_validation():
    """Verify walking cadence and body aspect ratio consistency."""
    # Consistent cadence (112 vs 114 BPM) and aspect ratio (2.5 vs 2.5)
    valid, score, delta_cad = reid.validate_kinematics(112, 114, 2.5, 2.5, strict_cloth_change=True)
    assert valid is True
    assert score > 0.9
    assert delta_cad == 2.0

    # Incompatible cadence (112 vs 165 BPM) in strict cloth-change mode -> rejected
    valid, _, delta_cad = reid.validate_kinematics(112, 165, 2.5, 2.5, strict_cloth_change=True)
    assert valid is False
    assert delta_cad == 53.0

    # Incompatible aspect ratio (2.5 vs 1.1) in strict cloth-change mode -> rejected
    valid, _, _ = reid.validate_kinematics(112, 114, 2.5, 1.1, strict_cloth_change=True)
    assert valid is False


def test_cloth_change_reid_ranking(mock_pedestrian_env):
    """Test full CC-ReID ranking: track 202 must match, while 203, 204, 205 must be rejected."""
    res = reid.rank(201, k=10, mode="cloth_change", use_st=True, use_kinematics=True)
    assert res["query"]["camera"] == "c041"
    assert res["query"]["cadence_bpm"] == 112

    matched_ids = [m["track_row"] for m in res["results"]]

    # Track 202 is the true match (valid travel window, matching cadence and aspect ratio)
    assert 202 in matched_ids
    # Track 203 rejected by kinematics (cadence 165 vs 112)
    assert 203 not in matched_ids
    # Track 204 rejected by spatio-temporal filter (dt = 782s)
    assert 204 not in matched_ids
    # Track 205 rejected by camera network topology (c041 -> c046 disconnected)
    assert 205 not in matched_ids


def test_api_reid_endpoint(mock_pedestrian_env):
    """Test FastAPI /api/reid/{tid} endpoint with spatio-temporal and kinematic parameters."""
    client = TestClient(app)
    resp = client.get("/api/reid/201?mode=cloth_change&use_st=true&use_kinematics=true")
    assert resp.status_code == 200
    data = resp.json()
    assert data["mode"] == "cloth_change"
    assert data["use_st"] is True
    assert data["use_kinematics"] is True

    result_ids = [r["id"] for r in data["results"]]
    assert 202 in result_ids
    assert 203 not in result_ids
    assert 204 not in result_ids
    assert 205 not in result_ids

    # Check that transition and ST score are attached to the card
    match = data["results"][0]
    assert match["id"] == 202
    assert "transition" in match
    assert match["st_score"] is not None
