"""Unit and integration tests for multi-camera vehicle tracking in teammate platform."""

import json
import os
import sqlite3
import tempfile
import pytest
from pathlib import Path
from fastapi.testclient import TestClient

import sys
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.vehicle import reid as vreid
from app.main import app


@pytest.fixture
def mock_db(monkeypatch):
    """Create a temporary SQLite database with mock chunks and vehicle tracks."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.executescript("""
        CREATE TABLE chunks(
          id INTEGER PRIMARY KEY, camera TEXT, source TEXT, path TEXT UNIQUE,
          start_s REAL, end_s REAL, motion REAL, kept INTEGER);
        CREATE TABLE tracks(
          id INTEGER PRIMARY KEY, chunk_id INTEGER, kind TEXT, track_id INTEGER,
          start_s REAL, end_s REAL, n_frames INTEGER, fps REAL, crops_dir TEXT,
          path_json TEXT, attrs_json TEXT, caption TEXT);
    """)

    # Chunks: c041, c042, c043 in a CityFlow layout (scene S06 + cam_timestamp), so they share a synchronised clock
    aic = Path(db_path).with_suffix("") / "AICity22_Track1_MTMC_Tracking"
    (aic / "cam_timestamp").mkdir(parents=True, exist_ok=True)
    (aic / "cam_timestamp" / "S06.txt").write_text("c041 0\nc042 0\nc043 0\n")
    src = {c: str(aic / "test" / "S06" / c / "vdo.avi") for c in ("c041", "c042", "c043")}
    conn.execute("INSERT INTO chunks VALUES(1, 'c041', ?, 'chunks/c041_1.mp4', 0.0, 30.0, 0.5, 1)", (src["c041"],))
    conn.execute("INSERT INTO chunks VALUES(2, 'c042', ?, 'chunks/c042_1.mp4', 0.0, 30.0, 0.5, 1)", (src["c042"],))
    conn.execute("INSERT INTO chunks VALUES(3, 'c043', ?, 'chunks/c043_1.mp4', 0.0, 30.0, 0.5, 1)", (src["c043"],))

    # Track 101 on c041: Blue car taking left (10.0s to 15.0s) -> Query Seed
    conn.execute(
        "INSERT INTO tracks VALUES(101, 1, 'vehicle', 1, 10.0, 15.0, 50, 10.0, '', '[]', ?, ?)",
        (json.dumps({"color": "blue", "type": "sedan", "turn": "left"}), "a blue sedan taking a left"),
    )

    # Track 102 on c042: Blue car driving forward (20.0s to 25.0s) -> Valid match (delta_t = 5s, within c041->c042 window 2-35s)
    conn.execute(
        "INSERT INTO tracks VALUES(102, 2, 'vehicle', 2, 20.0, 25.0, 50, 10.0, '', '[]', ?, ?)",
        (json.dumps({"color": "blue", "type": "sedan", "motion": "moving"}), "a blue sedan driving forward"),
    )

    # Track 103 on c042: Red truck (20.0s to 25.0s) -> Conflicting color (should be filtered by color gate)
    conn.execute(
        "INSERT INTO tracks VALUES(103, 2, 'vehicle', 3, 20.0, 25.0, 50, 10.0, '', '[]', ?, ?)",
        (json.dumps({"color": "red", "type": "truck", "motion": "moving"}), "a red pickup truck"),
    )

    # Track 104 on c042: Blue car (500.0s to 505.0s) -> far outside the +-60 s window on the synchronised clock
    conn.execute(
        "INSERT INTO tracks VALUES(104, 2, 'vehicle', 4, 500.0, 505.0, 50, 10.0, '', '[]', ?, ?)",
        (json.dumps({"color": "blue", "type": "sedan", "motion": "moving"}), "a blue sedan"),
    )

    conn.commit()
    conn.close()

    # Monkeypatch PATHS['sqlite'] to point to temporary database
    from src.common.config import PATHS
    orig_path = PATHS["sqlite"]
    PATHS["sqlite"] = db_path

    # Monkeypatch app's DEMO_DB/db connector
    def get_test_conn():
        c = sqlite3.connect(db_path, timeout=60)
        c.row_factory = sqlite3.Row
        return c

    monkeypatch.setattr("app.main._db", get_test_conn)

    yield db_path

    if os.path.exists(db_path):
        os.remove(db_path)
    PATHS["sqlite"] = orig_path


def test_color_extraction():
    """Verify vehicle color canonicalization."""
    assert vreid.extract_color({"color": "blue"}) == "blue"
    assert vreid.extract_color({"color": "RED"}) == "red"
    assert vreid.extract_color({}, "a silver SUV stopped") == "silver"
    assert vreid.extract_color({}, "black pickup truck") == "black"
    assert vreid.extract_color({}, "unknown object") is None


def test_topology_loading():
    """Verify camera topology connections for CityFlow c041-c046."""
    topo = vreid.load_topology()
    assert "cameras" in topo
    assert "c041" in topo["cameras"]
    connected, min_t, max_t = vreid.get_travel_window(topo, "c041", "c042")
    assert connected is True
    assert min_t == 2.0
    assert max_t == 35.0


def test_vehicle_reid_ranking(mock_db):
    """Vehicle re-ID with the time window on the synchronised clock and the (optional) colour gate."""
    res = vreid.rank(101, k=10, use_st_filter=True, use_color_gate=True)
    assert res["query"]["camera"] == "c041"
    assert res["query"]["color"] == "blue"

    matched_ids = [m["track_row"] for m in res["results"]]
    # Track 102 should be the only valid match
    assert 102 in matched_ids
    # Track 103 (red truck) must be rejected by color gate
    assert 103 not in matched_ids
    # Track 104 (485 s later) must be rejected by the time window
    assert 104 not in matched_ids
    assert next(m for m in res["results"] if m["track_row"] == 102)["time_gap_s"] == 5.0


def test_vehicle_reid_without_time_window_and_colour_gate_off_by_default(mock_db):
    res = vreid.rank(101, k=10, use_st_filter=False)
    ids = {m["track_row"] for m in res["results"]}
    assert {102, 103, 104} <= ids                  # no time window (104 is back), colour gate off (red 103 stays),
                                                   # and no per-camera collapsing (102 and 103 are both on c042)
    assert res["time_window_s"] is None


def test_fastapi_reid_vehicle_endpoint(mock_db):
    """Test FastAPI /api/reid/vehicle/{tid} endpoint."""
    client = TestClient(app)
    resp = client.get("/api/reid/vehicle/101?use_st=true&use_color=true")
    assert resp.status_code == 200
    data = resp.json()
    assert data["kind"] == "vehicle"
    assert data["query"]["camera"] == "c041"
    assert len(data["results"]) == 1
    assert data["results"][0]["id"] == 102
    assert data["results"][0]["camera"] == "c042"
    assert data["time_window_s"] == 60.0
