import sqlite3
from pathlib import Path
from src.common.config import PATHS

SCHEMA = """
CREATE TABLE IF NOT EXISTS chunks(
  id INTEGER PRIMARY KEY, camera TEXT, source TEXT, path TEXT UNIQUE,
  start_s REAL, end_s REAL, motion REAL, kept INTEGER);
CREATE TABLE IF NOT EXISTS tracks(
  id INTEGER PRIMARY KEY, chunk_id INTEGER, kind TEXT, track_id INTEGER,
  start_s REAL, end_s REAL, n_frames INTEGER, fps REAL, crops_dir TEXT,
  path_json TEXT, attrs_json TEXT, UNIQUE(chunk_id, kind, track_id));
CREATE TABLE IF NOT EXISTS processed(chunk_id INTEGER, kind TEXT, PRIMARY KEY(chunk_id, kind));
CREATE TABLE IF NOT EXISTS captions(
  id INTEGER PRIMARY KEY, chunk_id INTEGER, start_s REAL, end_s REAL, caption TEXT);
"""

def connect():
    Path(PATHS["sqlite"]).parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(PATHS["sqlite"], timeout=60)
    db.row_factory = sqlite3.Row
    db.execute("PRAGMA journal_mode=WAL")
    db.executescript(SCHEMA)
    return db
