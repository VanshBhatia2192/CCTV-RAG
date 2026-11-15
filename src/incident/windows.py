from src.common.config import PIPE
from src.index.sql_db import connect

SCHEMA = """CREATE TABLE IF NOT EXISTS windows(
  id INTEGER PRIMARY KEY, chunk_id INTEGER, rel_start REAL, rel_end REAL,
  start_s REAL, end_s REAL, embedded INTEGER DEFAULT 0, caption TEXT,
  UNIQUE(chunk_id, rel_start))"""

def main():
    C = PIPE["incident"]; db = connect(); db.execute(SCHEMA)
    for c in db.execute("SELECT id, start_s, end_s FROM chunks WHERE kept=1").fetchall():
        dur, s = c["end_s"] - c["start_s"], 0.0
        while s < dur - 1e-6:
            e = min(s + C["window_sec"], dur)
            if e - s >= 4:
                db.execute("INSERT OR IGNORE INTO windows(chunk_id,rel_start,rel_end,start_s,end_s) VALUES(?,?,?,?,?)",
                           (c["id"], s, e, c["start_s"] + s, c["start_s"] + e))
            s += C["step_sec"]
    db.commit()
    print("windows:", db.execute("SELECT COUNT(*) FROM windows").fetchone()[0])

if __name__ == "__main__":
    main()
