"""Copy the cosine collections from Qdrant into a running ResonanceDB server (same ids, vectors and payloads).

    QDRANT_URL=http://localhost:6335 RESONANCE_URL=http://localhost:31415 python scripts/resonance_sync.py [collections...]

Only ids not already copied are inserted, so it can be rerun after new footage is indexed.
"""
import sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.index import vector_db, resonance


def sync(coll, batch=512):
    c = vector_db.client()
    have = set(resonance.idmap(coll)["rev"])
    off, n, t0 = None, 0, time.time()
    while True:
        pts, off = c.scroll(coll, limit=batch, offset=off, with_payload=True, with_vectors=True)
        new = [p for p in pts if str(p.id) not in have]
        if new:
            resonance.upsert(coll, [p.id for p in new], [p.vector for p in new], [p.payload or {} for p in new])
            n += len(new)
            resonance.save(coll)
        if off is None:
            break
    print(f"{coll}: +{n} (total {resonance.count(coll)}, {time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    if not resonance.health():
        sys.exit(f"ResonanceDB is not answering at {resonance.URL}")
    for coll in sys.argv[1:] or sorted(resonance.COLLECTIONS):
        sync(coll)
