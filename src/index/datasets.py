"""Dataset separation. Every footage source added to the index carries a dataset tag on its chunks (meva, aicity,
...). Searches with a dataset selected, and re-identification of a track, must only ever return footage from that
one dataset, so every query path takes its candidate ids from here (SQLite is the source of truth: older Qdrant
points carry no dataset tag in their payload)."""
from functools import lru_cache
from src.index.sql_db import connect


def _ds(col="c.dataset"):
    return f"coalesce({col}, 'meva')"


@lru_cache(maxsize=None)
def ids(kind, dataset):
    """Frozen set of track ids (person / vehicle) or window ids (incident) that belong to `dataset`."""
    db = connect()
    if kind == "incident":
        q = f"SELECT w.id FROM windows w JOIN chunks c ON c.id=w.chunk_id WHERE {_ds()}=?"
        rows = db.execute(q, (dataset,))
    else:
        q = f"SELECT t.id FROM tracks t JOIN chunks c ON c.id=t.chunk_id WHERE t.kind=? AND {_ds()}=?"
        rows = db.execute(q, (kind, dataset))
    return frozenset(r[0] for r in rows)


def of_track(tid):
    """Dataset of one track (None if the track is unknown)."""
    r = connect().execute(f"SELECT {_ds()} FROM tracks t JOIN chunks c ON c.id=t.chunk_id WHERE t.id=?", (tid,)).fetchone()
    return r[0] if r else None


def clear():
    ids.cache_clear()
