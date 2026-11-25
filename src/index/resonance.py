"""ResonanceDB backend for the cosine collections (evaluation; VECTOR_BACKEND=resonance switches search to it).

ResonanceDB (github.com/LexProfi/ResonanceDB, Prosperity Public License 3.0: free for non-commercial use, commercial
use needs a licence after a 30-day trial) stores complex waveforms and ranks by "resonance" instead of distance.
Our embeddings are real-valued, so they go in with the sign-phase mapping its paper describes: amplitude |x|,
phase 0 for x >= 0 and pi for x < 0.

Its REST API has no payload filters and uses content-hash ids, so this module keeps, per collection, a map from
its ids to our ids plus our payloads (data/indexes/resonance/<collection>.json), over-fetches and applies
camera / id-restrict filters itself. The server takes one fixed pattern length (1536 by default), so shorter
embeddings are zero-padded. Gait (Euclidean) stays on Qdrant.
"""
import json, math, os, threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
import requests

URL = os.getenv("RESONANCE_URL", "http://localhost:31415")
ROOT = Path(os.getenv("RESONANCE_MAP_DIR", "data/indexes/resonance"))
DIM = int(os.getenv("RESONANCE_DIM", "1536"))       # the server accepts one fixed pattern length (resonance.pattern.len)
MAX_FETCH = int(os.getenv("RESONANCE_MAX_FETCH", "2000"))   # every match comes back with its full pattern: cap over-fetch
COLLECTIONS = {"person", "person_ft", "vehicle", "vehicle_ft", "incident", "text_person", "text_vehicle", "text_incident"}
_S = requests.Session()
_MAPS, _LOCK = {}, threading.Lock()


class Hit:
    __slots__ = ("id", "score", "payload")

    def __init__(s, i, score, payload):
        s.id, s.score, s.payload = i, score, payload


def wave(v):
    v = np.asarray(v, dtype=np.float64)
    n = np.linalg.norm(v)
    v = v / n if n else v
    if len(v) < DIM:                                  # zero-amplitude padding contributes nothing to the scores
        v = np.concatenate([v, np.zeros(DIM - len(v))])
    return {"amplitude": np.abs(v).tolist(), "phase": np.where(v < 0, math.pi, 0.0).tolist()}


def _path(coll):
    return ROOT / f"{coll}.json"


def idmap(coll):
    """{"ids": {res_id: [our ids]}, "rev": {our id: res_id}, "payload": {our id: payload}}"""
    if coll not in _MAPS:
        p = _path(coll)
        m = json.loads(p.read_text()) if p.exists() else {"ids": {}, "rev": {}, "payload": {}}
        _MAPS[coll] = m
    return _MAPS[coll]


def save(coll):
    ROOT.mkdir(parents=True, exist_ok=True)
    tmp = _path(coll).with_suffix(".tmp")
    tmp.write_text(json.dumps(idmap(coll)))
    tmp.replace(_path(coll))


def _post(path, body, timeout=60):
    r = _S.post(f"{URL}{path}", json=body, timeout=timeout)
    r.raise_for_status()
    return r.json()


def upsert(coll, ids, vectors, payloads, workers=8):
    m = idmap(coll)

    def one(args):
        i, v, p = args
        old = m["rev"].get(str(i))
        w = wave(v)
        try:
            if old:
                rid = _post(f"/corpora/{coll}/replace", {"id": old, "pattern": w, "metadata": {"id": str(i)}})["id"]
            else:
                rid = _post(f"/corpora/{coll}/insert", {"pattern": w, "metadata": {"id": str(i)}})["id"]
        except requests.HTTPError as e:
            if e.response is None or e.response.status_code != 409:
                raise
            # ids are content hashes: an identical pattern is already stored (a duplicate vector, or a rerun).
            # It is its own best match, so share that id.
            rid = _post(f"/corpora/{coll}/query", {"query": w, "topK": 1})[0]["id"]
        with _LOCK:
            if old and old != rid:
                lst = [x for x in m["ids"].get(old, []) if x != i]
                if lst:
                    m["ids"][old] = lst
                else:
                    m["ids"].pop(old, None)
            lst = m["ids"].setdefault(rid, [])      # identical vectors share one content-hash id
            if i not in lst:
                lst.append(i)
            m["rev"][str(i)] = rid
            m["payload"][str(i)] = p
    with ThreadPoolExecutor(workers) as ex:
        list(ex.map(one, zip(ids, vectors, payloads)))


def count(coll):
    return len(idmap(coll)["rev"])


def search(coll, vector, k=50, where=None, ids=None):
    m = idmap(coll)
    n = max(1, count(coll))
    allowed = set(ids) if ids is not None else None
    frac = 1.0
    if allowed is not None:
        frac = max(len(allowed) / n, 1e-6)
    if where:
        frac *= 0.1                                   # unknown selectivity: over-fetch generously
    top = min(n, int(math.ceil(k / frac * 2)) + k, max(MAX_FETCH, k))
    res = _post(f"/corpora/{coll}/query", {"query": wave(vector), "topK": top})
    out = []
    for r in res:
        for i in m["ids"].get(r["id"], []):
            if allowed is not None and i not in allowed:
                continue
            p = m["payload"].get(str(i), {})
            if where and any(p.get(a) != b for a, b in where.items()):
                continue
            out.append(Hit(i, float(r["energy"]), p))
        if len(out) >= k:
            break
    return out[:k]


def health():
    try:
        return _S.get(f"{URL}/health", timeout=3).ok
    except Exception:
        return False
