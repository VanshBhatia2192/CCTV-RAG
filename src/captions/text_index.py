import re, sys
import numpy as np, torch
from src.common.clip import DEV
from src.index import vector_db
from src.index.sql_db import connect

NAME = "BAAI/bge-small-en-v1.5"
QP = "Represent this sentence for searching relevant passages: "
_m = _t = None
_bm = {}

def _load():
    global _m, _t
    if _m is None:
        from transformers import AutoModel, AutoTokenizer
        try:
            _t = AutoTokenizer.from_pretrained(NAME, local_files_only=True)
            _m = AutoModel.from_pretrained(NAME, local_files_only=True).to(DEV).eval()
        except Exception:
            _t = AutoTokenizer.from_pretrained(NAME)
            _m = AutoModel.from_pretrained(NAME).to(DEV).eval()
    return _m, _t

@torch.no_grad()
def embed(texts, query=False, bs=64):
    m, t = _load(); out = []
    for i in range(0, len(texts), bs):
        b = [(QP + x if query else x) for x in texts[i:i + bs]]
        enc = t(b, padding=True, truncation=True, max_length=128, return_tensors="pt").to(DEV)
        v = m(**enc).last_hidden_state[:, 0]
        out.append(torch.nn.functional.normalize(v, dim=-1).cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, 384), dtype="float32")

def docs(kind):
    db = connect()
    if kind == "incident":
        rows = db.execute("SELECT id, caption FROM windows WHERE caption IS NOT NULL").fetchall()
    else:
        rows = db.execute("SELECT id, caption FROM tracks WHERE kind=? AND caption IS NOT NULL", (kind,)).fetchall()
    return [(r[0], r[1]) for r in rows]

def tok(s):
    return re.findall(r"[a-z0-9]+", s.lower())

def bm25_search(kind, q, k=100):
    if kind not in _bm:
        from rank_bm25 import BM25Okapi
        d = docs(kind)
        _bm[kind] = (BM25Okapi([tok(c) for _, c in d]) if d else None, [i for i, _ in d])
    bm, ids = _bm[kind]
    if bm is None:
        return []
    sc = bm.get_scores(tok(q))
    return [(ids[i], float(sc[i])) for i in np.argsort(-sc)[:k] if sc[i] > 0]

def build():
    for kind in ("person", "vehicle", "incident"):
        d = docs(kind)
        if not d:
            print(kind, "no captions"); continue
        vecs = embed([c for _, c in d])
        for j in range(0, len(d), 500):
            b = d[j:j + 500]
            vector_db.upsert(f"text_{kind}", [i for i, _ in b], vecs[j:j + 500].tolist(), [{"caption": c} for _, c in b])
        print(kind, len(d), "captions indexed")

if __name__ == "__main__" and sys.argv[1:] == ["build"]:
    build()
