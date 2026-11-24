import pickle, sys
from src.index import vector_db as v
F = "data/indexes/qdrant_export.pkl"
if sys.argv[1] == "export":
    out = {}
    for name in v.DIMS:
        pts, off = [], None
        while True:
            r, off = v.client().scroll(name, limit=1000, offset=off, with_vectors=True, with_payload=True)
            pts += [(p.id, p.vector, p.payload) for p in r]
            if off is None:
                break
        out[name] = pts
    pickle.dump(out, open(F, "wb")); print({k: len(x) for k, x in out.items()})
else:
    for name, pts in pickle.load(open(F, "rb")).items():
        for i in range(0, len(pts), 500):
            b = pts[i:i + 500]
            v.upsert(name, [p[0] for p in b], [p[1] for p in b], [p[2] for p in b])
        print(name, len(pts))
