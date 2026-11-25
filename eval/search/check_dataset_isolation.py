"""Dataset isolation check: re-ID and dataset-filtered search must never return footage from another dataset.
Run after adding a dataset:
    QDRANT_URL=http://localhost:6335 python -m eval.search.check_dataset_isolation
Exits non-zero if any result crosses datasets."""
import random, sys, time
from src.index.sql_db import connect
from src.index import datasets
from src.vehicle import reid as vreid
from src.gait import reid as preid
db = connect()
ALL = [r[0] for r in db.execute("select distinct coalesce(dataset, 'meva') from chunks order by 1")]
ds_of = lambda tid: datasets.of_track(tid)
random.seed(0)
bad = 0
for kind, fn in (("vehicle", lambda t: vreid.rank(t, k=12)), ("person", lambda t: preid.rank(t, k=20))):
    for ds in ALL:
        pool = sorted(datasets.ids(kind, ds))
        sample = random.sample(pool, min(20, len(pool)))
        n_res, lat, leaks = 0, [], 0
        for t in sample:
            t0 = time.time()
            try:
                r = fn(t)
            except KeyError:
                continue
            lat.append(time.time() - t0)
            for x in r["results"]:
                n_res += 1
                if ds_of(x["track_row"]) != ds:
                    leaks += 1
        bad += leaks
        if lat:
            print(f"{kind:7s} {ds:6s} queries {len(lat):2d} results {n_res:4d} cross-dataset {leaks}  median {sorted(lat)[len(lat)//2]:.2f}s")
from fastapi.testclient import TestClient
import app.main as M
c = TestClient(M.app)
for q, kind in (("red car", "vehicle"), ("white truck", "vehicle"), ("person walking", "person")):
    for ds in ALL + [ALL[-1].upper()]:
        r = c.get("/api/search", params=dict(kind=kind, q=q, k=24, dataset=ds)).json()
        wrong = [x["id"] for x in r["results"] if x["dataset"] != ds.lower()]
        bad += len(wrong)
        print(f"/api/search {q!r:16s} {ds:6s} n={r['n']:2d} cross-dataset {len(wrong)}")
    for ds in ALL:
        r = c.get("/api/ask", params=dict(q=q, n=24, dataset=ds)).json()
        wrong = [x["id"] for x in r["results"] if x["dataset"] != ds]; bad += len(wrong)
        print(f"/api/ask    {q!r:16s} {ds:6s} n={r['n']:2d} cross-dataset {len(wrong)}")
print("TOTAL cross-dataset results:", bad)
sys.exit(1 if bad else 0)
