"""Score the re-ID *implementations* (src/gait/reid.py rank(), unmodified) on MEVID ground truth.

MEVID test tracklets are loaded into a temporary Qdrant server + SQLite in the app's own format (person_ft,
person, person_gait collections; tracks/chunks tables with boxes, motion, cadence), then each checkout's rank()
is called exactly as the API calls it. Ranked lists are scored with eval_mevid's protocol (gallery = non-query
tracklets, same-id same-camera junk; cloth-changing split also drops same-outfit). MEVID has no timestamps, so
every tracklet gets start 0 and the topology gate sees overlapping observations (it can only pass or reject by
camera connectivity; the topology file has no MEVA cameras, so it passes).

Gait variants: deepgait = main's DeepGaitV2-ft (4096-d, Euclidean, what the live index holds);
               skeleton = the branch's ST-GCN pose gait (256-d, cosine, what the branch was designed for).

    python eval/gait/eval_reid_impl_mevid.py --branch ~/cctv-search-gait --main ~/cctv-search-ctx
"""
import argparse, json, os, shutil, sqlite3, subprocess, sys, tempfile, time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(HERE))
from eval.gait.eval_reid_cues import cues          # cadence + aspect from MEVID keypoints/crops
from eval.gait.eval_reid_live import load_module


def start_qdrant(port):
    d = Path(tempfile.mkdtemp(prefix="qd_eval_"))
    env = dict(os.environ, QDRANT__SERVICE__HTTP_PORT=str(port), QDRANT__SERVICE__GRPC_PORT=str(port + 1),
               QDRANT__TELEMETRY_DISABLED="true", QDRANT__STORAGE__STORAGE_PATH=str(d / "storage"))
    p = subprocess.Popen([str(HERE / "bin" / "qdrant")], cwd=d, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    import urllib.request
    for _ in range(60):
        try:
            urllib.request.urlopen(f"http://localhost:{port}/collections", timeout=1); return p, d
        except Exception:
            time.sleep(0.5)
    raise RuntimeError("qdrant did not start")


def build(url, dbp, z, gait_vecs, gait_metric, cad, asp):
    from qdrant_client import QdrantClient, models
    c = QdrantClient(url=url, timeout=60)
    n = len(z["tid"])
    payload = [dict(camera=str(int(z["cam"][i])), chunk_id=int(i), start_s=0.0, end_s=float(z["n"][i]) / 30.0,
                    motion="walking", cadence_bpm=None if np.isnan(cad[i]) else float(cad[i]),
                    aspect_ratio=None if np.isnan(asp[i]) else float(asp[i])) for i in range(n)]
    app = z["app_ft"].astype(np.float32)
    for name, vecs, dist in (("person_ft", app, models.Distance.COSINE), ("person", app, models.Distance.COSINE),
                             ("person_gait", gait_vecs, gait_metric)):
        if c.collection_exists(name):
            c.delete_collection(name)
        c.create_collection(name, vectors_config=models.VectorParams(size=vecs.shape[1], distance=dist))
        for s in range(0, n, 256):
            c.upsert(name, points=[models.PointStruct(id=i, vector=vecs[i].tolist(), payload=payload[i])
                                   for i in range(s, min(n, s + 256))])
    if Path(dbp).exists():
        Path(dbp).unlink()
    db = sqlite3.connect(dbp)
    db.executescript("""CREATE TABLE chunks(id INTEGER PRIMARY KEY, camera TEXT, source TEXT, path TEXT UNIQUE, start_s REAL,
                        end_s REAL, motion REAL, kept INTEGER, dataset TEXT DEFAULT 'meva');
                        CREATE TABLE tracks(id INTEGER PRIMARY KEY, chunk_id INTEGER, kind TEXT, track_id INTEGER, start_s REAL,
                        end_s REAL, n_frames INTEGER, fps REAL, crops_dir TEXT, path_json TEXT, attrs_json TEXT, caption TEXT);
                        CREATE TABLE processed(chunk_id INTEGER, kind TEXT, PRIMARY KEY(chunk_id, kind));
                        CREATE TABLE captions(id INTEGER PRIMARY KEY, chunk_id INTEGER, start_s REAL, end_s REAL, caption TEXT);""")
    for i in range(n):
        h = 100.0; w = h / asp[i] if not np.isnan(asp[i]) else 40.0
        db.execute("INSERT INTO chunks VALUES(?,?,?,?,?,?,?,?,?)", (i, payload[i]["camera"], f"mevid/{i}", f"mevid/{i}.mp4", 0.0,
                                                                  payload[i]["end_s"], 1.0, 1, "mevid"))
        db.execute("INSERT INTO tracks VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
                   (i, i, "person", i, 0.0, payload[i]["end_s"], int(z["n"][i]), 30.0, "",
                    json.dumps([[0, 50, 50, h, w], [1, 50, 50, h, w]]),
                    json.dumps({"motion": "walking", **({"cadence_bpm": payload[i]["cadence_bpm"]} if payload[i]["cadence_bpm"] else {})}), None))
    db.commit(); db.close()


def score(rank_fn, z, modes_of_impl, k):
    pid, cam, outfit, isq = z["pid"], z["cam"], z["outfit"], z["is_query"]
    qi = np.where(isq)[0]; gal = set(np.where(~isq)[0].tolist())
    out = {}
    for mode in modes_of_impl:
        res = {"all": [], "cc": []}; lat = []
        for q in qi:
            t = time.time()
            try:
                r = rank_fn(int(q), k=k, mode=mode)
            except Exception:
                for s in res: res[s].append(None)
                continue
            lat.append(time.time() - t)
            ranked = [x["track_row"] for x in r["results"] if x["track_row"] in gal]
            for split in ("all", "cc"):
                same = [g for g in ranked if pid[g] == pid[q]]
                junk = {g for g in ranked if pid[g] == pid[q] and (cam[g] == cam[q] or (split == "cc" and outfit[g] == outfit[q]))}
                pos_all = [g for g in gal if pid[g] == pid[q] and cam[g] != cam[q] and not (split == "cc" and outfit[g] == outfit[q])]
                if not pos_all:
                    res[split].append(None); continue
                lst = [g for g in ranked if g not in junk]
                hits = [j for j, g in enumerate(lst) if pid[g] == pid[q]]
                ap = sum((m + 1) / (h + 1) for m, h in enumerate(hits)) / len(pos_all)   # unreturned positives count as misses
                res[split].append((bool(hits and hits[0] < 1), bool(hits and hits[0] < 5), ap))
        for split, v in res.items():
            v = [x for x in v if x is not None]
            out[f"{mode}/{split}"] = dict(n=len(v), R1=round(100 * np.mean([x[0] for x in v]), 1),
                                          R5=round(100 * np.mean([x[1] for x in v]), 1), mAP=round(100 * np.mean([x[2] for x in v]), 1))
        out[f"{mode}/median_latency_s"] = round(float(np.median(lat)), 3) if lat else None
    return out


def main(a):
    z = np.load(a.cache)
    zp = np.load(a.pose_cache)
    cad, asp = map(lambda x: np.array([np.nan if v is None else v for v in x], float), zip(*[cues(t) for t in z["tid"]]))
    from qdrant_client import models
    gaits = {"deepgait": (z["gait_deepgaitv2_gait3d_ft"].reshape(len(z["tid"]), -1).astype(np.float32), models.Distance.EUCLID),
             "skeleton": (zp["gait_stgcn_pose_ft"].reshape(len(z["tid"]), -1).astype(np.float32), models.Distance.COSINE)}
    proc, qdir = start_qdrant(a.port)
    url = f"http://localhost:{a.port}"
    os.environ["QDRANT_URL"] = url
    dbp = str(Path(qdir) / "mevid.db")
    report = {}
    try:
        for gname, (vecs, metric) in gaits.items():
            build(url, dbp, z, vecs, metric, cad, asp)
            for name, root in (("main", a.main), ("branch", a.branch)):
                mod = load_module(root)
                sys.modules["src.common.config"].PATHS["sqlite"] = dbp
                vdb = sys.modules["src.index.vector_db"]; vdb._client = None
                modes = [m for m in ("any", "cloth_change", "appearance_only") if m in mod.MODES]
                r = score(mod.rank, z, modes, a.k)
                report[f"{gname}/{name}"] = r
                for key, v in r.items():
                    if isinstance(v, dict):
                        print(f"gait={gname:8s} {name:6s} {key:24s} R1 {v['R1']:5.1f}  R5 {v['R5']:5.1f}  mAP {v['mAP']:5.1f}  (n={v['n']})", flush=True)
                print(f"gait={gname:8s} {name:6s} latency: " + ", ".join(f"{k.split('/')[0]} {v}s" for k, v in r.items() if k.endswith("latency_s")), flush=True)
    finally:
        proc.terminate(); proc.wait(); shutil.rmtree(qdir, ignore_errors=True)
    Path(a.report).write_text(json.dumps(report, indent=1)); print("wrote", a.report)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--branch", required=True); ap.add_argument("--main", required=True)
    ap.add_argument("--cache", default="outputs/gait_prep/emb_test_ft.npz")
    ap.add_argument("--pose-cache", default="outputs/gait_prep/emb_test_pose.npz")
    ap.add_argument("--k", type=int, default=2000); ap.add_argument("--port", type=int, default=6355)
    ap.add_argument("--report", default="outputs/reports/eval_reid_impl_mevid.json")
    main(ap.parse_args())
