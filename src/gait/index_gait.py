"""Build `person_gait` vectors (4096-d = 16 parts x 256, Euclidean) for every indexed person track:
re-read the chunk video, cut the track boxes, segment (YOLOv8-seg), run the fine-tuned gait model on up to K windows of
64 consecutive frames, average the window embeddings. Tracks without enough clean frames get no vector."""
import argparse, json, sqlite3, time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
from src.gait.silhouettes import norm64
from src.gait.track_crops import read_crops, windows


def segment(seg, crops, bs=64):
    out = []
    for b in range(0, len(crops), bs):
        for r in seg.predict(crops[b:b + bs], imgsz=320, classes=[0], retina_masks=True, half=True, verbose=False):
            if r.masks is None or len(r.masks.data) == 0:
                out.append(None); continue
            d = r.masks.data; k = int(d.flatten(1).sum(1).argmax())
            out.append((d[k] > 0.5).byte().mul(255).cpu().numpy())
    return out


def clean_sil(m, min_asp):
    if m is None:
        return None
    s = norm64(m)
    if s is None:
        return None
    ys, xs = np.nonzero(m.any(1))[0], np.nonzero(m.any(0))[0]
    return s if (ys[-1] - ys[0] + 1) / (xs[-1] - xs[0] + 1) >= min_asp else None


def main():
    import torch
    from ultralytics import YOLO
    from src.gait import og
    from src.index import vector_db
    ap = argparse.ArgumentParser()
    ap.add_argument("--fmt", default="cxcywh", choices=["cxcywh", "tlwh", "cxcyhw", "tlhw"])
    ap.add_argument("--K", type=int, default=3); ap.add_argument("--min-asp", type=float, default=1.0)
    ap.add_argument("--min-clean", type=int, default=20); ap.add_argument("--prefetch", type=int, default=3)
    ap.add_argument("--model", default="deepgaitv2_gait3d"); ap.add_argument("--ckpt", default="models/finetuned/gait/deepgaitv2_gait3d_ft.pt")
    ap.add_argument("--limit-chunks", type=int, default=0, help="sample N evenly spaced chunks (throughput test)")
    ap.add_argument("--no-upsert", action="store_true"); ap.add_argument("--recreate", action="store_true")
    ap.add_argument("--skip-existing", action="store_true", help="skip tracks already present in person_gait")
    ap.add_argument("--part", default="0/1", help="i/N: process every N-th chunk starting at i (split the run into resumable jobs)")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    db = sqlite3.connect("data/indexes/sqlite/meta.db"); db.row_factory = sqlite3.Row
    groups = {}
    for r in db.execute("select t.id, t.chunk_id, t.start_s, t.end_s, t.n_frames, t.path_json, t.attrs_json, ch.path vpath, ch.camera "
                        "from tracks t join chunks ch on ch.id=t.chunk_id where t.kind='person' order by t.chunk_id, t.id"):
        groups.setdefault(r["chunk_id"], []).append(dict(r))
    if a.skip_existing and not a.recreate:
        try:
            c = vector_db.client()
            have_ids = set()
            offset = None
            while True:
                pts, offset = c.scroll("person_gait", limit=2000, offset=offset, with_payload=False, with_vectors=False)
                have_ids.update(p.id for p in pts)
                if offset is None:
                    break
            for cid in list(groups.keys()):
                groups[cid] = [t for t in groups[cid] if t["id"] not in have_ids]
                if not groups[cid]:
                    del groups[cid]
            print(f"Resume mode: skipping {len(have_ids)} already indexed tracks. {sum(len(v) for v in groups.values())} tracks remain.", flush=True)
        except Exception as e:
            print("Could not query existing gait IDs:", e, flush=True)
    chunks = sorted(groups)
    pi, pn = (int(x) for x in a.part.split("/")); chunks = chunks[pi::pn]
    if a.limit_chunks:
        chunks = [chunks[i] for i in np.linspace(0, len(chunks) - 1, a.limit_chunks).astype(int)]
    ntr = sum(len(groups[c]) for c in chunks)
    print(f"{len(chunks)} chunks, {ntr} person tracks | device {dev} | K={a.K} fmt={a.fmt}", flush=True)
    gait = og.load(a.model, dev, ckpt=a.ckpt)[0]; seg = YOLO("models/pretrained/yolov8m-seg.pt")
    if a.recreate and not a.no_upsert:
        vector_db.recreate("person_gait")

    def job(cid):
        tr = groups[cid]
        return cid, read_crops(tr[0]["vpath"], [json.loads(t["path_json"]) for t in tr], a.K, a.fmt)

    pool = ThreadPoolExecutor(a.prefetch); it = iter(chunks); q = deque()
    for _ in range(a.prefetch):
        c = next(it, None)
        if c is not None: q.append(pool.submit(job, c))
    ids, vecs, pays, t0 = [], [], [], time.time()
    st = dict(tracks=0, embedded=0, crops=0, clean=0, empty_windows=0)
    done = 0
    while q:
        cid, crops = q.popleft().result()
        c = next(it, None)
        if c is not None: q.append(pool.submit(job, c))
        keys = sorted(crops); flat = [x for k in keys for x in crops[k]]
        sils = iter(clean_sil(m, a.min_asp) for m in segment(seg, flat)) if flat else iter(())
        per = {k: [s for s in (next(sils) for _ in crops[k]) if s is not None] for k in keys}
        st["crops"] += len(flat)
        for ti, t in enumerate(groups[cid]):
            st["tracks"] += 1; embs, n_clean = [], 0
            for wi in range(len(windows(int(t["n_frames"]), a.K))):
                ss = per.get((ti, wi), [])
                if len(ss) < a.min_clean:
                    st["empty_windows"] += 1; continue
                x = torch.from_numpy(np.stack(ss)).to(dev).float().div_(255)[:, :, 10:-10].unsqueeze(0)
                with torch.autocast("cuda", dtype=torch.float16, enabled=(dev == "cuda")):
                    embs.append(og.embed(gait, x)[0].float().cpu().numpy())
                n_clean += len(ss)
            if not embs:
                continue
            st["embedded"] += 1; st["clean"] += n_clean
            ids.append(t["id"]); vecs.append(np.mean(embs, 0).reshape(-1).tolist())
            pays.append({"track_row": t["id"], "chunk_id": t["chunk_id"], "camera": t["camera"], "start_s": t["start_s"],
                         "end_s": t["end_s"], "gait_n": n_clean, **json.loads(t["attrs_json"] or "{}")})
        if len(ids) >= 256 and not a.no_upsert:
            vector_db.upsert("person_gait", ids, vecs, pays); ids, vecs, pays = [], [], []
        done += 1
        if done % 10 == 0 or not q:
            el = time.time() - t0
            print(f"{done}/{len(chunks)} chunks | {st['tracks']} tracks, {st['embedded']} embedded | {st['crops'] / el:.0f} crops/s | "
                  f"{el:.0f}s elapsed, ETA {(len(chunks) - done) * el / done / 60:.0f} min", flush=True)
    if ids and not a.no_upsert:
        vector_db.upsert("person_gait", ids, vecs, pays)
    print(f"done. tracks {st['tracks']}, with gait vector {st['embedded']} ({100 * st['embedded'] / max(st['tracks'], 1):.1f}%), "
          f"mean clean frames/vector {st['clean'] / max(st['embedded'], 1):.0f}, windows rejected {st['empty_windows']}, {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
