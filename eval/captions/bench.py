"""Caption benchmark: do human descriptions find the right item through the generated captions?

Both caption versions run on the same inputs with the same VLM (Qwen2.5-VL-7B); retrieval uses the system's text
search (BGE dense + BM25, fused with RRF), so a higher score means a caption makes the item easier to find.

  rstp      RSTPReid test images, downscaled to CCTV size (default 100 px high); queries = the 2 human captions per
            image; a hit is any image of the same person.  v1 = old prompt on the native-size crop.
  cityflow  CityFlow-NL vehicle tracks (crops from the AICity videos); queries = the 3 human descriptions per track.
  meva      MEVA activity clips (121 labelled videos, 36 classes); scene windows of those clips; one natural-language
            query per class (eval/e2e/eval_activities.NAT); v1 = the scene captions already in the index.

    python -m eval.captions.bench --sets rstp cityflow meva --n 600
"""
import argparse, json, random, time
from pathlib import Path
import numpy as np
from PIL import Image
from src.captions import v2, caption_tracks as v1t
from src.captions import text_index as TI

RAW = Path("data/raw")


# ---------------------------------------------------------------- retrieval
def rank_matrix(queries, captions):
    """RRF of BGE cosine and BM25 ranks -> score matrix [Q, G]."""
    from rank_bm25 import BM25Okapi
    caps = [c or "" for c in captions]
    G = TI.embed(caps); Q = TI.embed(queries, query=True)
    dense = Q @ G.T
    bm = BM25Okapi([TI.tok(c) for c in caps])
    lex = np.stack([bm.get_scores(TI.tok(q)) for q in queries])
    def ranks(s):
        r = np.empty_like(s, dtype=np.int64)
        r[np.arange(len(s))[:, None], np.argsort(-s, axis=1)] = np.arange(s.shape[1])[None, :]
        return r
    return 1.0 / (60 + ranks(dense) + 1) + 1.0 / (60 + ranks(lex) + 1)


def metrics(S, q_label, g_label):
    g_label = np.asarray(g_label); out = {"R@1": [], "R@5": [], "R@10": [], "mAP": []}
    for i, row in enumerate(S):
        order = np.argsort(-row); rel = g_label[order] == q_label[i]
        hits = np.where(rel)[0]
        if not len(hits):
            continue
        out["R@1"].append(hits[0] < 1); out["R@5"].append(hits[0] < 5); out["R@10"].append(hits[0] < 10)
        out["mAP"].append(np.mean([(k + 1) / (h + 1) for k, h in enumerate(hits)]))
    return {k: round(100 * float(np.mean(v)), 1) for k, v in out.items()} | {"queries": len(out["R@1"])}


def first_hit_metrics(S, q_label, g_label, g_group):
    """MEVA: rank clips (dedupe windows by clip), hit = first clip with the query's class."""
    out = []
    for i, row in enumerate(S):
        seen, r = [], 0
        for j in np.argsort(-row):
            if g_group[j] in seen:
                continue
            seen.append(g_group[j])
            if g_label[j] == q_label[i]:
                r = len(seen); break
        out.append(r)
    r = np.array(out)
    return {"R@1": round(100 * float(np.mean((r > 0) & (r <= 1))), 1), "R@5": round(100 * float(np.mean((r > 0) & (r <= 5))), 1),
            "R@10": round(100 * float(np.mean((r > 0) & (r <= 10))), 1),
            "MRR": round(float(np.mean([1 / x if x else 0 for x in r])), 3), "queries": len(r)}


# ---------------------------------------------------------------- captioning both ways
def v1_caption(kind, ims_list, bs=12):
    """Old pipeline: native-size crops (thumbnail to 336, never enlarged) + old free-text prompt."""
    out = []
    for i in range(0, len(ims_list), bs):
        items = []
        for ims in ims_list[i:i + bs]:
            small = []
            for im in ims:
                im = im.copy(); im.thumbnail((336, 336)); small.append(im)
            items.append((v1t.PROMPTS[kind], small))
        out += v2.generate(items, max_new_tokens=60)
    return out


def v2_caption(kind, ims_list, extra=None, bs=12):
    out, raws = [], []
    for i in range(0, len(ims_list), bs):
        res = v2.describe(kind, ims_list[i:i + bs], extra[i:i + bs] if extra else None)
        out += [c or "" for _, c, _ in res]; raws += [r for _, _, r in res]
    return out, raws


def enlarge(ims, kind):
    return [v2.crop_box(np.array(im), im.width / 2, im.height / 2, im.height / 1.24 if kind == "person" else im.height / 1.16,
                        im.width / (1.24 if kind == "person" else 1.16), kind) for im in ims]


# ---------------------------------------------------------------- sets
def set_rstp(n, h):
    rows = [d for d in json.load(open(RAW / "rstpreid/data_captions.json")) if d["split"] == "test"]
    random.Random(0).shuffle(rows); rows = rows[:n]
    nat, imgs, ids, qs, ql = [], [], [], [], []
    for d in rows:
        im = Image.open(RAW / "rstpreid/imgs" / d["img_path"]).convert("RGB")
        if h:
            im = im.resize((max(1, round(im.width * h / im.height)), h), Image.BICUBIC)
        nat.append([im]); ids.append(d["id"])
        for c in d["captions"]:
            qs.append(c); ql.append(d["id"])
    return "person", nat, ids, qs, np.array(ql), None


def set_cityflow(n):
    tr = json.load(open(RAW / "cityflow_nl/train-tracks.json"))
    keys = sorted(tr); random.Random(0).shuffle(keys); keys = keys[:n]
    nat, ids, qs, ql = [], [], [], []
    root = RAW / "cityflow_nl/AICity22_Track1_MTMC_Tracking"
    for k in keys:
        t = tr[k]
        vid = root / Path(t["frames"][0]).parent.parent / "vdo.avi"
        if not vid.exists():
            continue
        path = [[int(Path(f).stem) - 1, x + w / 2, y + h / 2, h, w] for f, (x, y, w, h) in zip(t["frames"], t["boxes"])]
        sel = v2.pick_frames(path, 3)
        frs = v2.read_frames(vid, [p[0] for p in sel])
        ims = []
        for p in sel:
            if p[0] in frs:   # native-size crop (v1 input); v2 enlarges it below
                fr = frs[p[0]]; cx, cy, hh, ww = p[1:]
                ims.append(Image.fromarray(fr[int(max(0, cy - hh / 2)):int(cy + hh / 2), int(max(0, cx - ww / 2)):int(cx + ww / 2)]))
        if not ims:
            continue
        nat.append(ims); ids.append(k)
        for c in t["nl"]:
            qs.append(c); ql.append(k)
    return "vehicle", nat, ids, qs, np.array(ql), None


def set_meva(max_windows):
    from src.index.sql_db import connect
    from eval.e2e.eval_activities import NAT, activity
    db = connect()
    rows = [dict(r) for r in db.execute("SELECT w.id, w.rel_start, w.rel_end, w.caption, c.path, c.source FROM windows w "
                                        "JOIN chunks c ON c.id=w.chunk_id")]
    rows = [r for r in rows if activity(r["source"])][:max_windows]
    classes = sorted({activity(r["source"]) for r in rows})
    qs = [NAT.get(a, a.replace("-", " ")) for a in classes]
    return rows, qs, np.array(classes)


def run_tracks(name, kind, nat, ids, qs, ql, rep, bs):
    t = time.time(); c1 = v1_caption(kind, nat, bs); t1 = time.time() - t
    big = [enlarge(ims, kind) for ims in nat]
    t = time.time(); c2, raw2 = v2_caption(kind, big, bs=bs); t2 = time.time() - t
    m1, m2 = metrics(rank_matrix(qs, c1), ql, ids), metrics(rank_matrix(qs, c2), ql, ids)
    fails = sum(not c for c in c2)
    print(f"[{name}] {len(ids)} items, {len(qs)} queries | v1 {m1} ({t1:.0f}s) | v2 {m2} ({t2:.0f}s, {fails} parse failures)", flush=True)
    rep[name] = {"items": len(ids), "v1": m1, "v2": m2, "v1_s": round(t1), "v2_s": round(t2), "v2_parse_failures": fails,
                 "samples": [{"query": qs[[i for i, l in enumerate(ql) if l == ids[j]][0]], "v1": c1[j], "v2": c2[j]}
                             for j in range(min(15, len(ids)))]}


def main(a):
    v2.load()
    rep = {"model": "Qwen2.5-VL-7B", "args": vars(a)}
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    for s in a.sets:
        if s == "rstp":
            kind, nat, ids, qs, ql, _ = set_rstp(a.n, a.cctv_height)
            run_tracks(f"rstp_h{a.cctv_height}", kind, nat, ids, qs, ql, rep, a.batch)
        elif s == "cityflow":
            kind, nat, ids, qs, ql, _ = set_cityflow(a.n // 2)
            run_tracks("cityflow", kind, nat, ids, qs, ql, rep, a.batch)
        elif s == "meva":
            from eval.e2e.eval_activities import activity
            rows, qs, cls = set_meva(a.max_windows)
            c1 = [r["caption"] or "" for r in rows]
            ims = [v2.window_images(r["path"], r["rel_start"], r["rel_end"]) for r in rows]
            t = time.time()
            c2, _ = v2_caption("window", ims, [{"dur": r["rel_end"] - r["rel_start"]} for r in rows], bs=max(1, a.batch // 3))
            t2 = time.time() - t
            lab = np.array([activity(r["source"]) for r in rows]); grp = [r["source"] for r in rows]
            m1 = first_hit_metrics(rank_matrix(qs, c1), cls, lab, grp)
            m2 = first_hit_metrics(rank_matrix(qs, c2), cls, lab, grp)
            print(f"[meva] {len(rows)} windows, {len(qs)} class queries | v1 {m1} | v2 {m2} ({t2:.0f}s)", flush=True)
            rep["meva"] = {"windows": len(rows), "v1": m1, "v2": m2, "v2_s": round(t2),
                           "samples": [{"label": lab[j], "v1": c1[j], "v2": c2[j]} for j in range(0, len(rows), max(1, len(rows) // 15))]}
        json.dump(rep, open(out, "w"), indent=1, default=str)
    print("wrote", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", nargs="+", default=["rstp", "cityflow", "meva"])
    ap.add_argument("--n", type=int, default=600)
    ap.add_argument("--cctv-height", type=int, default=100)
    ap.add_argument("--max-windows", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=12)
    ap.add_argument("--out", default="outputs/reports/caption_bench.json")
    main(ap.parse_args())
