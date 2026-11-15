import argparse, json, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from src.common.config import PIPE
from src.index.sql_db import connect
from src.incident import embed as iembed
from src.captions import hybrid
from src.query import parser, localize as loc, trim as tr

FT = Path("models/finetuned/text2person/clip_t2p.pt")

def _weights(kind):
    p = Path("configs/fusion.json")
    w = json.load(open(p)).get(kind) if p.exists() else None
    return tuple(w) if w else (1.0, 1.0, 1.0)

def search(query, rerank=False, use_llm=True, n_clips=3, ft=True):
    t0 = time.time(); T = {}
    db = connect()
    q = parser.parse(query, use_llm); kind, text = q["type"], q["text"]
    T["parse"] = time.time() - t0
    mode = "hybrid_ft" if (kind == "person" and ft and FT.exists()) else "hybrid"
    where = {"camera": q["camera"]} if q["camera"] else None
    t = time.time()
    cands = hybrid.retrieve(kind, text, PIPE["retrieval"]["top_k"], mode, where, w=_weights(kind))
    T["retrieve"] = time.time() - t
    if rerank and cands:
        from src.query import rerank as rr
        t = time.time()
        if kind != "incident":
            for c in cands:
                c["crops_dir"] = db.execute("SELECT crops_dir FROM tracks WHERE id=?", (c["id"],)).fetchone()[0]
        cands = rr.rerank(kind, text, cands[:PIPE["retrieval"]["rerank_k"]])
        T["rerank"] = time.time() - t
    seen, picks = set(), []
    for c in cands:
        if c["chunk_id"] in seen:
            continue
        seen.add(c["chunk_id"]); picks.append(c)
        if len(picks) == n_clips:
            break
    t = time.time()
    vec = iembed.embed_text([text])[0] if kind == "incident" and picks else None
    spans = []
    for c in picks:
        ch = dict(db.execute("SELECT * FROM chunks WHERE id=?", (c["chunk_id"],)).fetchone())
        a, b = loc.localize(kind, vec, c, ch)
        spans.append((c, ch, a, b))
    T["localize"] = time.time() - t
    t = time.time()
    with ThreadPoolExecutor(max(1, len(spans))) as ex:
        clips = list(ex.map(lambda s: tr.trim(s[1]["path"], s[2], s[3], dur=s[1]["end_s"] - s[1]["start_s"]), spans))
    T["trim"] = time.time() - t
    tbl = "windows" if kind == "incident" else "tracks"
    out = []
    for (c, ch, a, b), cl in zip(spans, clips):
        cap = db.execute(f"SELECT caption FROM {tbl} WHERE id=?", (c["id"],)).fetchone()
        out.append({"id": c["id"], "clip": cl, "camera": c["camera"], "score": round(c["score"], 4), "vlm": c.get("vlm"),
                    "start_s": ch["start_s"] + a, "end_s": ch["start_s"] + b, "caption": cap[0] if cap else None, "source": Path(ch["source"]).name})
    return {"query": q, "mode": mode, "results": out, "latency_s": round(time.time() - t0, 1),
            "timings": {k: round(v, 2) for k, v in T.items()}}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("query"); ap.add_argument("--rerank", action="store_true"); ap.add_argument("--no-llm", action="store_true")
    a = ap.parse_args()
    print(json.dumps(search(a.query, a.rerank, not a.no_llm), indent=1, default=str))
