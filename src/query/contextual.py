"""Context-aware search: plan the query, retrieve, score every requirement separately, optionally VLM-verify.

    from src.query import contextual
    r = contextual.search("red bluewhale")
    r["notice"]  -> 'No footage in the index mentions "bluewhale". Showing partial matches (red) only.'

Run from the project root:  python -m src.query.contextual "man in a red jacket with a backpack" [--llm] [--verify]
"""
import argparse, json, re, time
from pathlib import Path
from src.captions import hybrid
from src.query import plan as P, constraints as C

FT = {"person": "models/finetuned/text2person/clip_t2p.pt", "vehicle": "models/finetuned/vehicle/clip_vehicle.pt"}
_CAMS = None


def mode_for(kind):
    return "hybrid_ft" if kind in FT and Path(FT[kind]).exists() else "hybrid"


def fusion_weights(kind):
    p = Path("configs/fusion.json")
    w = json.load(open(p)).get(kind) if p.exists() else None
    return tuple(w) if w else (1.0, 1.0, 1.0)


def cameras():
    global _CAMS
    if _CAMS is None:
        from src.index.sql_db import connect
        db = connect()
        _CAMS = sorted({r[0] for r in db.execute("SELECT DISTINCT camera FROM chunks") if r[0]})
        db.close()
    return _CAMS


_CLOCK = None
STAMP = re.compile(r"(20\d\d-\d\d-\d\d)\.(\d\d)-(\d\d)-(\d\d)")


def clock_index():
    """{kind: [(id, 'YYYY-MM-DD', seconds since midnight)]} from the recording time in MEVA file names
    (2018-03-07.17-00-00....avi) plus the item's offset in the source video. Items without a timestamp are absent."""
    global _CLOCK
    if _CLOCK is None:
        from src.index.sql_db import connect
        db = connect(); _CLOCK = {"person": [], "vehicle": [], "incident": []}
        def stamp(src):
            m = STAMP.search(src or "")
            return (m.group(1), int(m.group(2)) * 3600 + int(m.group(3)) * 60 + int(m.group(4))) if m else None
        for i, k, src, st in db.execute("SELECT t.id, t.kind, c.source, t.start_s FROM tracks t JOIN chunks c ON c.id=t.chunk_id"):
            s0 = stamp(src)
            if s0 and k in _CLOCK:
                _CLOCK[k].append((i, s0[0], (s0[1] + int(st or 0)) % 86400))
        for i, src, st in db.execute("SELECT w.id, c.source, w.start_s FROM windows w JOIN chunks c ON c.id=w.chunk_id"):
            s0 = stamp(src)
            if s0:
                _CLOCK["incident"].append((i, s0[0], (s0[1] + int(st or 0)) % 86400))
        db.close()
    return _CLOCK


def dataset_ids(kind, dataset):
    """Ids of `kind` from one dataset (see src/index/datasets.py). Used as a restrict set so every channel (visual,
    caption text, BM25, rare terms) searches only that dataset."""
    from src.index import datasets
    return set(datasets.ids(kind, dataset))


def time_filter(kind, t):
    """Ids of `kind` recorded inside the plan's time window (None = no time constraint)."""
    if not t:
        return None
    sec = lambda hm: int(hm[:2]) * 3600 + int(hm[3:5]) * 60
    a, b = sec(t["start"]) if t.get("start") else 0, sec(t["end"]) if t.get("end") else 86400
    date = t.get("date")
    def ok(d, x):
        if date and not (d == date or (date.startswith("*-") and d.endswith(date[1:]))):
            return False
        return a <= x < b if a <= b else (x >= a or x < b)      # windows that wrap midnight ("night")
    return {i for i, d, x in clock_index().get(kind, []) if ok(d, x)}


def _notice(plan, hits, unsup, verified_on, n_rejected):
    full = sum(h["match"] == "full" for h in hits)
    msgs = []
    t = plan.get("time")
    if t and not hits:
        msgs.append("No footage was recorded in that time window (only videos with a timestamp in the file name can be "
                    "filtered by time).")
    if unsup:
        terms = ", ".join(f'"{c["value"]}"' for c in unsup)
        msgs.append(f"No caption in the index mentions {terms}.")
    if not full and hits:
        have = sorted({m for h in hits[:10] for m in h["matched"]})
        msgs.append("No result matches every part of the query. Showing partial matches"
                    + (f" ({', '.join(have)})" if have else "") + ".")
    if verified_on and n_rejected:
        msgs.append(f"VLM check rejected {n_rejected} of the top results.")
    return " ".join(msgs) or None


def search(q, n=24, use_llm=False, verify=False, camera=None, dataset=None, pool=150, verify_top=12,
           strict=False, rewrite=False):
    t0 = time.time(); T = {}
    cp = C.corpus()
    plan = P.make_plan(q, use_llm=use_llm, cameras=cameras(), vocab=cp.vocab)
    if camera:
        plan["camera"] = camera
    T["plan"] = time.time() - t0
    unsup = C.unsupported(plan, cp)

    t = time.time()
    dataset = dataset and dataset.lower()
    where = {"camera": plan["camera"]} if plan["camera"] else {}
    kinds = ["person", "vehicle"] if plan["kind"] == "any" else [plan["kind"]]
    plan["search_text"] = P.rewrite(plan) if rewrite else plan["text"]   # caption-style wording for the search
    terms = [c["value"] for c in plan["constraints"] if c["type"] in ("text", "term") and c not in unsup]
    hits = []
    for kind in kinds:
        extra = []
        for term in terms:                       # make sure rare named things are in the candidate pool
            ids = cp.ids_with(kind, term)
            if 0 < len(ids) <= pool:
                extra.append(ids)
        allowed = time_filter(kind, plan.get("time"))
        if dataset:
            ds = dataset_ids(kind, dataset)
            allowed = ds if allowed is None else allowed & ds
        if allowed is not None:
            extra = [[i for i in e if i in allowed] for e in extra]
        sat = cp.satisfying(kind, plan)          # captions that already satisfy every requirement
        if allowed is not None:
            sat = [i for i in sat if i in allowed]
        if sat:
            from src.captions import text_index as TI
            rank = {i: r for r, (i, _) in enumerate(TI.bm25_search(kind, plan["search_text"], 20000))}
            extra.append(sorted(sat, key=lambda i: rank.get(i, len(rank)))[:pool])
        k = min(max(n * 4, 48), pool)
        got = hybrid.retrieve(kind, plan["search_text"], k=k, mode=mode_for(kind), where=where or None,
                              w=fusion_weights(kind), extra=extra, restrict=allowed)
        for h in got:
            h["kind"] = kind
        hits += got
    T["retrieve"] = time.time() - t

    t = time.time()
    hits = C.score(plan, hits, cp)
    T["score"] = time.time() - t

    n_rej = 0
    if verify and hits:
        t = time.time()
        from src.query import verify as V
        # verify the best candidates: full matches first, and partial ones too when nothing matched fully
        V.verify(plan, hits, top=verify_top)
        n_rej = sum(h.get("match") == "rejected" for h in hits)
        tier = {"full": 2, "partial": 1}
        hits = sorted([h for h in hits if h.get("match") not in ("rejected", "excluded")],
                      key=lambda h: (tier.get(h["match"], 0), h.get("verified") is True, h["coverage"],
                                     h.get("vlm") or 0), reverse=True)
        T["verify"] = time.time() - t
    else:
        hits = [h for h in hits if h["match"] != "excluded"]
    if strict:
        hits = [h for h in hits if h["match"] == "full" or h.get("verified")]
    return {"plan": plan, "understood": P.describe(plan), "unsupported": [c["value"] for c in unsup],
            "notice": _notice(plan, hits, unsup, verify, n_rej), "hits": hits,
            "timings": {k: round(v, 3) for k, v in T.items()}, "latency_s": round(time.time() - t0, 2)}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("query"); ap.add_argument("--llm", action="store_true"); ap.add_argument("--verify", action="store_true")
    ap.add_argument("-n", type=int, default=10)
    a = ap.parse_args()
    r = search(a.query, n=a.n, use_llm=a.llm, verify=a.verify)
    print(json.dumps({k: v for k, v in r.items() if k != "hits"}, indent=1))
    for h in r["hits"][:a.n]:
        print(f'{h["match"]:8s} {h["kind"]:8s} {h["id"]:>7} cov={h["coverage"]:.2f} vlm={h.get("vlm")} '
              f'miss={h["missing"]} | {(h["caption"] or "")[:110]}')
