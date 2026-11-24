"""Context-aware search: plan the query, retrieve, score every requirement separately, optionally VLM-verify.

    from src.query import contextual
    r = contextual.search("red bluewhale")
    r["notice"]  -> 'No footage in the index mentions "bluewhale". Showing partial matches (red) only.'

Run from the project root:  python -m src.query.contextual "man in a red jacket with a backpack" [--llm] [--verify]
"""
import argparse, json, time
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


def _notice(plan, hits, unsup, verified_on, n_rejected):
    full = sum(h["match"] == "full" for h in hits)
    msgs = []
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
           strict=False):
    t0 = time.time(); T = {}
    cp = C.corpus()
    plan = P.make_plan(q, use_llm=use_llm, cameras=cameras(), vocab=cp.vocab)
    if camera:
        plan["camera"] = camera
    T["plan"] = time.time() - t0
    unsup = C.unsupported(plan, cp)

    t = time.time()
    where = {k: v for k, v in (("camera", plan["camera"]), ("dataset", dataset and dataset.lower())) if v}
    kinds = ["person", "vehicle"] if plan["kind"] == "any" else [plan["kind"]]
    terms = [c["value"] for c in plan["constraints"] if c["type"] in ("text", "term") and c not in unsup]
    hits = []
    for kind in kinds:
        extra = []
        for term in terms:                       # make sure rare named things are in the candidate pool
            ids = cp.ids_with(kind, term)
            if 0 < len(ids) <= pool:
                extra.append(ids)
        sat = cp.satisfying(kind, plan)          # captions that already satisfy every requirement
        if sat:
            from src.captions import text_index as TI
            rank = {i: r for r, (i, _) in enumerate(TI.bm25_search(kind, plan["text"], 20000))}
            extra.append(sorted(sat, key=lambda i: rank.get(i, len(rank)))[:pool])
        k = min(max(n * 4, 48), pool)
        got = hybrid.retrieve(kind, plan["text"], k=k, mode=mode_for(kind), where=where or None,
                              w=fusion_weights(kind), extra=extra)
        if not got and dataset:                  # dataset tag missing on some chunks: retry without it
            got = hybrid.retrieve(kind, plan["text"], k=k, mode=mode_for(kind),
                                  where={"camera": plan["camera"]} if plan["camera"] else None,
                                  w=fusion_weights(kind), extra=extra)
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
