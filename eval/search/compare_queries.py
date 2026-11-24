"""Before/after comparison of the /api/ask retrieval on a fixed query list.

before = old path: rules parser -> hybrid.retrieve(kind, text) (what the UI did with both toggles off)
after  = src.query.contextual.search (planner -> retrieval -> per-requirement scoring [-> VLM verify])

"full@10" counts top-10 results whose caption/attributes satisfy every planned requirement, judged by the same
constraint checker for both systems (so it measures "does the top of the list respect the query", not
ground truth). Captions are printed so results can be eyeballed.

    DEMO_DB=/tmp/meta_demo_$USER.db QDRANT_URL=http://localhost:6335 python -m eval.search.compare_queries [--llm] [--verify]
"""
import argparse, json, os, time
from pathlib import Path
from src.common.config import PATHS
if os.environ.get("DEMO_DB"):
    PATHS["sqlite"] = os.environ["DEMO_DB"]
from src.captions import hybrid
from src.query import parser, contextual, constraints as C

QUERIES = ["red bluewhale", "red car after 5pm", "woman in a black coat in the morning", "kid in a blue hoodie",
           "cop", "old man with a cane", "delivery guy carrying a box", "cyclist", "blue whale", "man in a red jacket with a backpack walking left", "person in red pants",
           "white truck turning right", "budget truck", 'bag with "catalyst" written on it',
           "woman in a black coat without a bag", "someone running", "black suv parked", "person wearing a green cap",
           "yellow school bus", "person riding a bicycle", "red car", "people fighting",
           "guy in a whitman football shirt", "persn in a blu hoodie"]


def before(q, k=10):
    p = parser.parse(q, use_llm=False)
    kind = p["type"]
    hits = hybrid.retrieve(kind, p["text"], k=96, mode=contextual.mode_for(kind), w=contextual.fusion_weights(kind))
    for h in hits:
        h["kind"] = kind
    return hits[:k]


def main(a):
    cp = C.corpus()
    rep = []
    for q in a.queries or QUERIES:
        t = time.time(); b = before(q); tb = time.time() - t
        r = contextual.search(q, n=a.k, use_llm=a.llm, verify=a.verify, rewrite=a.rewrite)
        plan = r["plan"]
        judged_b = C.score(plan, [dict(h) for h in b], cp)          # judge old results with the same checker
        fb = sum(h["match"] == "full" for h in judged_b)
        af = r["hits"][:a.k]
        fa = sum(h["match"] == "full" for h in af)
        print(f"\n=== {q!r}\n  understood: {r['understood']}   planner={plan['planner']}  unsupported={r['unsupported']}")
        print(f"  notice: {r['notice']}")
        print(f"  before full@{a.k}={fb}  ({tb:.2f}s) | after full@{a.k}={fa}  ({r['latency_s']}s {r['timings']})")
        for tag, lst in (("B", [h for h in b]), ("A", af)):
            for h in lst[:a.show]:
                cap = (h.get("caption") or cp.caption(h["kind"], h["id"]) or "")[:105]
                extra = f"{h.get('match', ''):8s} miss={h.get('missing', [])}" if tag == "A" else ""
                ver = f" vlm={h.get('vlm')}" if tag == "A" and a.verify else ""
                print(f"   {tag} {h['kind'][:3]} {h['id']:>7} {extra}{ver} | {cap}")
        rep.append(dict(query=q, plan=plan, understood=r["understood"], notice=r["notice"], unsupported=r["unsupported"],
                        before_full=fb, after_full=fa, latency_after=r["latency_s"], timings=r["timings"],
                        before=[dict(id=h["id"], kind=h["kind"], caption=cp.caption(h["kind"], h["id"])) for h in b],
                        after=[{k: h.get(k) for k in ("id", "kind", "match", "matched", "missing", "coverage",
                                                       "verified", "verify", "caption")} for h in af]))
    tot_b, tot_a = sum(x["before_full"] for x in rep), sum(x["after_full"] for x in rep)
    print(f"\nTOTAL full@{a.k}: before {tot_b} / after {tot_a} over {len(rep)} queries")
    out = Path(a.out or f"outputs/reports/search_compare{'_llm' if a.llm else ''}{'_verify' if a.verify else ''}"
                       f"{'_rewrite' if a.rewrite else ''}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(dict(k=a.k, llm=a.llm, verify=a.verify, total_before=tot_b, total_after=tot_a, queries=rep),
              open(out, "w"), indent=1, default=str)
    print("wrote", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("queries", nargs="*"); ap.add_argument("--llm", action="store_true")
    ap.add_argument("--verify", action="store_true"); ap.add_argument("-k", type=int, default=10)
    ap.add_argument("--rewrite", action="store_true")
    ap.add_argument("--show", type=int, default=5); ap.add_argument("--out")
    main(ap.parse_args())
