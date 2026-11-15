import time, json, statistics as st
from src.query import parser
from src.captions import hybrid
QS = ["a person running", "a red car turning left", "a man in a black jacket carrying a backpack",
      "a white van stopped", "two people fighting", "a person walking near a bus", "a truck driving away",
      "a woman with a blue bag", "a person standing near a car", "a car crash"]
out = {}
for use_llm in (False, True):
    parser.parse(QS[0], use_llm)  # warm-up (loads the parser LLM once)
    tp, tr = [], []
    for q in QS:
        t0 = time.time(); p = parser.parse(q, use_llm); t1 = time.time()
        hybrid.retrieve(p["type"], p["text"], k=20); t2 = time.time()
        tp.append(t1 - t0); tr.append(t2 - t1)
    tot = [a + b for a, b in zip(tp, tr)]
    out[f"llm={use_llm}"] = dict(parse_med=st.median(tp), retrieve_med=st.median(tr),
                                 total_med=st.median(tot), total_max=max(tot))
    print(f"llm={use_llm}", {k: round(v, 3) for k, v in out[f"llm={use_llm}"].items()})
json.dump(out, open("outputs/reports/latency_scaled.json", "w"), indent=1)
