python - <<'PY'
from src.index.sql_db import connect
d = connect(); d.execute("UPDATE tracks SET caption=REPLACE(caption,' none.','.') WHERE caption LIKE '% none.'"); d.commit()
PY
python -m src.captions.text_index build
python -m eval.e2e.tune_fusion
python -m eval.e2e.eval_activities --kinds person --modes visual visual_ft hybrid_ft --out outputs/reports/eval_activities_person_ft.json
python - <<'PY'
from src.query.search import search
for q in ["a person in a dark jacket carrying a bag", "a white car turning left", "a person getting into a car"]:
    for i in range(2):
        r = search(q, rerank=False)
        print("RUN", i, "|", q, "|", r["query"]["type"], r["mode"], "|", r["latency_s"], "s", r["timings"], "| clips:", len(r["results"]), flush=True)
PY
