mkdir -p outputs/logs
W=${WAVE:-1}; PER=${PER_WAVE:-100}; LIM=${DATA_LIMIT_GB:-270}
USED=$(du -sk data | cut -f1); echo "wave $W: data dir uses $((USED/1048576)) GB (limit $LIM)"
[ $((USED/1048576)) -ge $LIM ] && { echo "STOP: data dir over limit"; exit 1; }
if [ ! -f outputs/scale_order.txt ]; then
python - <<'PY'
import os
d = "data/raw/meva/mevid_videos"
fs = sorted(f for f in os.listdir(d) if f.lower().endswith((".avi", ".mp4")))
order = sorted(range(len(fs)), key=lambda i: (i * 0.6180339887498949) % 1.0)
open("outputs/scale_order.txt", "w").write("\n".join(f"{d}/{fs[i]}" for i in order) + "\n")
print(len(fs), "videos ordered for waves")
PY
fi
[ -f outputs/wave_${W}_start_id.txt ] || python -c "from src.index.sql_db import connect; print(connect().execute('select coalesce(max(id),0) from chunks').fetchone()[0])" > outputs/wave_${W}_start_id.txt
sed -n "$(( (W-1)*PER + 1 )),$(( W*PER ))p" outputs/scale_order.txt > outputs/wave_${W}_videos.txt
echo "wave $W videos: $(wc -l < outputs/wave_${W}_videos.txt)"
echo "chunking"; cat outputs/wave_${W}_videos.txt | xargs -P 6 -I{} python -m src.ingest.chunk {} > outputs/logs/chunk_w${W}.log 2>&1
tail -n 2 outputs/logs/chunk_w${W}.log; echo "bad videos: $(grep -c 'SKIP bad' outputs/logs/chunk_w${W}.log)"
python - <<PY
import sqlite3, numpy as np, os
old = int(open("outputs/wave_${W}_start_id.txt").read())
d = sqlite3.connect("data/indexes/sqlite/meta.db", timeout=60)
rows = d.execute("select id, motion, path from chunks where id>?", (old,)).fetchall()
if rows:
    m = np.array([r[1] for r in rows]); thr = max(float(np.percentile(m, 50)), 0.0003)
    d.execute("update chunks set kept=(motion>=?) where id>?", (thr, old)); d.commit()
    gone = 0
    for i, mo, p in rows:
        if mo < thr and os.path.exists(p):
            os.remove(p); gone += 1
    print("new chunks", len(m), "motion p10/p50/p90", np.percentile(m, [10, 50, 90]), "thr", thr, "kept", int((m >= thr).sum()), "deleted unkept files", gone)
else:
    print("no new chunks")
PY
echo "tracking (2 shards per kind)"
for k in person vehicle; do for s in 0 1; do
  SHARD=$s NSHARDS=2 python -m src.person.detect_track --kind $k --limit 1000000 > outputs/logs/track_${k}_$s.log 2>&1 &
done; done; wait
python -m src.scale.tally; du -sh data/processed; df -h . | tail -n 1
