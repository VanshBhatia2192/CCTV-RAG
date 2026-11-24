#!/usr/bin/env bash
set -e
cd "${PBS_O_WORKDIR:-$PWD}"
export QDRANT_URL=http://localhost:6333 QDRANT__TELEMETRY_DISABLED=true
NPROC=${NPROC:-8}; TRK=${TRK:-3}; CAP=${CAP:-3}; BIG=1000000; L=outputs/logs
python -c "import urllib.request; urllib.request.urlopen('$QDRANT_URL')" >/dev/null 2>&1 || { (cd data/indexes/qdrant && nohup ../../../bin/qdrant > ../../../$L/qdrant.log 2>&1 &); sleep 6; }

echo "1/5 ingest"; find data/raw/meva/videos \( -iname '*.mp4' -o -iname '*.avi' \) | xargs -P $NPROC -I{} python -m src.ingest.chunk {} || true

echo "2/5 tracking (person + vehicle, $TRK shards each, one GPU)"
for k in person vehicle; do for s in $(seq 0 $((TRK-1))); do
  SHARD=$s NSHARDS=$TRK python -m src.person.detect_track --kind $k --limit $BIG > $L/track_${k}_$s.log 2>&1 &
done; done; wait

echo "3/5 windows + embeddings (3 jobs in parallel)"
python -m src.incident.windows
python -m src.person.embed --kind person --limit $BIG > $L/embed_person.log 2>&1 &
python -m src.person.embed --kind vehicle --limit $BIG > $L/embed_vehicle.log 2>&1 &
python -m src.incident.embed --limit $BIG > $L/embed_incident.log 2>&1 &
wait

if [ "${CAPTIONS:-1}" = 1 ]; then
  echo "4/5 captions ($CAP Qwen2.5-VL shards, ~16 GB each)"
  for s in $(seq 0 $((CAP-1))); do
    SHARD=$s NSHARDS=$CAP python -m src.incident.caption --limit $BIG > $L/caption_$s.log 2>&1 &
  done; wait
fi

echo "5/5 export"
python -c "from src.index.sql_db import connect; connect().execute('PRAGMA wal_checkpoint(TRUNCATE)')"
python -m scripts.qdrant_io export

pkill -f bin/qdrant || true
