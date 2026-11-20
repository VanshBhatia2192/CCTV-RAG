cd ~/cctv-search && source /home/soft/anaconda3/etc/profile.d/conda.sh && conda activate cctv
export QDRANT_URL=http://localhost:6335 NO_PROXY=localhost,127.0.0.1 HF_HOME=$PWD/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4
export LD_PRELOAD=$CONDA_PREFIX/lib/libstdc++.so.6
export DEMO_DB=/tmp/meta_demo_$USER.db
rm -f $DEMO_DB $DEMO_DB-wal $DEMO_DB-shm
python - <<'PY'
import os, sqlite3
src = sqlite3.connect("file:data/indexes/sqlite/meta.db?mode=ro&immutable=1", uri=True)
dst = sqlite3.connect(os.environ["DEMO_DB"]); src.backup(dst); dst.close()
print("local index copy:", os.environ["DEMO_DB"])
PY
pgrep -f bin/qdrant > /dev/null || { (cd data/indexes/qdrant && QDRANT__SERVICE__HTTP_PORT=6335 QDRANT__SERVICE__GRPC_PORT=6336 QDRANT__TELEMETRY_DISABLED=true nohup ../../../bin/qdrant > ../../../outputs/logs/qdrant_mgmt.log 2>&1 &); sleep 12; }
exec uvicorn app.main:app --host 0.0.0.0 --port 8000
