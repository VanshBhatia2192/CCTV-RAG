#!/usr/bin/env bash
set -e
cd "${PBS_O_WORKDIR:-$PWD}"
source /home/soft/anaconda3/etc/profile.d/conda.sh
conda activate cctv
[ -n "$CONDA_PREFIX" ] && export LD_PRELOAD=$CONDA_PREFIX/lib/libstdc++.so.6
export HF_HOME=$PWD/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONUNBUFFERED=1

echo "=== Running Multi-Camera Vehicle Re-ID Fine-Tuning on AI City Dataset ==="
python -m training.vehicle.train_vehicle_reid --epochs ${EPOCHS:-12} --batch ${BATCH:-48} --lr ${LR:-1e-5}

echo "=== Re-embedding Vehicle Index in Qdrant with Fine-Tuned Weights ==="
export QDRANT_URL=http://localhost:6335 NO_PROXY=localhost,127.0.0.1
python -m src.vehicle.embed_ft --kind vehicle --collection vehicle_ft

echo "=== Vehicle Re-ID Fine-Tuning and Re-indexing Complete! ==="

