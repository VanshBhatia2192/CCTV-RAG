set -e
cd "${PBS_O_WORKDIR:-$PWD}"
[ -n "$CONDA_PREFIX" ] && export LD_PRELOAD=$CONDA_PREFIX/lib/libstdc++.so.6
export HF_HOME=$PWD/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
python -m src.vehicle.embed_ft --kind vehicle --collection vehicle_ft
