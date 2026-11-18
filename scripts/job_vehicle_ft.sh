set -e
cd "${PBS_O_WORKDIR:-$PWD}"
[ -n "$CONDA_PREFIX" ] && export LD_PRELOAD=$CONDA_PREFIX/lib/libstdc++.so.6
export HF_HOME=$PWD/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
python -m training.vehicle.finetune_clip --epochs ${EPOCHS:-12} --lr ${LR:-1e-5} --batch ${BATCH:-96}
