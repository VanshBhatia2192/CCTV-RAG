set -e
cd "${PBS_O_WORKDIR:-$PWD}"
[ -n "$CONDA_PREFIX" ] && export LD_PRELOAD=$CONDA_PREFIX/lib/libstdc++.so.6
python -m training.gait.finetune --model ${MODEL:-deepgaitv2_gait3d} --keep-trunc --min-asp 1.0 --steps ${STEPS:-2000} --lr ${LR:-0.01}
