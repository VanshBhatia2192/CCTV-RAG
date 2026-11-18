set -e
cd "${PBS_O_WORKDIR:-$PWD}"
[ -n "$CONDA_PREFIX" ] && export LD_PRELOAD=$CONDA_PREFIX/lib/libstdc++.so.6
python -m src.gait.index_gait --limit-chunks 40 --no-upsert --fmt ${FMT:-cxcywh} --K ${K:-3}
