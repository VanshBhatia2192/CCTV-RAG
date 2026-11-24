set -e
cd "${PBS_O_WORKDIR:-$PWD}"
export LD_PRELOAD=$CONDA_PREFIX/lib/libstdc++.so.6
mkdir -p outputs/reports
python -m eval.person.eval_pseudo --n 300 --seed 0 2>&1 | tee outputs/reports/eval_pseudo_scaled.txt
python -m eval.latency_scaled
