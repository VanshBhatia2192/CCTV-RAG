set -e
cd "${PBS_O_WORKDIR:-$PWD}"
[ -n "$CONDA_PREFIX" ] && export LD_PRELOAD=$CONDA_PREFIX/lib/libstdc++.so.6
python -m eval.gait.eval_mevid --stage all --keep-trunc --min-asp ${ASP:-1.0} --cache outputs/gait_prep/emb_test_loose.npz --report outputs/reports/eval_mevid_loose.json
