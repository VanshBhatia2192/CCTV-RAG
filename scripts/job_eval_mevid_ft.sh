set -e
cd "${PBS_O_WORKDIR:-$PWD}"
[ -n "$CONDA_PREFIX" ] && export LD_PRELOAD=$CONDA_PREFIX/lib/libstdc++.so.6
M=${MODEL:-deepgaitv2_gait3d}
python -m eval.gait.eval_mevid --stage all --keep-trunc --min-asp 1.0 --gait deepgaitv2_gait3d deepgaitv2_grew gaitbase_grew --gait-ft $M:outputs/gait_ft/${M}_ft.pt --cache outputs/gait_prep/emb_test_ft.npz --report outputs/reports/eval_mevid_ft.json
