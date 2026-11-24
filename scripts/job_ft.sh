#!/usr/bin/env bash
set -e
export LD_PRELOAD=$CONDA_PREFIX/lib/libstdc++.so.6

python -m src.person.embed_ft
python -m eval.person.eval_pseudo --n 300
