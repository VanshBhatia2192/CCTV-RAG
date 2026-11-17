#!/usr/bin/env bash
# One-time setup on the UPES CoE-AI cluster login node (light work only: installs + small downloads).
#   bash scripts/hpc/setup_env.sh
# Creates a private Python 3.11 env (torch for CUDA 12.8, OpenCV, ffmpeg) in ~/envs/sr and fetches the
# official BasicVSR++ REDS4 weights. Safe to re-run: finished steps are skipped.
set -euo pipefail
cd "$(dirname "$0")/../.."
ENV="${SR_ENV:-$HOME/envs/sr}"
CONDA=/home/soft/anaconda3/bin/conda

if [ ! -x "$ENV/bin/python" ]; then
  echo ">> creating $ENV (python 3.11 + ffmpeg from conda-forge)"
  "$CONDA" create -y -q -p "$ENV" --override-channels -c conda-forge python=3.11 ffmpeg
fi
echo ">> installing torch (cu128) and deps"
"$ENV/bin/pip" install -q --upgrade pip
"$ENV/bin/pip" install -q torch torchvision --index-url https://download.pytorch.org/whl/cu128
"$ENV/bin/pip" install -q opencv-python-headless pyyaml numpy pytest

W=models/pretrained/basicvsrpp_reds4.pth
if [ ! -s "$W" ]; then
  echo ">> downloading REDS4 weights"
  mkdir -p models/pretrained
  curl -fL --retry 3 -o "$W.part" \
    https://download.openmmlab.com/mmediting/restorers/basicvsr_plusplus/basicvsr_plusplus_c64n7_8x1_600k_reds4_20210217-db622b2f.pth
  mv "$W.part" "$W"
fi

echo ">> checks"
"$ENV/bin/python" -c "import torch, torchvision, cv2; print('torch', torch.__version__, 'cuda build', torch.version.cuda, '| torchvision', torchvision.__version__, '| cv2', cv2.__version__)"
"$ENV/bin/ffmpeg" -hide_banner -version | head -1
"$ENV/bin/python" -m pytest -q tests/test_sr.py -p no:cacheprovider
echo ">> setup ok"
