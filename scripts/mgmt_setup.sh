#!/usr/bin/env bash
# Run ON THE MGMT NODE (has internet), once.
set -e
cd ~/cctv-search
source /home/soft/anaconda3/etc/profile.d/conda.sh
conda env list | grep -q '^cctv ' || conda create -y -n cctv python=3.11
conda activate cctv
conda install -y -c conda-forge ffmpeg
pip install -r requirements.txt
mkdir -p data/indexes/{sqlite,qdrant} data/processed outputs/logs bin models/pretrained
export HF_HOME=$PWD/hf_cache   # models live inside the project, so the GPU node can see them
[ -x bin/qdrant ] || curl -L https://github.com/qdrant/qdrant/releases/latest/download/qdrant-x86_64-unknown-linux-gnu.tar.gz | tar xz -C bin
hf download microsoft/xclip-base-patch32 --local-dir models/pretrained/xclip
hf download Qwen/Qwen2.5-VL-7B-Instruct --local-dir models/pretrained/qwen2.5-vl
hf download Qwen/Qwen2.5-3B-Instruct
hf download openai/clip-vit-base-patch32
ls models/pretrained/yolov8m.pt || echo "!! copy yolov8m.pt from your Mac (step 2)"
python -c "import torch; print('torch', torch.__version__)"
