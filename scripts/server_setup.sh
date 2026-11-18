#!/usr/bin/env bash
set -e
cd ~/cctv-search
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
[ -d .venv ] || uv venv --python 3.11
source .venv/bin/activate
uv pip install -r requirements.txt
mkdir -p data/indexes/{sqlite,qdrant} data/processed outputs/logs bin models/pretrained
[ -x bin/qdrant ] || curl -L https://github.com/qdrant/qdrant/releases/latest/download/qdrant-x86_64-unknown-linux-gnu.tar.gz | tar xz -C bin
hf download microsoft/xclip-base-patch32 --local-dir models/pretrained/xclip
hf download Qwen/Qwen2.5-VL-7B-Instruct --local-dir models/pretrained/qwen2.5-vl
hf download Qwen/Qwen2.5-3B-Instruct
command -v ffmpeg >/dev/null && command -v ffprobe >/dev/null || echo "!! install ffmpeg + ffprobe (apt or a static build)"
nvidia-smi --query-gpu=name,memory.total --format=csv
