#!/usr/bin/env bash
set -e
R=data/raw

# 0. Python env with uv (brew install uv, if missing)
command -v uv >/dev/null || brew install uv
[ -d .venv ] || uv venv --python 3.11
source .venv/bin/activate
uv pip install -U awscli huggingface_hub gdown

# 1. Check GitHub SSH auth first
ssh -T git@github.com 2>&1 | grep -q "successfully authenticated" || {
  echo "GitHub SSH not set up: ssh-keygen -t ed25519, then add ~/.ssh/id_ed25519.pub at github.com/settings/keys"; exit 1; }

# 2. Repos over SSH
clone() { [ -d "$2" ] || git clone "$1" "$2"; }
clone git@github.com:Kitware/MEVID.git $R/mevid/repo
clone git@github.com:NjtechCVLab/RSTPReid-Dataset.git $R/rstpreid/repo
clone git@github.com:Xuange923/Surveillance-Video-Understanding.git $R/uca/repo
clone git@github.com:Kitware/meva-data-repo.git $R/meva/annotations
clone git@github.com:ShiqiYu/OpenGait.git models/pretrained/OpenGait

# 3. MEVA videos: list first, sync a small subset only
aws s3 ls --no-sign-request s3://mevadata-public-01/ | head
# aws s3 sync --no-sign-request s3://mevadata-public-01/<subfolder> $R/meva/videos --exclude "*" --include "*2018-03-07*"

# 4. Weights
mkdir -p models/pretrained
[ -f models/pretrained/RealESRGAN_x4plus.pth ] || curl -L -o models/pretrained/RealESRGAN_x4plus.pth \
  https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth
hf download microsoft/xclip-base-patch32 --local-dir models/pretrained/xclip
hf download Qwen/Qwen2.5-VL-7B-Instruct --local-dir models/pretrained/qwen2.5-vl

cat <<MSG
MANUAL (request/registration required, place files in the folder shown):
 data/raw/cityflow_nl   <- https://www.aicitychallenge.org
 data/raw/cuhk_pedes    <- request via GitHub repo form
 data/raw/ucf_crime     <- https://www.crcv.ucf.edu/projects/real-world/
 data/raw/gait/{grew,gait3d,casia_b,ou_mvlp} <- request pages
MSG
