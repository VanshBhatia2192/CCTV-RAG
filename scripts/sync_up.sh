#!/usr/bin/env bash
set -e
S=${1:?usage: sync_up.sh user@host [remote_dir]}; R=${2:-cctv-search}
ssh $S "mkdir -p $R/data/raw/meva"
rsync -avP --exclude /data --exclude /models --exclude /.venv --exclude /outputs --exclude __pycache__ --exclude .git ./ $S:$R/
rsync -avP data/raw/meva/videos $S:$R/data/raw/meva/
