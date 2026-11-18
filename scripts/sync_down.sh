#!/usr/bin/env bash
set -e
S=${1:?usage: sync_down.sh user@host [remote_dir]}; R=${2:-cctv-search}
T=$(date +%s); mkdir -p data/_mac_backup
[ -d data/indexes ] && mv data/indexes data/_mac_backup/indexes_$T
[ -d data/processed ] && mv data/processed data/_mac_backup/processed_$T
mkdir -p data/indexes data/processed outputs
rsync -avP $S:$R/data/indexes/sqlite $S:$R/data/indexes/qdrant_export.pkl data/indexes/
rsync -avP $S:$R/data/processed/chunks $S:$R/data/processed/crops data/processed/
rsync -avP $S:$R/outputs/logs outputs/
source .venv/bin/activate && python -m scripts.qdrant_io import
