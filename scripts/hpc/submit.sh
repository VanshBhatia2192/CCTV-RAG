#!/usr/bin/env bash
# Submit a job script pinned to one node, then watch it for ~40 s and cancel it if PBS keeps
# retrying (a node that rejects the job makes PBS loop thousands of times on a shared cluster).
#   bash scripts/hpc/submit.sh scripts/hpc/sr_gpu_test.pbs            # pinned to gpu01 by default
#   SR_HOST=gpu02 bash scripts/hpc/submit.sh scripts/hpc/sr_train.pbs
# Note (Nov 2026): gpu02 rejected every job for new accounts ("pbs_mom: System error: No such file or
# directory"), so gpu01 is the default until that is fixed.
set -euo pipefail
cd "$(dirname "$0")/../.."
P=/opt/pbs/bin
SCRIPT="$1"; shift || true
HOST="${SR_HOST:-gpu01}"
SEL=$(grep -E '^#PBS -l select=' "$SCRIPT" | head -1 | sed 's/^#PBS -l select=//')
J=$($P/qsub -l "select=${SEL}:host=${HOST}" "$@" "$SCRIPT")
echo "submitted $J to $HOST (select=${SEL})"
for i in $(seq 1 10); do
  sleep 4
  S=$($P/qstat -xf "$J" 2>/dev/null | grep -E "job_state|run_count" | tr -s ' ' | tr '\n' ' ')
  RC=$(echo "$S" | grep -oE "run_count = [0-9]+" | grep -oE "[0-9]+" || echo 0)
  if [ "${RC:-0}" -gt 3 ]; then
    $P/qdel "$J"; echo "job was being rejected and retried ($RC runs) -> cancelled"; exit 1
  fi
  case "$S" in *"job_state = R"*) echo "running: $S"; exit 0;; *"job_state = F"*) echo "finished: $S"; exit 0;; esac
done
echo "still queued (waiting for a free GPU): $S"
