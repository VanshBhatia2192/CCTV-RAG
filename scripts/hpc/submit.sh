#!/usr/bin/env bash
# Submit a job script, then watch it for ~40 s and cancel it if PBS keeps retrying (a node that
# rejects the job makes PBS loop thousands of times on a shared cluster).
#   bash scripts/hpc/submit.sh scripts/hpc/sr_gpu_test.pbs            # any node with a free GPU (default)
#   SR_HOST=gpu02 bash scripts/hpc/submit.sh scripts/hpc/sr_train.pbs # pin to one node
# Earlier (Nov 2026) gpu02 rejected jobs for new accounts; it accepts them now. If a node starts
# rejecting again, the retry guard below cancels the job: pin the other node with SR_HOST.
set -euo pipefail
cd "$(dirname "$0")/../.."
P=/opt/pbs/bin
SCRIPT="$1"; shift || true
HOST="${SR_HOST:-any}"
SEL=${SR_SELECT:-$(grep -E '^#PBS -l select=' "$SCRIPT" | head -1 | sed 's/^#PBS -l select=//')}   # SR_SELECT overrides, e.g. 1:ncpus=4:ngpus=1:mem=12gb
if [ "$HOST" = any ]; then PIN=""; else PIN=":host=${HOST}"; fi
J=$($P/qsub -l "select=${SEL}${PIN}" "$@" "$SCRIPT")
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
