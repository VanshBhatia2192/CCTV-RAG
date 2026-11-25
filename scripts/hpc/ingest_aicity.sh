#!/usr/bin/env bash
# Submit the AI City indexing chain: ingest -> v2 captions (person, vehicle, 2 window shards, vLLM) -> import.
# The shard counts match the existing outputs/captions_v2 files, so already captioned MEVA rows are skipped.
# Progress: tail -f outputs/logs/aicity_ingest.log outputs/logs/aicity_import.log ; qstat -u $USER
set -euo pipefail
cd "$(dirname "$0")/../.."
Q=/opt/pbs/bin/qsub
A=$($Q scripts/hpc/ingest_aicity.pbs)
echo "ingest:   $A"
B=()
for spec in person:0:1 vehicle:0:1 window:0:2 window:1:2; do
  IFS=: read -r k s n <<< "$spec"
  B+=("$($Q -W depend=afterok:$A -v MODE=caption,KIND=$k,SHARD=$s,NSHARDS=$n,BACKEND=vllm scripts/hpc/captions_v2.pbs)")
  echo "caption:  ${B[-1]} ($k shard $s/$n)"
done
C=$($Q -W depend=afterok:$(IFS=:; echo "${B[*]}") -v STAGE=import scripts/hpc/ingest_aicity.pbs)
echo "import:   $C"
