#!/usr/bin/env bash
# Print where the GPU demo is running and the tunnel command for your laptop (waits until the app answers).
#   bash scripts/hpc/demo_url.sh
cd "$(dirname "$0")/../.."
L=outputs/logs/demo_gpu.log
for i in $(seq 1 90); do
  A=$(grep -oE "DEMO_AT [^ ]+" "$L" 2>/dev/null | tail -1 | cut -d" " -f2)
  if [ -n "$A" ] && curl -s --max-time 3 "http://$A/api/health" > /dev/null; then
    echo "demo is up on $A ($(grep "DEMO_AT" "$L" | tail -1 | cut -d" " -f3-))"
    echo "run on your laptop:  ssh -N -L 8000:$A hpc"
    echo "then open:           http://localhost:8000"
    exit 0
  fi
  sleep 5
done
echo "demo not answering yet; check: tail -n 40 $L   and   qstat -u \$USER"; exit 1
