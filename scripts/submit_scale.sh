#!/usr/bin/env bash
cd /home/vansh.122433/cctv-search
W0=${FROM_WAVE:-1}; W1=${TO_WAVE:-4}; PREV=""
for W in $(seq $W0 $W1); do
  dep=""; [ -n "$PREV" ] && dep="-W depend=afterok:$PREV"
  A=$(qsub -l walltime=10:00:00 $dep -v J=scripts/job_scale_a.sh,WAVE=$W scripts/gpu_job.pbs)
  B=$(qsub -l walltime=06:00:00 -W depend=afterok:$A -v J=scripts/job_scale_b.sh scripts/gpu_job.pbs)
  C=$(qsub -l walltime=12:00:00 -W depend=afterok:$B -v J=scripts/job_scale_c.sh scripts/gpu_job.pbs)
  echo "wave $W: A=$A B=$B C=$C"; PREV=$C
done
