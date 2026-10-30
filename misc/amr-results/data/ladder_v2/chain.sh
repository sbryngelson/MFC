#!/bin/bash
# Runs the given run directories back to back inside one allocation (each via its generated MFC.sh).
# A "?dir" entry runs only if the previous run failed (239-step fallback after a failed 240-step run).
# Usage (sbatch): sbatch -N <nodes> ... chain.sh <dir> [<dir> ...]
ulimit -c 0
B=/lustre/orion/cfd154/scratch/sbryngelson/mfc-simple-v2
echo "job $SLURM_JOB_ID nodes $SLURM_JOB_NODELIST start $(date)"
sha256sum /lustre/orion/cfd154/scratch/sbryngelson/mfc-amr-bench2/build/install/gpu-mp-*/bin/*
rc=0
for d in "$@"; do
  if [ "${d#\?}" != "$d" ]; then d=${d#\?}; [ $rc -eq 0 ] && { echo "$d skipped (previous run succeeded)"; continue; }; fi
  cd $B/$d || continue
  [ -e MFC.out ] && mv MFC.out $B/trash/$d.MFC.out.$(date +%s)
  echo "job $SLURM_JOB_ID nodes $SLURM_JOB_NODELIST" > MFC.out
  bash MFC.sh >> MFC.out 2>&1; rc=$?; echo "$d exit $rc $(date)"
done
