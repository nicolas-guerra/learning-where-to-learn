#!/usr/bin/env bash
set -euo pipefail

# Ns to run
# Ns=(12 48 84 120 156 192 228 264 300 336)
Ns=(12 24 36 48 60 72 84 96 108 120)
mkdir -p logs

# run_group GPU EPI LOGPREFIX
run_group() {
  local GPU=$1
  local EPI=$2
  local PREFIX=$3
  local AGGLOG="test_logs/${PREFIX}.log"

  echo "[$(date)] START group ep_index=${EPI} on GPU ${GPU}" | tee -a "${AGGLOG}"
  for N in "${Ns[@]}"; do
    local RUNLOG="test_logs/${PREFIX}_N${N}_$(date +%Y%m%d-%H%M%S).log"
    echo "[$(date)] GPU=${GPU} ep=${EPI} N=${N} -> ${RUNLOG}" | tee -a "${AGGLOG}"
    # -u for unbuffered python output, tee appends to AGGLOG and also writes per-run file
    CUDA_VISIBLE_DEVICES=${GPU} python3 -u particle_script.py --ep_index "${EPI}" --N "${N}" 2>&1 | tee -a "${AGGLOG}" > "${RUNLOG}"
    echo "[$(date)] Finished N=${N}" | tee -a "${AGGLOG}"
  done
  echo "[$(date)] DONE group ep_index=${EPI} on GPU ${GPU}" | tee -a "${AGGLOG}"
}

# launch each group in background (each will run its loop sequentially on its GPU)
run_group 0 -3 gpu0_ep-3 &
run_group 1 3  gpu1_ep3  &
run_group 2 -1 gpu2_ep-1 &

# wait for all background jobs to finish
wait
echo "[$(date)] All groups finished."
