#!/usr/bin/env bash
# Nut-yaw sensitivity (user, 2026-11-14), offline on the 250.ckpts, after run_final_queue.sh (so the token-block and
# k128 checkpoints are included). Detached:
#   setsid nohup bash scripts/sim/run_yaw_sens.sh > R3D/data/outputs/sim/mimicgen/yaw_sensitivity/chain.log 2>&1 &
set -uo pipefail
here="$(cd "$(dirname "$0")/../.." && pwd)"
cd "${here}"
log() { echo "[$(date '+%F %T')] $*"; }
busy() { ps -eo cmd | awk '($1=="bash" && $2 ~ /run_final_queue|run_mimicgen_queue|run_remover_sweep|run_claim_sweep|run_token_budget/) || ($1 ~ /python/ && ($0 ~ /train\.py/ || $0 ~ /eval_mimicgen_ckpt/ || $0 ~ /eval_robotwin2_ckpt/))' | grep -q .; }
log "waiting for the final queue to finish and the GPU to be free for 10 min"
free=0
while [ "${free}" -lt 10 ]; do
    if busy; then free=0; else free=$((free + 1)); fi
    sleep 60
done
log "start"
( eval "$(micromamba shell hook -s bash)"; micromamba activate r3d_sim; unset PYTHONPATH
  python -W ignore scripts/sim/sensitivity_yaw.py --stride 1 ) > R3D/data/outputs/sim/mimicgen/yaw_sensitivity/run.log 2>&1
log "exit $?: R3D/data/outputs/sim/mimicgen/yaw_sensitivity/sensitivity_yaw.md"
