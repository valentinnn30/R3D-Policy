#!/usr/bin/env bash
# Threading OOD test (eval only): the 250 checkpoints of uncut / arm cut / GObalAll, trained in the narrowed tripod
# window, evaluated on STOCK Threading D1 (tripod anywhere in D1's range), 200 scenes of seed block 1, then
# scripts/sim/ood_threading_report.py splits the scenes into inside / outside the training window. Detached:
#
#   setsid nohup bash scripts/sim/run_threading_ood.sh > R3D/data/outputs/sim/mimicgen/threading/ood.log 2>&1 &
#
# Waits until no training / eval / chain has run for 10 min; a 2-scene smoke first (its json deleted).
set -euo pipefail
here="$(cd "$(dirname "$0")/../.." && pwd)"
cd "${here}"
log() { echo "[$(date '+%F %T')] $*"; }
env_on() { eval "$(micromamba shell hook -s bash)"; micromamba activate "$1"; unset PYTHONPATH; }
sim=R3D/data/outputs/sim
ckpts=("${sim}/mimicgen_sim-threading_d1n-p4096-g64-42/checkpoints/250.ckpt"
       "${sim}/mimicgen_sim-threading_d1n-p4096-g64-cut5-42/checkpoints/250.ckpt"
       "${sim}/mimicgen_sim_remover-threading_d1n-p4096-g64-rm-GObalAll-42/checkpoints/250.ckpt")

busy() { ps -eo cmd | awk '($1=="bash" && $2 ~ /run_threading_chain|run_remover_sweep|run_cut_pair|run_round[0-9]+_chain/) || ($1 ~ /python/ && ($0 ~ /train\.py/ || $0 ~ /eval_mimicgen_ckpt/))' | grep -q .; }
log "waiting for the GPU to be free for 10 min"
free=0
while [ "${free}" -lt 10 ]; do
    if busy; then free=0; else free=$((free + 1)); fi
    sleep 60
done
log "GPU free for 10 min"
for c in "${ckpts[@]}"; do [ -f "${c}" ] || { log "missing ${c}"; exit 1; }; done

log "smoke (2 scenes)"
( env_on r3d_robosuite; python -W ignore scripts/sim/eval_mimicgen_ckpt.py "${ckpts[0]}" --episodes 2 --workers 1 \
    --env-name Threading_D1 ) > "${sim}/mimicgen/threading/ood_smoke.log" 2>&1
grep "^ckpt:" "${sim}/mimicgen/threading/ood_smoke.log"
rm -f "${sim}/mimicgen_sim-threading_d1n-p4096-g64-42/eval/250_2ep_block1_env-Threading_D1.json"

for c in "${ckpts[@]}"; do
    log "eval $(basename "$(dirname "$(dirname "${c}")")") on Threading_D1"
    ( env_on r3d_robosuite; python -W ignore scripts/sim/eval_mimicgen_ckpt.py "${c}" --episodes 200 --workers 5 \
        --env-name Threading_D1 ) >> "$(dirname "$(dirname "${c}")")/eval.log" 2>&1
    log "$(grep '^ckpt:' "$(dirname "$(dirname "${c}")")/eval.log" | tail -1 | sed 's|.*checkpoints/||; s|  [0-9.]* min.*||')"
done
( env_on r3d_sim; python scripts/sim/ood_threading_report.py ) | tee "${sim}/mimicgen/threading/ood_report.txt"
log "threading OOD done"
