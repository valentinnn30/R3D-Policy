#!/usr/bin/env bash
# Remaining runs after the user's restructure (2026-11-14), replacing the rest of run_claim_sweep.sh and
# run_token_budget.sh. Dropped by the user: uncut s44, armsparse / scenesparse s43, DROP-1024 s43, DROP s44, E1 s44
# (rule: one seed for side checks, two for main results). Detached:
#
#   setsid nohup bash scripts/sim/run_final_queue.sh > R3D/data/outputs/sim/mimicgen/final_queue.log 2>&1 &
#
#   0 wait until the running scenesparse s42 driver (claim sweep) has finished and the GPU was free 10 min
#   1 DROP at 1024 points, seed 42 (DROP-43's overrides; ckpts 200 + 250; default eval = the uncut cloud)
#   2 E1: uncut s43 250.ckpt with the arm cut at inference only
#   3 claim report (claim_sweep/report.md)
#   4 token block, queue token_budget/queue.txt (re-read; remove lines there to skip): 4096/128 uncut, 4096/128 cut,
#     4096/256 uncut, 2048/128 cut, 4096/256 cut -- seed 42, ckpt 250 only
#   5 token report (token_budget/report.txt) and the claim report again
# Steps whose output exists are skipped, so it can be restarted.
set -uo pipefail
here="$(cd "$(dirname "$0")/../.." && pwd)"
cd "${here}"
S=R3D/data/outputs/sim
cs=${S}/mimicgen/claim_sweep
tb=${S}/mimicgen/token_budget
log() { echo "[$(date '+%F %T')] $*"; }
env_on() { eval "$(micromamba shell hook -s bash)"; micromamba activate "$1"; unset PYTHONPATH; }
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
P=mimicgen_sim-square_d2-p4096-g64
DROP_OV="policy.link_remover.learn_density=false policy.link_remover.init_keep=0.5 policy.link_remover.theta_keep=0.5 training.link_remover.lr=0 training.link_remover.phi_every=1000000000 training.link_remover.finetune_frac=0 +training.checkpoint_epochs=[200,250]"

busy() { ps -eo cmd | awk '($1=="bash" && $2 ~ /run_claim_sweep|run_token_budget|run_remover_sweep|run_mimicgen_queue|run_round[0-9]+_chain|run_table_eval|run_thr_p1024|run_threading/) || ($1 ~ /python/ && ($0 ~ /train\.py/ || $0 ~ /eval_mimicgen_ckpt/ || $0 ~ /eval_robotwin2_ckpt/))' | grep -q .; }
log "waiting for scenesparse s42 (claim sweep driver) to finish and the GPU to be free for 10 min"
free=0
while [ "${free}" -lt 10 ]; do
    if busy; then free=0; else free=$((free + 1)); fi
    sleep 60
done
log "GPU free; disk free $(df -h --output=avail R3D/data | tail -1)"
log "scenesparse s42: $(grep 'scenesparse-42/eval' ${cs}/plain_driver.log | tail -1 | sed 's|.*success|success|; s| -> .*||')"

free_gb() { df --output=avail -BG R3D/data | tail -1 | tr -dc 0-9; }
[ "$(free_gb)" -ge 10 ] || { log "only $(free_gb) GB free: stopping, ask the user"; exit 1; }

# ---- 1 DROP at 1024, seed 42 ---------------------------------------------------------------------------------
printf '%s\n' "1024 42 DROP ${DROP_OV}" > "${cs}/remover_step.txt"
MID_CKPTS="" KEEP_ONLY_EVALUATED=0 SWEEP_OUT="${cs}/remover_DROP1024-42" \
    bash scripts/sim/run_remover_sweep.sh "${cs}/remover_step.txt" >> "${cs}/remover_driver.log" 2>&1
log "DROP-1024 s42: $(grep 'eval exit' "${cs}/remover_driver.log" | tail -1 | sed 's|.*eval exit||')"

# ---- 2 E1 uncut s43 -------------------------------------------------------------------------------------------
run=${S}/${P}-43
if [ ! -f "${run}/eval/250_200ep_block1_evalcut-5.json" ] && [ -f "${run}/checkpoints/250.ckpt" ]; then
    log "E1 uncut s43: arm cut [5] at inference"
    ( env_on r3d_robosuite; python -W ignore scripts/sim/eval_mimicgen_ckpt.py "${run}/checkpoints/250.ckpt" \
        --episodes 200 --workers 5 --cut-labels 5 ) >> "${run}/eval_250_evalcut-5.log" 2>&1
    log "  $(grep '^ckpt:' "${run}/eval_250_evalcut-5.log" | tail -1 | sed 's|.*success|success|; s| -> .*||')"
fi

# ---- 3 claim report -------------------------------------------------------------------------------------------
python3 scripts/sim/claim_sweep_report.py > "${cs}/report.log" 2>&1 && log "claim report: ${cs}/report.md"

# ---- 4 token block --------------------------------------------------------------------------------------------
[ "$(free_gb)" -ge 8 ] || { log "only $(free_gb) GB free before the token block: stopping, ask the user"; exit 1; }
CKPT_EPOCHS='[250]' bash scripts/sim/run_mimicgen_queue.sh "${tb}/queue.txt" >> "${tb}/driver.log" 2>&1
grep -E "eval exit" "${tb}/driver.log" | tail -5

# ---- 5 reports ------------------------------------------------------------------------------------------------
python3 scripts/sim/token_budget_report.py > /dev/null && log "token report: ${tb}/report.txt"
python3 scripts/sim/claim_sweep_report.py > "${cs}/report.log" 2>&1
log "final queue done"
