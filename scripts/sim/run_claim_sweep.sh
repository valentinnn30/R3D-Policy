#!/usr/bin/env bash
# Claim sweep (plan: R3D/data/outputs/sim/mimicgen/claim_sweep/claim_sweep_plan.md, user 2026-11-12). Detached:
#
#   setsid nohup bash scripts/sim/run_claim_sweep.sh > R3D/data/outputs/sim/mimicgen/claim_sweep/chain.log 2>&1 &
#
#   0 wait until the token-dropout, table-eval and Threading-1024 chains have exited and the GPU was free 10 min
#   1 checks (must pass): check_prefix_thin.py (dataset + --eval), mechanism table + armsparse GATE
#   2 smoke per new cloud (armsparse, scenesparse): 10 epochs x 40 batches, 64 x 2, no checkpoint; folders deleted
#   3 queue in the plan's order (section 5), one run at a time through the existing drivers:
#       E1 uncut s42 (+ hanging_mug uncut 600) | R1 uncut s43 | R2 armsparse s42 | R3 scenesparse s42 | R1 uncut s44 |
#       R2 armsparse s43 | R3 scenesparse s43 | R4 DROP-1024 s42, s43 | R5 DROP s44 | E1 uncut s43, s44
#     plain runs: run_mimicgen_queue.sh (CKPT_EPOCHS=[200,250]); DROP runs: run_remover_sweep.sh (DROP-43's overrides,
#     ckpts 200 + 250, no intermediate evals). Pass 1 only: 250 is evaluated, 200 kept for later evals on request.
#   4 report: claim_sweep_report.py -> claim_sweep/report.md (rerun any time)
# Disk: before every training, if < 8 GB are free, the 200.ckpt of FINISHED sweep runs are deleted (oldest first;
# never a 250.ckpt, never outside this sweep); if still < 5 GB the chain stops. Steps whose output exists are
# skipped, so the chain can be restarted. Any failed check stops it.
set -uo pipefail
here="$(cd "$(dirname "$0")/../.." && pwd)"
cd "${here}"
out=R3D/data/outputs/sim/mimicgen/claim_sweep
S=R3D/data/outputs/sim
mkdir -p "${out}"
log() { echo "[$(date '+%F %T')] $*"; }
env_on() { eval "$(micromamba shell hook -s bash)"; micromamba activate "$1"; unset PYTHONPATH; }
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
P=mimicgen_sim-square_d2-p4096-g64
DROP_OV="policy.link_remover.learn_density=false policy.link_remover.init_keep=0.5 policy.link_remover.theta_keep=0.5 training.link_remover.lr=0 training.link_remover.phi_every=1000000000 training.link_remover.finetune_frac=0 +training.checkpoint_epochs=[200,250]"

busy() { ps -eo cmd | awk '($1=="bash" && $2 ~ /run_round[0-9]+_chain|run_remover_sweep|run_mimicgen_queue|run_finetune_branches|run_mask_override|run_branch_test|run_early_branch|run_cut_keep7|run_cut_pair|run_mug_drop|run_threading|run_token_dropout|run_table_eval|run_thr_p1024/) || ($1 ~ /python/ && ($0 ~ /train\.py/ || $0 ~ /eval_mimicgen_ckpt/ || $0 ~ /eval_robotwin2_ckpt/))' | grep -q .; }
log "waiting for the GPU to be free for 10 min (after the token-dropout, table-eval and Threading-1024 chains)"
free=0
while [ "${free}" -lt 10 ]; do
    if busy; then free=0; else free=$((free + 1)); fi
    sleep 60
done
log "GPU free for 10 min; disk free $(df -h --output=avail R3D/data | tail -1)"

# ---- 1 checks ---------------------------------------------------------------------------------------------
if [ ! -f "${out}/checks.ok" ]; then
    log "checks"
    ( env_on r3d_sim; python -W ignore scripts/sim/check_prefix_thin.py ) > "${out}/check_prefix_thin.log" 2>&1
    grep -q "^ALL PASS" "${out}/check_prefix_thin.log" || { log "dataset check FAILED (see check_prefix_thin.log): stopping"; exit 1; }
    ( env_on r3d_robosuite; python -W ignore scripts/sim/check_prefix_thin.py --eval ) > "${out}/check_prefix_thin_eval.log" 2>&1
    grep -q "^ALL PASS" "${out}/check_prefix_thin_eval.log" || { log "eval check FAILED: stopping"; exit 1; }
    ( env_on r3d_sim; python -W ignore scripts/sim/claim_sweep_mechanism.py --stride 20 ) > "${out}/mechanism.log" 2>&1 \
        || { log "mechanism GATE FAILED (armsparse moves tokens): stopping, ask the user"; tail -3 "${out}/mechanism.log"; exit 1; }
    tail -1 "${out}/mechanism.log"
    touch "${out}/checks.ok"
fi

# ---- 2 smoke ------------------------------------------------------------------------------------------------
for rule in arm:armsparse non_arm:scenesparse; do
    set_=${rule%%:*}; tag=${rule##*:}
    [ -f "${out}/smoke_${tag}.ok" ] && continue
    sm=${S}/mimicgen_sim-square_d2-smoke-${tag}-42
    log "smoke ${tag}"
    TASK_CFG=mimicgen_sim SETTING=agentview_p4096-200 EPOCHS=10 CHECKPOINT_EVERY=125 BATCH_SIZE=64 GRAD_ACCUM=2 \
        TAG="smoke-${tag}" bash scripts/train_robotwin2_sim.sh square_d2 42 0 task.num_group=64 task.num_points=4096 \
        "task.thin_point_set=${set_}" '+training.checkpoint_epochs=[]' training.max_train_steps=40 \
        > "${out}/smoke_${tag}.log" 2>&1
    rc=$?
    rm -rf "${sm}"
    [ ${rc} -eq 0 ] || { log "smoke ${tag} FAILED (exit ${rc}, see smoke_${tag}.log): stopping"; exit 1; }
    touch "${out}/smoke_${tag}.ok"
done

# ---- helpers ----------------------------------------------------------------------------------------------
SWEEP_RUNS=("${P}-43" "${P}-armsparse-42" "${P}-scenesparse-42" "${P}-44" "${P}-armsparse-43" "${P}-scenesparse-43"
            "mimicgen_sim_remover-square_d2-p1024-g64-rm-DROP-42" "mimicgen_sim_remover-square_d2-p1024-g64-rm-DROP-43"
            "mimicgen_sim_remover-square_d2-p4096-g64-rm-DROP-44")
free_gb() { df --output=avail -BG R3D/data | tail -1 | tr -dc 0-9; }
disk_guard() {
    [ "$(free_gb)" -ge 8 ] && return 0
    for r in "${SWEEP_RUNS[@]}"; do
        [ "$(free_gb)" -ge 8 ] && break
        f=${S}/${r}/checkpoints/200.ckpt
        [ -f "${f}" ] && [ -f "${S}/${r}/eval/250_200ep_block1.json" ] || continue
        log "disk $(free_gb) GB: delete ${r}/checkpoints/200.ckpt (run finished, 250 kept)"; rm -f "${f}"
    done
    [ "$(free_gb)" -ge 5 ] || { log "only $(free_gb) GB free after deleting the sweep's 200.ckpt: stopping, ask the user"; exit 1; }
}
plain() {  # queue line
    printf '%s\n' "$1" > "${out}/plain_step.txt"
    disk_guard
    CKPT_EPOCHS='[200,250]' bash scripts/sim/run_mimicgen_queue.sh "${out}/plain_step.txt" >> "${out}/plain_driver.log" 2>&1
    tail -2 "${out}/plain_driver.log"
}
remover() {  # queue line, report tag
    printf '%s\n' "$1" > "${out}/remover_step.txt"
    disk_guard
    MID_CKPTS="" KEEP_ONLY_EVALUATED=0 SWEEP_OUT="${out}/remover_$2" \
        bash scripts/sim/run_remover_sweep.sh "${out}/remover_step.txt" >> "${out}/remover_driver.log" 2>&1
    tail -2 "${out}/remover_driver.log"
}
e1() {  # seed: uncut 250.ckpt with the arm cut at inference only
    local run=${S}/${P}-$1
    [ -f "${run}/eval/250_200ep_block1_evalcut-5.json" ] && return 0
    [ -f "${run}/checkpoints/250.ckpt" ] || { log "E1 s$1: no 250.ckpt, skipped"; return 0; }
    log "E1 uncut s$1: arm cut [5] at inference"
    ( env_on r3d_robosuite; python -W ignore scripts/sim/eval_mimicgen_ckpt.py "${run}/checkpoints/250.ckpt" \
        --episodes 200 --workers 5 --cut-labels 5 ) >> "${run}/eval_250_evalcut-5.log" 2>&1
    log "  $(grep '^ckpt:' "${run}/eval_250_evalcut-5.log" | tail -1 | sed 's|.*success|success|; s| -> .*||')"
}
e1_mug() {
    local run=${S}/robotwin2_sim-hanging_mug_labeled-p4096-g64-42
    [ -f "${run}/checkpoints/600.eval_200ep_seed1_shared_cut-45.json" ] && return 0
    log "E1 hanging_mug uncut 600: arms [4,5] cut at inference (shared list)"
    ( env_on r3d_sim; python -W ignore scripts/eval_robotwin2_ckpt.py "${run}/checkpoints/600.ckpt" --episodes 200 --seed 1 \
        --tag shared --drop-point-labels 4,5 ) > "${run}/eval_600_200ep_seed1_shared_cut-45.log" 2>&1
    log "  success $(python3 -c "import json;print(json.load(open('${run}/checkpoints/600.eval_200ep_seed1_shared_cut-45.json'))['success_rate'])" 2>/dev/null)"
}

# ---- 3 queue (plan section 5) -------------------------------------------------------------------------------
e1 42; e1_mug
plain "4096 64 43 -"
plain "4096 64 42 - armsparse task.thin_point_set=arm"
plain "4096 64 42 - scenesparse task.thin_point_set=non_arm"
plain "4096 64 44 -"
plain "4096 64 43 - armsparse task.thin_point_set=arm"
plain "4096 64 43 - scenesparse task.thin_point_set=non_arm"
remover "1024 42 DROP ${DROP_OV}" DROP1024-42
remover "1024 43 DROP ${DROP_OV}" DROP1024-43
remover "4096 44 DROP ${DROP_OV}" DROP-44
e1 43; e1 44

# ---- 4 report -------------------------------------------------------------------------------------------------
python3 scripts/sim/claim_sweep_report.py > "${out}/report.log" 2>&1 && log "report: ${out}/report.md" || log "report failed (report.log)"
log "claim sweep done"
