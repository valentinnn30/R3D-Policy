#!/usr/bin/env bash
# Density convergence + density specialist (user, 2026-11-19). Detached:
#
#   setsid nohup bash scripts/sim/run_density_long.sh \
#       > R3D/data/outputs/sim/mimicgen/density_long/chain.log 2>&1 < /dev/null &
#
# 1 DOlong   DOscene (round 2: density only, 8 robot regions + label-free scene region, scene
#            floor 0.1, NOT the decoupled DObal) re-run with a longer phi phase so every
#            region's density can settle, and init rho 0.5 instead of 0.9.
#            651 epochs = warm-up 50 | phi 601 (DOscene: 175), NO fine-tune (user: 25 epochs at the end
#            of the cosine schedule do nothing) -> 650.ckpt only; eval = EMA theta + phi's final rho.
# 2 convergence of DOlong's density per region (scripts/sim/density_convergence.py, last 100
#   phi epochs) -> density_long/convergence.md; its final deterministic rho = the mask.
# 3 SPECrho  the density SPECIALIST: fixed_rho = DOlong's final rho, from scratch, nothing
#            learned, standard budget (251 epochs, 64 x 2, seed 42), 200.ckpt + 250.ckpt.
# 4 evals on the 200 fixed scenes (block 1): SPECrho 250 -> DOlong 650 -> SPECrho 200, then
#   analyze_remover_run.py on both.
# Steps whose output exists are skipped, so the chain can be restarted.
set -uo pipefail
here="$(cd "$(dirname "$0")/../.." && pwd)"
out="${here}/R3D/data/outputs/sim/mimicgen/density_long"
mkdir -p "${out}"
sim="${here}/R3D/data/outputs/sim"
long_run="${sim}/mimicgen_sim_remover-square_d2-p4096-g64-rm-DOlong-42"
spec_run="${sim}/mimicgen_sim_remover-square_d2-p4096-g64-rm-SPECrho-42"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
log() { echo "[$(date '+%F %T')] $*"; }

train() {  # tag epochs ckpt_list overrides...
    local tag=$1 epochs=$2 ckpts=$3; shift 3
    log "train ${tag} (${epochs} epochs, ckpts ${ckpts}) $*"
    ALG=r3d_robotwin2_remover TASK_CFG=mimicgen_sim_remover SETTING=agentview_p4096-200 EPOCHS="${epochs}" \
        CHECKPOINT_EVERY=100000 BATCH_SIZE=64 GRAD_ACCUM=2 TAG="p4096-g64-rm-${tag}" \
        bash "${here}/scripts/train_robotwin2_sim.sh" square_d2 42 0 \
        task.num_points=4096 task.num_group=64 "+training.checkpoint_epochs=${ckpts}" "$@" \
        > "${sim}/mimicgen_sim_remover-square_d2-p4096-g64-rm-${tag}-42.train.log" 2>&1 < /dev/null
    log "train ${tag} exit $?"
}

evaluate() {  # run epoch
    local run=$1 e=$2
    [ -f "${run}/eval/${e}_200ep_block1.json" ] && return 0
    [ -f "${run}/checkpoints/${e}.ckpt" ] || { log "no ${run}/checkpoints/${e}.ckpt"; return 1; }
    log "eval $(basename "${run}") ${e}.ckpt (200 scenes)"
    (
        eval "$(micromamba shell hook -s bash)"; micromamba activate r3d_robosuite; unset PYTHONPATH
        python -W ignore "${here}/scripts/sim/eval_mimicgen_ckpt.py" "${run}/checkpoints/${e}.ckpt" --episodes 200 \
            --workers 5 >> "${run}/eval.log" 2>&1 < /dev/null
    )
    log "eval exit $?: $(grep '^ckpt:' "${run}/eval.log" | tail -1 | sed 's|.*checkpoints/||; s|  [0-9.]* min.*||')"
}

in_sim() {  # python in r3d_sim
    ( eval "$(micromamba shell hook -s bash)"; micromamba activate r3d_sim; unset PYTHONPATH; python -W ignore "$@" )
}

# ---- 1 the long density run ---------------------------------------------------------------------
# warmup_frac 0.0768 -> round(0.0768 * 651) = 50; finetune_frac 0 -> phi phase 50-650, no fine-tune
if [ ! -f "${long_run}/checkpoints/650.ckpt" ]; then
    train DOlong 651 "[650]" \
        policy.link_remover.learn_gates=false policy.link_remover.init_rho=0.5 \
        policy.link_remover.region_groups=default+scene \
        training.link_remover.warmup_frac=0.0768 training.link_remover.finetune_frac=0.0
    [ -f "${long_run}/checkpoints/650.ckpt" ] || { log "DOlong wrote no 650.ckpt -- stop"; exit 1; }
fi

# ---- 2 convergence + the mask -------------------------------------------------------------------
in_sim "${here}/scripts/sim/density_convergence.py" "${long_run}" --window 100 --out "${out}" > "${out}/convergence.log" 2>&1
fixed=$(grep '^FIXED_RHO=' "${out}/convergence.log" | sed 's/^FIXED_RHO=//')
[ -n "${fixed}" ] || { log "no FIXED_RHO from density_convergence.py -- stop"; exit 1; }
log "convergence: $(grep 'Run converged' "${out}/convergence.md")  mask ${fixed}"

# ---- 3 the density specialist -------------------------------------------------------------------
if [ ! -f "${spec_run}/checkpoints/250.ckpt" ]; then
    train SPECrho 251 "[200,250]" \
        policy.link_remover.learn_gates=false policy.link_remover.region_groups=default+scene \
        "policy.link_remover.fixed_rho=${fixed}"
    [ -f "${spec_run}/checkpoints/250.ckpt" ] || { log "SPECrho wrote no 250.ckpt -- stop"; exit 1; }
fi

# ---- 4 evals + analyses -------------------------------------------------------------------------
evaluate "${spec_run}" 250
evaluate "${long_run}" 650
evaluate "${spec_run}" 200
in_sim "${here}/scripts/sim/analyze_remover_run.py" "${long_run}" "${spec_run}" >> "${out}/analysis.log" 2>&1 \
    || log "analysis failed (see ${out}/analysis.log)"
log "chain done"
