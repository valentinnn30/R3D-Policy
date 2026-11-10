#!/usr/bin/env bash
# Arm cut minus link7, built EXACTLY like the arm cut run (mimicgen_sim-square_d2-p4096-g64-cut5-42): plain DP3,
# the dataset's oracle cut after augmentation, same overrides -- only `task.drop_point_labels=[5]` is replaced by
# `task.drop_point_links=[link0..link6]` (the same cut_points_by_label call, keyed on data/point_link; link0..7
# reproduces the label-5 cut bit for bit, scripts/sim/check_link_cut.py). Strictly one GPU job at a time:
#
#   setsid nohup bash scripts/sim/run_cut_keep7.sh > R3D/data/outputs/sim/mimicgen/cut_keep7/chain.log 2>&1 &
#
#   0 wait until no training / eval / sweep driver has run for 10 min
#   1 checks: check_link_cut.py (dataset + --eval), check_frozen_robot.py --off-only
#   2 smoke: 2 short epochs, 1.ckpt, 2-scene eval (the eval path), then the smoke dir is deleted
#   3 train: seed 42, 251 epochs, 64 x 2, only 200.ckpt and 250.ckpt written
#   4 eval 250, then 200, on the 200 scenes (5 workers)
# Any failure stops the chain (set -e). Steps whose output exists are skipped.
set -euo pipefail
here="$(cd "$(dirname "$0")/../.." && pwd)"
cd "${here}"
out=R3D/data/outputs/sim/mimicgen/cut_keep7
mkdir -p "${out}"
log() { echo "[$(date '+%F %T')] $*"; }
env_on() { eval "$(micromamba shell hook -s bash)"; micromamba activate "$1"; unset PYTHONPATH; }
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
run=R3D/data/outputs/sim/mimicgen_sim-square_d2-p4096-g64-cutL06-42
smoke=R3D/data/outputs/sim/mimicgen_sim-square_d2-p4096-g64-cutL06smoke-42
cut="task.drop_point_links=[link0,link1,link2,link3,link4,link5,link6]"

busy() { ps -eo cmd | awk -v me="$$" '($1=="bash" && $2 ~ /run_round[0-9]+_chain|run_remover_sweep|run_finetune_branches|run_mask_override|run_branch_test|run_early_branch/) || ($1 ~ /python/ && ($0 ~ /train\.py/ || $0 ~ /eval_mimicgen_ckpt/))' | grep -q .; }
log "waiting for the GPU to be free for 10 min"
free=0
while [ "${free}" -lt 10 ]; do
    if busy; then free=0; else free=$((free + 1)); fi
    sleep 60
done
log "GPU free for 10 min"

# ---- 1 checks -------------------------------------------------------------------------------
if [ ! -f "${out}/checks.ok" ]; then
    log "checks"
    ( env_on r3d_sim; python -W ignore scripts/sim/check_link_cut.py ) > "${out}/check_link_cut.log" 2>&1
    tail -1 "${out}/check_link_cut.log"
    ( env_on r3d_robosuite; python -W ignore scripts/sim/check_link_cut.py --eval ) > "${out}/check_link_cut_eval.log" 2>&1 || true
    grep -q "^ALL PASS" "${out}/check_link_cut_eval.log" || { log "eval-side check failed"; exit 1; }
    ( env_on r3d_sim; python -W ignore scripts/sim/check_frozen_robot.py --off-only ) > "${out}/check_frozen_off.log" 2>&1
    tail -1 "${out}/check_frozen_off.log"
    touch "${out}/checks.ok"
fi

# ---- 2 smoke --------------------------------------------------------------------------------
if [ ! -f "${out}/smoke.ok" ]; then
    log "smoke run"
    rm -rf "${smoke}" "${smoke}.train.log"
    TASK_CFG=mimicgen_sim SETTING=agentview_p4096-200 EPOCHS=2 CHECKPOINT_EVERY=125 BATCH_SIZE=64 GRAD_ACCUM=2 \
        TAG=p4096-g64-cutL06smoke bash scripts/train_robotwin2_sim.sh square_d2 42 0 \
        task.num_group=64 task.num_points=4096 "${cut}" '+training.checkpoint_epochs=[1]' \
        training.max_train_steps=6 training.max_val_steps=3 > "${smoke}.train.log" 2>&1
    [ "$(ls "${smoke}/checkpoints")" = "1.ckpt" ] || { log "smoke: expected exactly 1.ckpt"; exit 1; }
    ( env_on r3d_robosuite; python -W ignore scripts/sim/eval_mimicgen_ckpt.py "${smoke}/checkpoints/1.ckpt" \
        --episodes 2 --workers 1 ) > "${out}/smoke_eval.log" 2>&1
    grep "^ckpt:" "${out}/smoke_eval.log"
    grep -q "drop_point_links" "${smoke}/.hydra/config.yaml"
    cp "${smoke}.train.log" "${out}/smoke_train.log"
    rm -rf "${smoke}" "${smoke}.train.log"
    log "smoke ok, smoke run dir deleted"; touch "${out}/smoke.ok"
fi

# ---- 3 train --------------------------------------------------------------------------------
if [ ! -f "${run}/checkpoints/250.ckpt" ]; then
    log "train cutL06 (arm cut, link7 kept)"
    TASK_CFG=mimicgen_sim SETTING=agentview_p4096-200 EPOCHS=251 CHECKPOINT_EVERY=125 BATCH_SIZE=64 GRAD_ACCUM=2 \
        TAG=p4096-g64-cutL06 bash scripts/train_robotwin2_sim.sh square_d2 42 0 \
        task.num_group=64 task.num_points=4096 "${cut}" '+training.checkpoint_epochs=[200,250]' \
        > "${run}.train.log" 2>&1
    log "train exit $?: $(ls "${run}/checkpoints" | tr '\n' ' ')"
fi

# ---- 4 eval ---------------------------------------------------------------------------------
for e in 250 200; do
    [ -f "${run}/eval/${e}_200ep_block1.json" ] && continue
    log "eval ${e}"
    ( env_on r3d_robosuite; python -W ignore scripts/sim/eval_mimicgen_ckpt.py "${run}/checkpoints/${e}.ckpt" \
        --episodes 200 --workers 5 ) >> "${run}/eval.log" 2>&1
    log "eval ${e}: $(grep '^ckpt:' "${run}/eval.log" | tail -1 | sed 's|.*checkpoints/||; s|  [0-9.]* min.*||')"
done
log "cut_keep7 chain done"
