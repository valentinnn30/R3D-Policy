#!/usr/bin/env bash
# Round 7 (shortcut hypothesis), strictly sequential, one GPU job at a time. Detached:
#
#   WAIT_PID=<pid> setsid nohup bash scripts/sim/run_round7_chain.sh \
#       > R3D/data/outputs/sim/mimicgen/round7/chain.log 2>&1 &
#
#   0 wait for WAIT_PID (the SPEC06 sweep driver: its training AND evals) to exit
#   1 frame-0 reference sidecar (make_frame0_reference.py) + eval-scene reference check
#   2 checks: check_frozen_robot.py (all), check_link_remover.py
#   3 sensitivity test on the existing 250.ckpts (sensitivity_shortcut.py)
#   4 smoke: 2 short epochs with the freeze on, 1.ckpt, 2-scene eval (exercises the eval path);
#     then the smoke run dir is deleted
#   5 frozen run: uncut 4096/64 seed 42 + task.freeze_robot_labels=[5,7], 251 epochs, 64 x 2,
#     only 200.ckpt and 250.ckpt written
#   6 eval 200 + 250 on the 200 scenes, 5 workers
# Any failure stops the chain (set -e). Steps whose output exists are skipped.
set -euo pipefail
here="$(cd "$(dirname "$0")/../.." && pwd)"
cd "${here}"
out=R3D/data/outputs/sim/mimicgen/round7
mkdir -p "${out}"
log() { echo "[$(date '+%F %T')] $*"; }
env_on() { eval "$(micromamba shell hook -s bash)"; micromamba activate "$1"; unset PYTHONPATH; }
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
sidecar=R3D/data/square_d2-agentview_p4096-200.frame0_p8192.npz
run=R3D/data/outputs/sim/mimicgen_sim-square_d2-p4096-g64-frozen-42
smoke=R3D/data/outputs/sim/mimicgen_sim-square_d2-p4096-g64-frozensmoke-42
freeze="task.freeze_robot_labels=[5,7]"

if [ -n "${WAIT_PID:-}" ]; then
    log "waiting for pid ${WAIT_PID}"
    while kill -0 "${WAIT_PID}" 2>/dev/null; do sleep 60; done
    log "pid ${WAIT_PID} gone"
fi

# ---- 1 sidecar ------------------------------------------------------------------------------
if [ ! -f "${sidecar}" ]; then
    log "frame-0 reference sidecar"
    ( env_on r3d_robosuite; python -W ignore scripts/sim/make_frame0_reference.py --workers 4 ) > "${out}/frame0_reference.log" 2>&1
    tail -2 "${out}/frame0_reference.log"
fi
if [ ! -f "${out}/eval_scene_reference.ok" ]; then
    log "eval-scene reference check (200 scenes)"
    ( env_on r3d_robosuite; python -W ignore scripts/sim/make_frame0_reference.py --eval-scenes 200 --workers 4 ) \
        > "${out}/eval_scene_reference.log" 2>&1
    tail -1 "${out}/eval_scene_reference.log"; touch "${out}/eval_scene_reference.ok"
fi

# ---- 2 checks -------------------------------------------------------------------------------
if [ ! -f "${out}/checks.ok" ]; then
    log "check_frozen_robot.py"
    ( env_on r3d_sim; python -W ignore scripts/sim/check_frozen_robot.py ) > "${out}/check_frozen_robot.log" 2>&1
    grep -E "PASS|FAIL" "${out}/check_frozen_robot.log" | tail -30
    log "check_link_remover.py"
    ( env_on r3d_sim; python -W ignore scripts/sim/check_link_remover.py ) > "${out}/check_link_remover.log" 2>&1
    tail -3 "${out}/check_link_remover.log"
    touch "${out}/checks.ok"
fi

# ---- 3 sensitivity --------------------------------------------------------------------------
if [ ! -f "${out}/sensitivity.ok" ]; then
    log "sensitivity test"
    ( env_on r3d_sim; python -W ignore scripts/sim/sensitivity_shortcut.py --out "${out}/sensitivity.json" ) \
        > "${out}/sensitivity.log" 2>&1
    grep -v "^\s*$" "${out}/sensitivity.log" | tail -9; touch "${out}/sensitivity.ok"
fi

# ---- 4 smoke --------------------------------------------------------------------------------
if [ ! -f "${out}/smoke.ok" ]; then
    log "smoke run"
    rm -rf "${smoke}" "${smoke}.train.log"
    TASK_CFG=mimicgen_sim SETTING=agentview_p4096-200 EPOCHS=2 CHECKPOINT_EVERY=125 BATCH_SIZE=64 GRAD_ACCUM=2 \
        TAG=p4096-g64-frozensmoke bash scripts/train_robotwin2_sim.sh square_d2 42 0 \
        task.num_group=64 task.num_points=4096 "${freeze}" '+training.checkpoint_epochs=[1]' \
        training.max_train_steps=6 training.max_val_steps=3 > "${smoke}.train.log" 2>&1
    ls "${smoke}/checkpoints"
    [ "$(ls "${smoke}/checkpoints")" = "1.ckpt" ] || { log "smoke: expected exactly 1.ckpt"; exit 1; }
    ( env_on r3d_robosuite; python -W ignore scripts/sim/eval_mimicgen_ckpt.py "${smoke}/checkpoints/1.ckpt" \
        --episodes 2 --workers 1 ) > "${out}/smoke_eval.log" 2>&1
    grep "^ckpt:" "${out}/smoke_eval.log"
    grep -q "freeze_robot_labels" "${smoke}/.hydra/config.yaml"
    cp "${smoke}.train.log" "${out}/smoke_train.log"
    rm -rf "${smoke}" "${smoke}.train.log"
    log "smoke ok, smoke run dir deleted"; touch "${out}/smoke.ok"
fi

# ---- 5 train --------------------------------------------------------------------------------
if [ ! -f "${run}/checkpoints/250.ckpt" ]; then
    log "train frozen run"
    TASK_CFG=mimicgen_sim SETTING=agentview_p4096-200 EPOCHS=251 CHECKPOINT_EVERY=125 BATCH_SIZE=64 GRAD_ACCUM=2 \
        TAG=p4096-g64-frozen bash scripts/train_robotwin2_sim.sh square_d2 42 0 \
        task.num_group=64 task.num_points=4096 "${freeze}" '+training.checkpoint_epochs=[200,250]' \
        > "${run}.train.log" 2>&1
    log "train exit $?: $(ls "${run}/checkpoints" | tr '\n' ' ')"
fi

# ---- 6 eval ---------------------------------------------------------------------------------
log "eval 200 + 250"
( env_on r3d_robosuite; python -W ignore scripts/sim/eval_mimicgen_ckpt.py "${run}/checkpoints/200.ckpt" \
    "${run}/checkpoints/250.ckpt" --episodes 200 --workers 5 ) >> "${run}/eval.log" 2>&1
grep "^ckpt:" "${run}/eval.log" | tail -2
log "round 7 chain done"
