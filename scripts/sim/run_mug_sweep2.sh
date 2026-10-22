#!/usr/bin/env bash
# hanging_mug mechanism sweep at 1024 points (decided 2026-10-21). Same recipe as
# run_mug_sweep.sh (seed 42, 601 epochs, 64 x 4, ckpts at 300/600), but the
# epoch-300 50-scene check is skipped (all runs were still improving 300 -> 600,
# so it adds little). 600.ckpt evaluated on the SAME 200 scenes (seed block 1).
# The 1024/64 no-cut baseline exists from run_mug_sweep.sh (61.0%).
#   1  1024 / 64  arms cut   coverage already ~100%: is it arm content or coverage?
#   2  1024 / 32  no cut     coverage 81% reference
#   3  1024 / 32  arms cut
#   4  1024 / 16  no cut     handle 0.4 tokens, 44% coverage
#   5  1024 / 16  arms cut
set -uo pipefail
here="$(cd "$(dirname "$0")/../.." && pwd)"
sim="${here}/R3D/data/outputs/sim"
task=hanging_mug_labeled
seed=42
zarr="R3D/data/${task}-sim_stock_p4096-50.zarr"
log() { echo "[$(date '+%F %T')] $*"; }
eval "$(micromamba shell hook -s bash)"
micromamba activate r3d_sim
unset PYTHONPATH
run_dir() { echo "${sim}/robotwin2_sim-${task}-$1-${seed}"; }
train() {  # tag overrides...
    local tag=$1; shift
    if [ -f "$(run_dir "${tag}")/checkpoints/600.ckpt" ]; then log "${tag}: 600.ckpt exists, skip"; return; fi
    log "train ${tag} ($*)"
    TAG="${tag}" SETTING=sim_stock_p1024 EPOCHS=601 CHECKPOINT_EVERY=300 BATCH_SIZE=64 GRAD_ACCUM=4 \
        bash "${here}/scripts/train_robotwin2_sim.sh" "${task}" "${seed}" 0 \
        task.num_points=1024 "task.dataset.zarr_path=${zarr}" "$@" > "${sim}/train_${task}_${tag}.log" 2>&1
    log "train ${tag} exit $?"
}
evaluate() {  # tag
    local ck; ck="$(run_dir "$1")/checkpoints"
    [ -f "${ck}/600.ckpt" ] || { log "missing ${ck}/600.ckpt"; return; }
    [ -f "${ck}/600.eval_200ep_seed1.json" ] && return
    python -W ignore "${here}/scripts/eval_robotwin2_ckpt.py" "${ck}/600.ckpt" --episodes 200 --seed 1 \
        > "$(dirname "${ck}")/eval_600_200ep_seed1.log" 2>&1
    log "RESULT $1: epoch 600 / 200 scenes $(python -c "import json;print(json.load(open('${ck}/600.eval_200ep_seed1.json'))['success_rate'])" 2>/dev/null)"
}
for spec in "p1024-g64-cut45 64 [4,5]" "p1024-g32 32 -" "p1024-g32-cut45 32 [4,5]" "p1024-g16 16 -" "p1024-g16-cut45 16 [4,5]"; do
    set -- ${spec}
    if [ "$3" = "-" ]; then train "$1" task.num_group=$2; else train "$1" task.num_group=$2 "task.drop_point_labels=$3"; fi
    evaluate "$1"
done
log "done"
