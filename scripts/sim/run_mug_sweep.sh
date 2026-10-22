#!/usr/bin/env bash
# hanging_mug sweep (CLAUDE.md "Sim baseline", decided 2026-10-19). One run at a
# time (two sharing the GPU are slower than sequential). Every run: seed 42,
# 601 epochs, batch 64 x 4, checkpoints ONLY at 300 and 600 (disk: 2.15 GB each).
# Every run is evaluated: 600.ckpt on 200 scenes (seed block 1 = 200000..) and
# 300.ckpt on 50 scenes (seed block 0) as a convergence check.
#
#   stage A  4096 points, 128 tokens          no cut
#   stage A  4096 points,  64 tokens          no cut
#   rule     G = the token count (64 or 128) whose success (epoch 600, 200 scenes) is in
#            [0.30, 0.40]; both -> closest to 0.35; neither -> closest to 0.35 overall
#            (flagged in_range=false). Ties -> 64.
#   stage B  4096 points,   G tokens          arms cut            (drop [4,5])
#   stage B  4096 points,   G tokens          arms+grippers cut   (drop [4,5,6,7])
#   stage C  2048 points,   G tokens          no cut   (patch-size ablation)
#   stage C  1024 points,   G tokens          no cut
#
# The 2048/1024 runs train on the FIRST N points of the 4096-point zarr (= the
# FPS-N cloud, FPS is greedy) and evaluate with sim_stock_p2048 / sim_stock_p1024.
#
#   setsid nohup bash scripts/sim/run_mug_sweep.sh > R3D/data/outputs/sim/mug_sweep_driver.log 2>&1 < /dev/null &
set -uo pipefail
here="$(cd "$(dirname "$0")/../.." && pwd)"
sim="${here}/R3D/data/outputs/sim"
task=hanging_mug_labeled
seed=42
zarr="R3D/data/${task}-sim_stock_p4096-50.zarr"
decision="${sim}/mug_sweep_decision.json"
log() { echo "[$(date '+%F %T')] $*"; }

eval "$(micromamba shell hook -s bash)"
micromamba activate r3d_sim
unset PYTHONPATH

run_dir() { echo "${sim}/robotwin2_sim-${task}-$1-${seed}"; }

train() {  # tag setting overrides...
    local tag=$1 setting=$2; shift 2
    if [ -f "$(run_dir "${tag}")/checkpoints/600.ckpt" ]; then log "${tag}: 600.ckpt exists, skip training"; return; fi
    log "train ${tag} (setting ${setting}; $*)"
    TAG="${tag}" SETTING="${setting}" EPOCHS=601 CHECKPOINT_EVERY=300 BATCH_SIZE=64 GRAD_ACCUM=4 \
        bash "${here}/scripts/train_robotwin2_sim.sh" "${task}" "${seed}" 0 "$@" \
        > "${sim}/train_${task}_${tag}.log" 2>&1
    log "train ${tag} exit $?"
}

evaluate() {  # tag
    local ck; ck="$(run_dir "$1")/checkpoints"
    for spec in "600 200 1" "300 50 0"; do
        set -- ${spec}
        [ -f "${ck}/$1.ckpt" ] || { log "missing ${ck}/$1.ckpt"; continue; }
        [ -f "${ck}/$1.eval_$2ep_seed$3.json" ] && continue
        python -W ignore "${here}/scripts/eval_robotwin2_ckpt.py" "${ck}/$1.ckpt" --episodes "$2" --seed "$3" \
            > "$(dirname "${ck}")/eval_$1_$2ep_seed$3.log" 2>&1
        log "eval ${ck##*/sim/} epoch $1 on $2 scenes: $(python -c "import json;print(json.load(open('${ck}/$1.eval_$2ep_seed$3.json'))['success_rate'])" 2>/dev/null)"
    done
}

success() {  # tag -> 600-epoch success on 200 scenes
    python -c "import json;print(json.load(open('$(run_dir "$1")/checkpoints/600.eval_200ep_seed1.json'))['success_rate'])"
}

# ---- stage A: baselines at 4096 points
for g in 128 64; do
    train "p4096-g${g}" sim_stock_p4096 task.num_points=4096 task.num_group=${g}
    evaluate "p4096-g${g}"
done

# ---- rule: token count for stages B and C
s64=$(success p4096-g64 2> /dev/null || echo "nan"); s128=$(success p4096-g128 2> /dev/null || echo "nan")
G=$(python "${here}/scripts/sim/pick_token_count.py" "${s64}" "${s128}" "${decision}")
log "decision: 4096/128 ${s128}, 4096/64 ${s64} -> cut and ablation runs at ${G} tokens"

# ---- stage B: oracle cuts at 4096 points
train "p4096-g${G}-cut45"   sim_stock_p4096 task.num_points=4096 task.num_group=${G} "task.drop_point_labels=[4,5]"
evaluate "p4096-g${G}-cut45"
train "p4096-g${G}-cut4567" sim_stock_p4096 task.num_points=4096 task.num_group=${G} "task.drop_point_labels=[4,5,6,7]"
evaluate "p4096-g${G}-cut4567"

# ---- stage C: patch-size ablation, no cut, fewer points from the same zarr
for n in 2048 1024; do
    train "p${n}-g${G}" "sim_stock_p${n}" task.num_points=${n} task.num_group=${G} "task.dataset.zarr_path=${zarr}"
    evaluate "p${n}-g${G}"
done

log "done"
for t in p4096-g128 p4096-g64 "p4096-g${G}-cut45" "p4096-g${G}-cut4567" "p2048-g${G}" "p1024-g${G}"; do
    ck="$(run_dir "${t}")/checkpoints"
    a=$(python -c "import json;print(json.load(open('${ck}/600.eval_200ep_seed1.json'))['success_rate'])" 2> /dev/null)
    b=$(python -c "import json;print(json.load(open('${ck}/300.eval_50ep_seed0.json'))['success_rate'])" 2> /dev/null)
    log "RESULT ${t}: epoch 600 / 200 scenes ${a:-?} | epoch 300 / 50 scenes ${b:-?}"
done
