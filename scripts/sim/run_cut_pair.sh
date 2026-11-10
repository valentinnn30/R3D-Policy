#!/usr/bin/env bash
# Arm cut vs arm cut with link7 kept, ONE seed, the identical process for both: plain DP3, the dataset's oracle cut
# after augmentation via `task.drop_point_links` -- [link0..link7] (bit-identical to drop_point_labels=[5], the arm
# cut; scripts/sim/check_link_cut.py) vs [link0..link6]. The list is the only difference. Strictly one GPU job at a time:
#
#   SEED=44 setsid nohup bash scripts/sim/run_cut_pair.sh > R3D/data/outputs/sim/mimicgen/cut_pair_s44/chain.log 2>&1 &
#
#   0 wait until no training / eval / sweep driver has run for 10 min
#   1 checks: check_link_cut.py (dataset + --eval)
#   2 train cutL07, eval 250; train cutL06, eval 250 (seed SEED, 251 epochs, 64 x 2, ckpts 200 + 250)
#   3 eval 200 of both
# Any failure stops the chain (set -e). Steps whose output exists are skipped.
set -euo pipefail
here="$(cd "$(dirname "$0")/../.." && pwd)"
cd "${here}"
seed=${SEED:?SEED=<training seed>}
out=R3D/data/outputs/sim/mimicgen/cut_pair_s${seed}
mkdir -p "${out}"
log() { echo "[$(date '+%F %T')] $*"; }
env_on() { eval "$(micromamba shell hook -s bash)"; micromamba activate "$1"; unset PYTHONPATH; }
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
declare -A cut=(
    [cutL07]="task.drop_point_links=[link0,link1,link2,link3,link4,link5,link6,link7]"
    [cutL06]="task.drop_point_links=[link0,link1,link2,link3,link4,link5,link6]"
)
rd() { echo "R3D/data/outputs/sim/mimicgen_sim-square_d2-p4096-g64-$1-${seed}"; }

busy() { ps -eo cmd | awk '($1=="bash" && $2 ~ /run_round[0-9]+_chain|run_remover_sweep|run_finetune_branches|run_mask_override|run_branch_test|run_early_branch|run_cut_keep7/) || ($1 ~ /python/ && ($0 ~ /train\.py/ || $0 ~ /eval_mimicgen_ckpt/))' | grep -q .; }
log "waiting for the GPU to be free for 10 min"
free=0
while [ "${free}" -lt 10 ]; do
    if busy; then free=0; else free=$((free + 1)); fi
    sleep 60
done
log "GPU free for 10 min"

if [ ! -f "${out}/checks.ok" ]; then
    log "checks"
    ( env_on r3d_sim; python -W ignore scripts/sim/check_link_cut.py ) > "${out}/check_link_cut.log" 2>&1
    tail -1 "${out}/check_link_cut.log"
    ( env_on r3d_robosuite; python -W ignore scripts/sim/check_link_cut.py --eval ) > "${out}/check_link_cut_eval.log" 2>&1 || true
    grep -q "^ALL PASS" "${out}/check_link_cut_eval.log" || { log "eval-side check failed"; exit 1; }
    touch "${out}/checks.ok"
fi

train() {
    local tag=$1 run; run=$(rd "$1")
    [ -f "${run}/checkpoints/250.ckpt" ] && return 0
    log "train ${tag} seed ${seed}: ${cut[$tag]}"
    TASK_CFG=mimicgen_sim SETTING=agentview_p4096-200 EPOCHS=251 CHECKPOINT_EVERY=125 BATCH_SIZE=64 GRAD_ACCUM=2 \
        TAG="p4096-g64-${tag}" bash scripts/train_robotwin2_sim.sh square_d2 "${seed}" 0 \
        task.num_group=64 task.num_points=4096 "${cut[$tag]}" '+training.checkpoint_epochs=[200,250]' \
        > "${run}.train.log" 2>&1
    log "train ${tag} exit: $(ls "${run}/checkpoints" | tr '\n' ' ')"
}

evaluate() {
    local run; run=$(rd "$1")
    [ -f "${run}/eval/$2_200ep_block1.json" ] && return 0
    log "eval $1 $2"
    ( env_on r3d_robosuite; python -W ignore scripts/sim/eval_mimicgen_ckpt.py "${run}/checkpoints/$2.ckpt" \
        --episodes 200 --workers 5 ) >> "${run}/eval.log" 2>&1
    log "eval $1 $2: $(grep '^ckpt:' "${run}/eval.log" | tail -1 | sed 's|.*checkpoints/||; s|  [0-9.]* min.*||')"
}

train cutL07; evaluate cutL07 250
train cutL06; evaluate cutL06 250
evaluate cutL07 200; evaluate cutL06 200
log "cut pair seed ${seed} done"
