#!/usr/bin/env bash
# Round 4b (2026-11-03): WHY does GObal work? Fine-tune branches from the dropout-only run.
#
# GObal and DROP train theta identically up to epoch 225 (keep-0.5 region dropout; phi never touches
# theta's masks). GObal then fine-tunes 25 epochs on its learned mask; DROP does not. DROP fell from
# 18.0% at 200 to 13.0% at 250, GObal from 22.0% to 19.5%. Branches: DROP-42's 200.ckpt resumed with
# the IDENTICAL config, so every branch is the same network up to epoch 225 (training is deterministic;
# checked afterwards on the val loss of epochs 201-225), then a different fine-tune 226-250:
#
#   FTcont     no fine-tune (dropout to 250)        -- the resume control (a re-run of DROP's tail)
#   FTlearned  fine-tune on link5 + link6 removed   -- GObal's procedure without phi learning
#   FTnone     fine-tune on the full cloud           -- does ANY fixed-cloud fine-tune do it?
#   FTarm      fine-tune on link5 + 6 + 7 removed   -- another mask
#
# plus DROP seed 43 (dropout-only replication) and GObal-43's 200.ckpt on the uncut cloud (the
# pre-fine-tune comparison on seed 43). EVERYTHING runs one job at a time, and only once no other
# sweep/eval driver is running: an eval's 5 workers take ~25 of the 32 cores, and a training next to
# one ran at 5.8 s/it instead of ~0.1 (measured 2026-11-03). Eval threads are NOT capped: that could
# change borderline scenes and break the per-scene comparability with every earlier eval.
# Order: the 4 short branch trainings (~45 min each), their evals, then DROP-43 (train + 2 evals),
# then GObal-43 200 on the uncut cloud.
#
#   setsid nohup bash scripts/sim/run_finetune_branches.sh > R3D/data/outputs/sim/mimicgen/remover_tune4b_driver.log 2>&1 &
#
# Steps whose output exists are skipped, so the driver can be restarted.
set -uo pipefail
here="$(cd "$(dirname "$0")/../.." && pwd)"
sim="${here}/R3D/data/outputs/sim"
src="${sim}/mimicgen_sim_remover-square_d2-p4096-g64-rm-DROP-42/checkpoints/200.ckpt"
rd() { echo "${sim}/mimicgen_sim_remover-square_d2-p4096-g64-rm-$1-$2"; }
log() { echo "[$(date '+%F %T')] $*"; }
act() { eval "$(micromamba shell hook -s bash)"; micromamba activate "$1"; unset PYTHONPATH; }
drop_ov=(policy.link_remover.learn_density=false policy.link_remover.init_keep=0.5 policy.link_remover.theta_keep=0.5
         training.link_remover.lr=0 training.link_remover.phi_every=1000000000)
declare -A ft=(
    [FTcont]="training.link_remover.finetune_frac=0"
    [FTlearned]="training.link_remover.finetune_mask=[link5,link6]"
    [FTnone]="training.link_remover.finetune_mask=[]"
    [FTarm]="training.link_remover.finetune_mask=[link5,link6,link7]"
)
order="FTlearned FTcont FTnone FTarm"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

train() {  # tag seed overrides...
    local tag=$1 seed=$2; shift 2
    ALG=r3d_robotwin2_remover TASK_CFG=mimicgen_sim_remover SETTING=agentview_p4096-200 EPOCHS=251 CHECKPOINT_EVERY=50 \
        BATCH_SIZE=64 GRAD_ACCUM=2 TAG="p4096-g64-rm-${tag}" bash "${here}/scripts/train_robotwin2_sim.sh" square_d2 "${seed}" 0 \
        task.num_points=4096 task.num_group=64 "${drop_ov[@]}" "$@"
}

busy() {   # another sweep / eval driver still owns the GPU and the CPUs
    ps -eo cmd | awk '$1=="bash" && ($2=="scripts/sim/run_remover_sweep.sh" || $2=="scripts/sim/run_mask_override_evals.sh")' | grep -q .
}

branches() {
    for b in ${order}; do
        run=$(rd "${b}" 42)
        [ -f "${run}/.train_done" ] && continue
        mkdir -p "${run}/checkpoints"
        # a copy of DROP's 200.ckpt without its `_output_dir` pickle, so the resume writes here
        (act r3d_sim; python - "${src}" "${run}/checkpoints/latest.ckpt" <<'EOF'
import sys, torch, dill
p = torch.load(open(sys.argv[1], "rb"), pickle_module=dill, map_location="cpu")
p["pickles"].pop("_output_dir", None)
torch.save(p, open(sys.argv[2], "wb"), pickle_module=dill)
EOF
        ) || { log "${b}: checkpoint copy failed"; continue; }
        while busy; do sleep 60; done
        log "train ${b} (resume DROP-42 200.ckpt) ${ft[$b]}"
        train "${b}" 42 training.resume=True "${ft[$b]}" > "${run}.train.log" 2>&1 < /dev/null
        log "train ${b} exit $?"
        rm -f "${run}/checkpoints/latest.ckpt"
        [ -f "${run}/checkpoints/250.ckpt" ] && touch "${run}/.train_done" || log "${b}: no 250.ckpt"
    done
}

drop43() {
    run=$(rd DROP 43)
    if [ ! -f "${run}/.train_done" ]; then
        log "train DROP seed 43"
        train DROP 43 training.link_remover.finetune_frac=0 > "${run}.train.log" 2>&1 < /dev/null
        log "train DROP-43 exit $?"
        for e in 50 100 150; do rm -f "${run}/checkpoints/${e}.ckpt"; done
        [ -f "${run}/checkpoints/250.ckpt" ] && touch "${run}/.train_done" || log "DROP-43: no 250.ckpt"
    fi
}

evaluate() {  # run ckpt-epoch mask(natural|none)
    local run=$1 e=$2 m=$3 suffix="" extra=()
    [ "${m}" = none ] && { suffix="_mask-none"; extra=(--remover-removed none); }
    local out="${run}/eval/${e}_200ep_block1${suffix}.json"
    [ -f "${out}" ] && return 0
    [ -f "${run}/checkpoints/${e}.ckpt" ] || { log "skip eval $(basename "${run}") ${e}: no checkpoint"; return 1; }
    while busy; do sleep 60; done
    log "eval $(basename "${run}") ${e}.ckpt mask ${m}"
    (act r3d_robosuite; python -W ignore "${here}/scripts/sim/eval_mimicgen_ckpt.py" "${run}/checkpoints/${e}.ckpt" \
        --episodes 200 --workers 5 "${extra[@]}" >> "${run}/eval.log" 2>&1 < /dev/null)
    log "eval exit $?: $(grep '^ckpt:' "${run}/eval.log" | tail -1 | sed 's|.*checkpoints/||; s|  [0-9.]* min.*||')"
}

branches
evaluate "$(rd FTlearned 42)" 250 natural
evaluate "$(rd FTcont 42)" 250 natural
evaluate "$(rd FTnone 42)" 250 natural
evaluate "$(rd FTarm 42)" 250 natural
evaluate "$(rd FTlearned 42)" 250 none
drop43
evaluate "$(rd DROP 43)" 250 natural
evaluate "$(rd DROP 43)" 200 natural
evaluate "$(rd GObal 43)" 200 none
log "round 4b done"
