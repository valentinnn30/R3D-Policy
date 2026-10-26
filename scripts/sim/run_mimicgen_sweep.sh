#!/usr/bin/env bash
# MimicGen token sweep (docs/sim_token_allocation_learnings.md, "Next task: MimicGen Square D2").
# Detached driver: trainings run SEQUENTIALLY (two runs on one 4090 are slower in total),
# then every checkpoint is evaluated on the same 200 seeded scenes.
#
#   setsid nohup bash scripts/sim/run_mimicgen_sweep.sh > R3D/data/outputs/sim/mimicgen/sweep1_driver.log 2>&1 &
#
# Defaults = the user's plan (2026-10-24): square_d2, 200 demos, 32 / 16 / 8 points per token at every
# density -- 1024: 128 64 32, 2048: 256 128 64, 4096: 512 256 128 tokens -- one density at a time
# (train its 3 runs largest first, evaluate them, next density),
# EquiDiff budget (250 epochs at batch 128 = 64 x 2), checkpoints 125 + 250, final reported.
#   TOKENS_1024="128 64 32" POINTS="1024" EXTRA="task.drop_point_labels=[5]" TAGSUFFIX=-cut5 ...
set -uo pipefail
here="$(cd "$(dirname "$0")/../.." && pwd)"
task=${TASK:-square_d2}
setting=${SETTING:-agentview_p4096-200}
points_list=${POINTS:-"1024 2048 4096"}
seed=${SEED:-42}
extra=${EXTRA:-}
tagsuffix=${TAGSUFFIX:-}
episodes=${EPISODES:-200}          # final checkpoints: the result (>= 200 paired scenes, agreed principle)
episodes_mid=${EPISODES_MID:-100}  # mid checkpoints: convergence check only; scenes = the first 100 of the 200
workers=${EVAL_WORKERS:-5}
# measured 2026-10-24: 4096/512 OOMs at micro-batch 64 (32 fits); 2048/256 peaks 20.4 of 24 GB at 64
export MB_4096_512=${MB_4096_512:-32}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   # fragmentation guard, numerically identical
log() { echo "[$(date '+%F %T')] $*"; }
tokens_for() { local v="TOKENS_$1" d; case $1 in 1024) d="128 64 32";; 2048) d="256 128 64";; 4096) d="512 256 128";; esac; echo "${!v:-$d}"; }
# micro-batch per points x tokens (effective batch stays 128 = micro x accum); MB_<points>_<tokens> overrides.
# Default 64; set from the memory test before launch.
micro_batch_for() { local v="MB_$1_$2"; echo "${!v:-64}"; }

train_and_eval() {   # $1 = points
    local points=$1 runs=() g tag run ck ep
    for g in $(tokens_for "${points}"); do
        tag="p${points}-g${g}${tagsuffix}"
        run="${here}/R3D/data/outputs/sim/mimicgen_sim-${task}-${tag}-${seed}"
        runs+=("${run}")
        if [ -f "${run}/checkpoints/250.ckpt" ]; then log "train ${tag}: 250.ckpt exists, skipping"; continue; fi
        log "train ${tag} (task.num_group=${g} task.num_points=${points} ${extra})"
        local mb; mb=$(micro_batch_for "${points}" "${g}")
        log "  micro-batch ${mb} x accum $((128 / mb))"
        TASK_CFG=mimicgen_sim SETTING="${setting}" EPOCHS=251 CHECKPOINT_EVERY=125 BATCH_SIZE="${mb}" GRAD_ACCUM=$((128 / mb)) \
            TAG="${tag}" bash "${here}/scripts/train_robotwin2_sim.sh" "${task}" "${seed}" 0 \
            task.num_group="${g}" task.num_points="${points}" ${extra} > "${run}.train.log" 2>&1
        log "train ${tag} exit $?"
    done
    (   # robosuite env only inside this subshell; training activates r3d_sim itself
        eval "$(micromamba shell hook -s bash)"
        micromamba activate r3d_robosuite
        unset PYTHONPATH
        for ep in 250 125; do          # final checkpoints first: they are the result
            for run in "${runs[@]}"; do
                ck="${run}/checkpoints/${ep}.ckpt"
                [ -f "${ck}" ] || { log "eval: missing ${ck}"; continue; }
                local n=${episodes}; [ "${ep}" = 250 ] || n=${episodes_mid}
                log "eval ${ck} (${n} scenes)"
                python -W ignore "${here}/scripts/sim/eval_mimicgen_ckpt.py" "${ck}" --episodes "${n}" \
                    --workers "${workers}" > "${run}/eval_${ep}.log" 2>&1
                log "eval exit $?: $(grep '^ckpt:' "${run}/eval_${ep}.log" | tail -1)"
            done
        done
    )
}

for points in ${points_list}; do
    train_and_eval "${points}"
done
log "done"
