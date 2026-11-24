#!/usr/bin/env bash
# Run queue for MimicGen runs (docs/sim_token_allocation_learnings.md). Detached:
#
#   setsid nohup bash scripts/sim/run_mimicgen_queue.sh R3D/data/outputs/sim/mimicgen/queue.txt \
#       > R3D/data/outputs/sim/mimicgen/queue_driver.log 2>&1 &
#
# The queue file is RE-READ before every run, so lines can be added, removed or reordered
# while it runs. One run per line, '#' comments allowed:
#
#   <points> <tokens> <seed> <drop_point_labels or ->     e.g.  1024 64 42 [5]
#   <points> <tokens> <seed> <drop or -> <tag> [hydra overrides...]   e.g.  4096 64 42 - armsparse task.thin_point_set=arm
#
# Optional <tag> (claim sweep, 2026-11-12): run dir suffix "-<tag>" instead of the cut suffix, and any further
# fields are passed to train.py as hydra overrides. CKPT_EPOCHS='[200,250]' saves exactly those epochs
# (+training.checkpoint_epochs) instead of 125 + 250. Lines without a tag and CKPT_EPOCHS unset: unchanged.
#
# Each run: train (skipped if 250.ckpt exists; batch 128 = micro x accum, 251 epochs, ckpts
# 125 + 250), then evaluate 250.ckpt on the 200 fixed scenes (skipped if the json exists).
# A run is done when its eval json exists. Stops when no line is left to do.
set -uo pipefail
here="$(cd "$(dirname "$0")/../.." && pwd)"
queue=${1:?usage: run_mimicgen_queue.sh <queue file>}
task=${TASK:-square_d2}
setting=${SETTING:-agentview_p4096-200}
workers=${EVAL_WORKERS:-5}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
log() { echo "[$(date '+%F %T')] $*"; }
micro_batch() { case "$1/$2" in 4096/512) echo 32;; *) echo 64;; esac; }   # measured 2026-10-24

run_dir() {  # points tokens seed drop [tag]
    local suffix=""; [ "$4" != "-" ] && suffix="-cut$(echo "$4" | tr -d '[] ' | tr ',' '_')"
    [ -n "${5:-}" ] && [ "${5}" != "-" ] && suffix="-$5"
    echo "${here}/R3D/data/outputs/sim/mimicgen_sim-${task}-p$1-g$2${suffix}-$3"
}

qfile="${here}/${queue#${here}/}"
mark_failed() {  # exact line -> "# FAILED <line>"
    local n; n=$(grep -nxF -- "$1" "${qfile}" | head -1 | cut -d: -f1)
    [ -n "${n}" ] && sed -i "${n}s|^|# FAILED |" "${qfile}"
}
ckpt_args=(); [ -n "${CKPT_EPOCHS:-}" ] && ckpt_args=("+training.checkpoint_epochs=${CKPT_EPOCHS}")

while true; do
    next=""
    while read -r line; do
        read -r p g s d t _ <<< "${line}"
        [[ -z "${p}" || "${p}" == \#* ]] && continue
        [ -f "$(run_dir "$p" "$g" "$s" "$d" "${t:-}")/eval/250_200ep_block1.json" ] && continue
        next="${line}"; break
    done < "${qfile}"
    [ -z "${next}" ] && { log "queue empty, done"; break; }
    read -r p g s d t rest <<< "${next}"
    read -r -a over <<< "${rest:-}"
    run=$(run_dir "$p" "$g" "$s" "$d" "${t:-}"); tag=$(basename "${run}"); tag=${tag#mimicgen_sim-${task}-}; tag=${tag%-${s}}
    extra=(); [ "$d" != "-" ] && extra=("task.drop_point_labels=$d")
    extra+=("${over[@]}" "${ckpt_args[@]}")
    if [ ! -f "${run}/checkpoints/250.ckpt" ]; then
        mb=$(micro_batch "$p" "$g")
        log "train ${tag} seed ${s} (points ${p} tokens ${g} drop ${d}; micro-batch ${mb} x $((128 / mb))) ${extra[*]}"
        TASK_CFG=mimicgen_sim SETTING="${setting}" EPOCHS=251 CHECKPOINT_EVERY=125 BATCH_SIZE="${mb}" \
            GRAD_ACCUM=$((128 / mb)) TAG="${tag}" bash "${here}/scripts/train_robotwin2_sim.sh" "${task}" "${s}" 0 \
            task.num_group="${g}" task.num_points="${p}" "${extra[@]}" > "${run}.train.log" 2>&1
        rc=$?; log "train ${tag} seed ${s} exit ${rc}"
        [ -f "${run}/checkpoints/250.ckpt" ] || { log "no 250.ckpt for ${tag} seed ${s}: skipping it"; mark_failed "${next}"; continue; }
    fi
    log "eval ${tag} seed ${s} (200 scenes)"
    (
        eval "$(micromamba shell hook -s bash)"; micromamba activate r3d_robosuite; unset PYTHONPATH
        python -W ignore "${here}/scripts/sim/eval_mimicgen_ckpt.py" "${run}/checkpoints/250.ckpt" --episodes 200 \
            --workers "${workers}" > "${run}/eval_250.log" 2>&1
    )
    log "eval exit $?: $(grep '^ckpt:' "${run}/eval_250.log" | tail -1 | sed 's|.*success|success|')"
    [ -f "${run}/eval/250_200ep_block1.json" ] || { log "eval wrote no json for ${tag} seed ${s}: marking FAILED"; mark_failed "${next}"; }
done
