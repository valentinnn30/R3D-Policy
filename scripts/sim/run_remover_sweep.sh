#!/usr/bin/env bash
# Sweep driver for the learned link remover (docs/learned_remover_design.md). Detached:
#
#   setsid nohup bash scripts/sim/run_remover_sweep.sh R3D/data/outputs/sim/mimicgen/remover_queue.txt \
#       > R3D/data/outputs/sim/mimicgen/remover_sweep_driver.log 2>&1 &
#
# Queue file, RE-READ before every step (lines can be added / removed / reordered while it
# runs), one run per line, '#' comments allowed:
#
#   <points> <seed> <tag> [hydra overrides ...]
#   4096 42 G                                          # the defaults of r3d_robotwin2_remover
#   4096 42 lr1e-3 training.link_remover.lr=1e-3       # a tuning variant: its own tag
#
# Run dir: R3D/data/outputs/sim/mimicgen_sim_remover-square_d2-p<N>-g64-rm-<tag>-<seed>.
# Same budget as the baselines (64 tokens, 251 epochs, batch 128 = 64 x 2), checkpoints
# 50 / 100 / 150 / 200 / 250.
#   pass 1, per line in order: train -> evaluate 250.ckpt on the 200 fixed scenes -> analysis
#   pass 2, per line in order: evaluate 50 / 100 / 150 / 200 -> analysis
#   then the sweep report over every run of the queue (analyze_remover_run.py).
# Steps whose output exists are skipped, so the driver can be restarted at any time.
set -uo pipefail
here="$(cd "$(dirname "$0")/../.." && pwd)"
queue=${1:?usage: run_remover_sweep.sh <queue file>}
[[ "${queue}" = /* ]] || queue="${here}/${queue}"
task=${TASK:-square_d2}
setting=${SETTING:-agentview_p4096-200}
workers=${EVAL_WORKERS:-5}
epochs=${EPOCHS:-251}
ckpt_every=${CKPT_EVERY:-50}
final=${FINAL_CKPT:-250}
mids=${MID_CKPTS:-"50 100 150 200"}
episodes=${EVAL_EPISODES:-200}         # the knobs above only for testing the driver itself
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
log() { echo "[$(date '+%F %T')] $*"; }
run_dir() { echo "${here}/R3D/data/outputs/sim/mimicgen_sim_remover-${task}-p$1-g64-rm-$3-$2"; }
lines() { grep -v '^\s*#' "${queue}" | grep -v '^\s*$'; }
mark_failed() { sed -i "s|^$1 $2 $3\b|# FAILED $1 $2 $3|" "${queue}"; }

evaluate() {  # run_dir, ckpt epochs...
    local run=$1; shift
    local todo=()
    for e in "$@"; do
        [ -f "${run}/checkpoints/${e}.ckpt" ] && [ ! -f "${run}/eval/${e}_${episodes}ep_block1.json" ] && todo+=("${run}/checkpoints/${e}.ckpt")
    done
    [ ${#todo[@]} -eq 0 ] && return 0
    log "eval $(basename "${run}") ckpts $* (${episodes} scenes)"
    (
        eval "$(micromamba shell hook -s bash)"; micromamba activate r3d_robosuite; unset PYTHONPATH
        python -W ignore "${here}/scripts/sim/eval_mimicgen_ckpt.py" "${todo[@]}" --episodes "${episodes}" \
            --workers "${workers}" >> "${run}/eval.log" 2>&1 < /dev/null
    )
    log "eval exit $?: $(grep '^ckpt:' "${run}/eval.log" | tail -${#todo[@]} | sed 's|.*checkpoints/||; s|  [0-9.]* min.*||' | tr '\n' ' ')"
}

analyze() {  # run dirs...
    (
        eval "$(micromamba shell hook -s bash)"; micromamba activate r3d_sim; unset PYTHONPATH
        python -W ignore "${here}/scripts/sim/analyze_remover_run.py" "$@" >> "${here}/R3D/data/outputs/sim/mimicgen/remover_analysis.log" 2>&1 < /dev/null
    ) || log "analysis failed (see remover_analysis.log)"
}

# ---- pass 1: train + final eval + analysis, per line -------------------------------------
while true; do
    next=""
    while read -r p s t rest; do
        run=$(run_dir "$p" "$s" "$t")
        [ -f "${run}/eval/${final}_${episodes}ep_block1.json" ] && continue
        next="$p $s $t ${rest}"; break
    done < <(lines)
    [ -z "${next}" ] && break
    read -r p s t rest <<< "${next}"
    read -r -a extra <<< "${rest}"
    run=$(run_dir "$p" "$s" "$t")
    if [ ! -f "${run}/checkpoints/${final}.ckpt" ]; then
        log "train p${p} seed ${s} tag ${t} ${rest}"
        ALG=r3d_robotwin2_remover TASK_CFG=mimicgen_sim_remover SETTING="${setting}" EPOCHS="${epochs}" \
            CHECKPOINT_EVERY="${ckpt_every}" BATCH_SIZE=64 GRAD_ACCUM=2 TAG="p${p}-g64-rm-${t}" \
            bash "${here}/scripts/train_robotwin2_sim.sh" "${task}" "${s}" 0 \
            task.num_points="${p}" task.num_group=64 "${extra[@]}" > "${run}.train.log" 2>&1 < /dev/null
        rc=$?; log "train p${p} seed ${s} tag ${t} exit ${rc}"
        [ -f "${run}/checkpoints/${final}.ckpt" ] || { log "no ${final}.ckpt: marking FAILED"; mark_failed "$p" "$s" "$t"; continue; }
    fi
    evaluate "${run}" "${final}"
    [ -f "${run}/eval/${final}_${episodes}ep_block1.json" ] || { log "eval wrote no json: marking FAILED"; mark_failed "$p" "$s" "$t"; continue; }
    analyze "${run}"
done

# ---- pass 2: intermediate checkpoints ---------------------------------------------------------
runs=()
while read -r p s t rest; do
    run=$(run_dir "$p" "$s" "$t")
    [ -d "${run}" ] || continue
    runs+=("${run}")
    # shellcheck disable=SC2086
    evaluate "${run}" ${mids}
    analyze "${run}"
done < <(lines)
[ ${#runs[@]} -gt 0 ] && analyze "${runs[@]}"
log "sweep done: ${#runs[@]} runs -> R3D/data/outputs/sim/mimicgen/remover_sweep/sweep_report.md"
