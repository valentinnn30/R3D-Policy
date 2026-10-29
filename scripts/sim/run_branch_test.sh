#!/usr/bin/env bash
# Branch test for the branch-and-prune mask search (2026-10-27): does a network trained on the
# UNCUT cloud adapt to the arm cut within a short fine-tune? Resume the uncut 4096/64 baseline
# (seed 42) from its 125.ckpt with the arm cut (task.drop_point_labels=[5]) on the SAME
# schedule (cosine continues, epochs 126..250). Only the evaluated checkpoints are kept
# (checkpoint_every 25 is the only interval that hits 175; the rest is deleted after training,
# val loss comes from wandb every epoch). Compare val loss / success
# with the uncut run (8.5%) and the full arm-cut run (20.0%) via scripts/sim/wandb_val_loss.py.
#
#   WAIT_PID=<pid> setsid nohup bash scripts/sim/run_branch_test.sh > <log> 2>&1 &
#
# The branch dir needs checkpoints/latest.ckpt = the source 125.ckpt (prepared by hand; its
# `_output_dir` pickle dropped so a resume can never write into the source run).
set -uo pipefail
here="$(cd "$(dirname "$0")/../.." && pwd)"
out="${here}/R3D/data/outputs/sim"
src="${out}/mimicgen_sim-square_d2-p4096-g64-42"
cut="${out}/mimicgen_sim-square_d2-p4096-g64-cut5-42"
run="${out}/mimicgen_sim-square_d2-p4096-g64-cut5from125-42"
evals=${EVAL_CKPTS:-"250 175"}
log() { echo "[$(date '+%F %T')] $*"; }
if [ -n "${WAIT_PID:-}" ]; then
    log "waiting for pid ${WAIT_PID}"
    while kill -0 "${WAIT_PID}" 2>/dev/null; do sleep 30; done
fi
[ -f "${run}/checkpoints/latest.ckpt" ] || { log "missing ${run}/checkpoints/latest.ckpt"; exit 1; }
if [ ! -f "${run}/checkpoints/250.ckpt" ]; then
    log "train branch (resume from uncut 125.ckpt, arm cut)"
    TASK_CFG=mimicgen_sim SETTING=agentview_p4096-200 EPOCHS=251 CHECKPOINT_EVERY=25 BATCH_SIZE=64 GRAD_ACCUM=2 \
        TAG=p4096-g64-cut5from125 bash "${here}/scripts/train_robotwin2_sim.sh" square_d2 42 0 \
        task.num_group=64 task.num_points=4096 "task.drop_point_labels=[5]" training.resume=True \
        > "${run}.train.log" 2>&1 < /dev/null
    log "train exit $?"
    [ -f "${run}/checkpoints/250.ckpt" ] || { log "no 250.ckpt"; exit 1; }
fi
for f in "${run}"/checkpoints/*.ckpt; do      # keep only the checkpoints that get evaluated
    e=$(basename "${f}" .ckpt)
    [[ " ${evals} " == *" ${e} "* ]] || { log "delete ${f##*/}"; rm -f "${f}"; }
done
todo=()
for e in ${evals}; do
    [ -f "${run}/checkpoints/${e}.ckpt" ] && [ ! -f "${run}/eval/${e}_200ep_block1.json" ] && todo+=("${run}/checkpoints/${e}.ckpt")
done
if [ ${#todo[@]} -gt 0 ]; then
    log "eval ${evals}"
    (
        eval "$(micromamba shell hook -s bash)"; micromamba activate r3d_robosuite; unset PYTHONPATH
        cd "${here}" && python -W ignore scripts/sim/eval_mimicgen_ckpt.py "${todo[@]}" --episodes 200 --workers 5 \
            >> "${run}/eval.log" 2>&1 < /dev/null
    )
    log "eval exit $?: $(grep '^ckpt:' "${run}/eval.log" | sed 's|.*checkpoints/||; s|  [0-9.]* min.*||' | tr '\n' ' ')"
fi
(
    eval "$(micromamba shell hook -s bash)"; micromamba activate r3d_sim; unset PYTHONPATH
    cd "${here}" && python -W ignore scripts/sim/wandb_val_loss.py "${src}" "${cut}" "${run}" --at 125 150 175 200 250 \
        --json "${run}/branch_test.json"
)
log "branch test done -> ${run}/branch_test.json"
