#!/usr/bin/env bash
# Round-3 follow-up (2026-11-02): evaluate learned-remover checkpoints with ANOTHER mask than
# the learned one (eval_mimicgen_ckpt.py --remover-removed), on the same 200 scenes. Detached:
#
#   WAIT_PID=<sweep driver pid> setsid nohup bash scripts/sim/run_mask_override_evals.sh \
#       > R3D/data/outputs/sim/mimicgen/remover_tune3_override.log 2>&1 &
#
# Questions: does GObal keep its success on the UNCUT cloud (mask "none": the dropout-trained
# network, not the mask, carries the gain), and with link7 also removed (the arm cut's mask)?
# Output: <run>/eval/250_200ep_block1_mask-<removed>.json; existing results are skipped.
set -uo pipefail
here="$(cd "$(dirname "$0")/../.." && pwd)"
sim="${here}/R3D/data/outputs/sim"
runs=${RUNS:-"mimicgen_sim_remover-square_d2-p4096-g64-rm-GObal-42 mimicgen_sim_remover-square_d2-p4096-g64-rm-GObal-43"}
masks=${MASKS:-"none link5,link6,link7"}
ckpt=${CKPT:-250}
episodes=${EVAL_EPISODES:-200}
workers=${EVAL_WORKERS:-5}
log() { echo "[$(date '+%F %T')] $*"; }

if [ -n "${WAIT_PID:-}" ]; then
    log "waiting for pid ${WAIT_PID} (GPU queue) to exit"
    while kill -0 "${WAIT_PID}" 2>/dev/null; do sleep 60; done
fi
eval "$(micromamba shell hook -s bash)"; micromamba activate r3d_robosuite; unset PYTHONPATH
for m in ${masks}; do                  # mask-major: the uncut-cloud result of both seeds first
    tag=$([ "${m}" = none ] && echo none || echo "${m//,/+}")
    for r in ${runs}; do
        out="${sim}/${r}/eval/${ckpt}_${episodes}ep_block1_mask-${tag}.json"
        [ -f "${out}" ] && { log "exists: ${out}"; continue; }
        log "eval ${r} ${ckpt}.ckpt, removed: ${m}"
        python -W ignore "${here}/scripts/sim/eval_mimicgen_ckpt.py" "${sim}/${r}/checkpoints/${ckpt}.ckpt" \
            --episodes "${episodes}" --workers "${workers}" --remover-removed "${m}" >> "${sim}/${r}/eval.log" 2>&1 < /dev/null
        log "exit $?: $(grep '^ckpt:' "${sim}/${r}/eval.log" | tail -1 | sed 's|.*checkpoints/||; s|  [0-9.]* min.*||')"
    done
done
log "override evals done"
