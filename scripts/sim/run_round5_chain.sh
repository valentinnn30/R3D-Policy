#!/usr/bin/env bash
# Round 5 chain (2026-11-04, user): link0-4 were never removable, but link3 / link4 come into view in ~4-5%
# of frames with up to ~1k of 4096 points. Order:
#   1. let round 4b's running eval (DROP-43 250) finish, then pause round 4b (its remaining evals rerun in 4)
#   2. GObal-42 250.ckpt evaluated with link0..link6 removed (only link7, gripper, scene kept)
#   3. GObalAll: GObal with gates for link0..link4 too (remover_tune5_queue.txt), eval 250 then 200
#   4. resume round 4b (DROP-43 200, GObal-43 200 on the uncut cloud; finished steps are skipped)
#   setsid nohup bash scripts/sim/run_round5_chain.sh > R3D/data/outputs/sim/mimicgen/remover_tune5_driver.log 2>&1 &
set -uo pipefail
here="$(cd "$(dirname "$0")/../.." && pwd)"
sim="${here}/R3D/data/outputs/sim"
log() { echo "[$(date '+%F %T')] $*"; }
tree() { local c; for c in $(ps -o pid= --ppid "$1"); do tree "${c}"; echo "${c}"; done; }
cd "${here}"

drop43="${sim}/mimicgen_sim_remover-square_d2-p4096-g64-rm-DROP-43"
until [ -f "${drop43}/eval/250_200ep_block1.json" ]; do sleep 30; done
for r in $(ps -eo pid,cmd | awk '$2=="bash" && $3=="scripts/sim/run_finetune_branches.sh" {print $1}'); do
    k="$(tree "${r}") ${r}"; kill ${k} 2>/dev/null
done
sleep 10
log "round 4b paused after DROP-43 250 ($(python3 -c "import json;print(json.load(open('${drop43}/eval/250_200ep_block1.json'))['success_rate'])"))"

g="${sim}/mimicgen_sim_remover-square_d2-p4096-g64-rm-GObal-42"
out="${g}/eval/250_200ep_block1_mask-link0+link1+link2+link3+link4+link5+link6.json"
if [ ! -f "${out}" ]; then
    log "eval GObal-42 250 with link0..link6 removed"
    (eval "$(micromamba shell hook -s bash)"; micromamba activate r3d_robosuite; unset PYTHONPATH
     python -W ignore scripts/sim/eval_mimicgen_ckpt.py "${g}/checkpoints/250.ckpt" --episodes 200 --workers 5 \
        --remover-removed link0,link1,link2,link3,link4,link5,link6 >> "${g}/eval.log" 2>&1 < /dev/null)
    log "eval exit $?: $(grep '^ckpt:' "${g}/eval.log" | tail -1 | sed 's|.*checkpoints/||; s|  [0-9.]* min.*||')"
fi

log "round 5 sweep (GObalAll)"
KEEP_ONLY_EVALUATED=1 SWEEP_OUT=R3D/data/outputs/sim/mimicgen/remover_tune5 MID_CKPTS=200 \
    bash scripts/sim/run_remover_sweep.sh R3D/data/outputs/sim/mimicgen/remover_tune5_queue.txt
log "round 5 sweep exit $?"

log "resume round 4b"
bash scripts/sim/run_finetune_branches.sh >> "${sim}/mimicgen/remover_tune4b_driver.log" 2>&1 < /dev/null
log "round 5 chain done"
