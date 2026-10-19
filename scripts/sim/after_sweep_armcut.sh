#!/usr/bin/env bash
# Follow-up to the token sweep (CLAUDE.md "Sim baseline"), queued 2026-10-17:
#   1. wait until the sweep's evaluation has finished (WAIT_PID = eval_sweep.sh)
#   2. pick the token count whose 600-epoch success is in [30%, 40%]
#      (several -> closest to 35%; none -> closest to 35% overall, run anyway,
#      as decided by the user -- the choice and all numbers go to selection.json)
#   3. train the SAME run with the arm points cut (oracle label cut,
#      task.drop_point_labels=$DROP, default [5] = right arm): same data, seed,
#      601 epochs (-> 600.ckpt), batch 64 x 4
#   4. evaluate its 200/400/600 checkpoints, 50 episodes each
#
#   WAIT_PID=<eval_sweep pid> bash scripts/sim/after_sweep_armcut.sh                  # task 1 (cup)
#   TASK=stack_blocks_two_fov SETTING=sim_stock_p1024 DROP='[4,5]' WAIT_PID=<pid> \
#       bash scripts/sim/after_sweep_armcut.sh                                         # task 2, both arms
#
# Never writes into the sweep's run dirs: the new run is TAG g<N>-armcut, its
# eval logs and jsons live in that run's own directory.
set -uo pipefail
here="$(cd "$(dirname "$0")/../.." && pwd)"
sim="${here}/R3D/data/outputs/sim"
task=${TASK:-place_empty_cup_right}
setting=${SETTING:-sim_head_p1024}
drop=${DROP:-[5]}             # point_labels classes: 4 left arm, 5 right arm (grippers 6/7 kept)
sel_file="${sim}/armcut_selection_${task}.json"
seed=${SEED:-42}
episodes=${EPISODES:-50}
log() { echo "[$(date '+%F %T')] $*"; }

if [ -n "${WAIT_PID:-}" ]; then
    log "waiting for PID ${WAIT_PID} (sweep evaluation)"
    while kill -0 "${WAIT_PID}" 2> /dev/null; do sleep 120; done
fi

eval "$(micromamba shell hook -s bash)"
micromamba activate r3d_sim
unset PYTHONPATH

sel=$(python - "${sim}" "${task}" "${seed}" "${episodes}" <<'EOF'
import glob, json, os, re, sys
sim, task, seed, eps = sys.argv[1:]
res = {}
for f in glob.glob(f"{sim}/*-{task}-g*-{seed}/checkpoints/600.eval_{eps}ep_seed0.json"):
    m = re.search(r"-g(\d+)-" + seed + "/", f)
    if m and "armcut" not in f:
        res[int(m.group(1))] = json.load(open(f))["success_rate"]
if not res:
    print("NONE"); sys.exit()
inside = {g: s for g, s in res.items() if 0.30 <= s <= 0.40}
pool = inside or res
g = min(pool, key=lambda k: (abs(pool[k] - 0.35), k))
out = {"success_600": {str(k): v for k, v in sorted(res.items())}, "chosen_num_group": g,
       "rule": "600-epoch success in [0.30, 0.40]; several -> closest to 0.35; none -> closest to 0.35 overall",
       "in_range": bool(inside)}
json.dump(out, open(f"{sim}/armcut_selection_{task}.json", "w"), indent=2)
print(g)
EOF
)
if [ "${sel}" = "NONE" ] || [ -z "${sel}" ]; then
    log "no 600-epoch eval results found under ${sim} -- NOT starting the arm-cut run"
    exit 1
fi
log "selection: $(cat "${sel_file}" | tr -d '\n' | tr -s ' ')"
g=${sel}
tag="g${g}-armcut"

log "training ${tag} (task.num_group=${g} task.drop_point_labels=${drop}, 601 epochs -> 600.ckpt, 64 x 4)"
TAG="${tag}" SETTING="${setting}" EPOCHS=601 BATCH_SIZE=64 GRAD_ACCUM=4 bash "${here}/scripts/train_robotwin2_sim.sh" \
    "${task}" "${seed}" 0 task.num_group="${g}" "task.drop_point_labels=${drop}" \
    > "${sim}/sweep_${task}_${tag}_seed${seed}.log" 2>&1
log "training exit $?"

ckdir=$(ls -d "${sim}"/*-"${task}"-"${tag}"-"${seed}"/checkpoints 2> /dev/null | head -1)
if [ -z "${ckdir}" ]; then log "no checkpoint dir for ${tag}"; exit 1; fi
# one process per checkpoint, staggered (the eval-seed cache is already filled
# by the sweep evaluation, so they only read it)
for e in 200 400 600; do
    [ -f "${ckdir}/${e}.ckpt" ] || { log "missing ${e}.ckpt"; continue; }
    python -W ignore "${here}/scripts/eval_robotwin2_ckpt.py" "${ckdir}/${e}.ckpt" --episodes "${episodes}" \
        > "$(dirname "${ckdir}")/eval_${e}.log" 2>&1 &
    sleep 120
done
wait
log "evaluation done"
for e in 200 400 600; do
    f="${ckdir}/${e}.eval_${episodes}ep_seed0.json"
    [ -f "${f}" ] && log "${tag} epoch ${e}: success $(python -c "import json;print(json.load(open('${f}'))['success_rate'])")"
done
