#!/usr/bin/env bash
# Evaluate every run of a token sweep at several epochs (CLAUDE.md "Sim baseline").
#
#   bash scripts/sim/eval_sweep.sh                       # all g* runs, 200 400 600, 50 eps
#   EPISODES=100 EPOCHS="600" bash scripts/sim/eval_sweep.sh
#   WAIT_PID=<pid of the sweep driver> bash scripts/sim/eval_sweep.sh   # start after training
#
# One process per run (the RoboTwin runner is built once per process), PARALLEL at
# a time, starts staggered: RoboTwin validates eval seeds with the expert and
# caches them in robotwin2/seeds_list/, a shared json -- the first process fills
# it, later ones read it instead of racing on the write.
set -uo pipefail
here="$(cd "$(dirname "$0")/../.." && pwd)"
runs_glob=${RUNS:-"${here}/R3D/data/outputs/sim/*-g*-*/checkpoints"}
epochs=${EPOCHS:-"200 400 600"}
episodes=${EPISODES:-50}
parallel=${PARALLEL:-2}
stagger=${STAGGER:-900}

if [ -n "${WAIT_PID:-}" ]; then
    while kill -0 "${WAIT_PID}" 2> /dev/null; do sleep 120; done
fi
eval "$(micromamba shell hook -s bash)"
micromamba activate r3d_sim
unset PYTHONPATH

running=0
for ckdir in ${runs_glob}; do
    run=$(basename "$(dirname "${ckdir}")")
    ckpts=()
    for e in ${epochs}; do [ -f "${ckdir}/${e}.ckpt" ] && ckpts+=("${ckdir}/${e}.ckpt"); done
    [ ${#ckpts[@]} -eq 0 ] && { echo "[eval] ${run}: no checkpoints"; continue; }
    log="$(dirname "${ckdir}")/eval.log"
    echo "[eval] ${run}: ${#ckpts[@]} ckpts -> ${log}"
    python -W ignore "${here}/scripts/eval_robotwin2_ckpt.py" "${ckpts[@]}" \
        --episodes "${episodes}" > "${log}" 2>&1 &
    running=$((running + 1))
    if [ "${running}" -ge "${parallel}" ]; then wait -n; running=$((running - 1)); else sleep "${stagger}"; fi
done
wait
echo "[eval] done"
grep -h "success" "${here}"/R3D/data/outputs/sim/*-g*-*/eval.log | grep "ckpt:" | sort
