#!/usr/bin/env bash
# Token sweep for the sim baseline: same data, same budget, only num_group varies.
#
#   bash scripts/sweep_robotwin2_sim_tokens.sh <gpu_id> [token counts...]
#   bash scripts/sweep_robotwin2_sim_tokens.sh 0                 # 128 64 32 16
#   PARALLEL=1 bash scripts/sweep_robotwin2_sim_tokens.sh 0 64 32
#
# Fewer tokens is the lever the crop method works through: group_size stays 32,
# so 128/64/32/16 tokens cover 4x/2x/1x/0.5x of the 1024 points.  Runs land in
# R3D/data/outputs/sim/robotwin2_sim_p1024-<task>-g<N>-<seed>/ (the g<N> tag is
# what differs; the 2026-10-17 cup runs carry the old config name
# robotwin2_sim_p1024_g128).  Logs: outputs/sim/sweep_<task>_g<N>_seed<seed>.log.
# Task 2: TASK=stack_blocks_two_fov SETTING=sim_stock_p1024 bash ... 0

set -uo pipefail
gpu_id=${1:?usage: sweep_robotwin2_sim_tokens.sh <gpu_id> [token counts...]}
shift
tokens=("$@"); [ ${#tokens[@]} -eq 0 ] && tokens=(128 64 32 16)
task=${TASK:-place_empty_cup_right}
seed=${SEED:-42}
parallel=${PARALLEL:-2}   # runs sharing the GPU at once (micro-batch 64 each)
here="$(cd "$(dirname "$0")" && pwd)"
outroot="${here}/../R3D/data/outputs/sim"
mkdir -p "${outroot}"

running=0
for g in "${tokens[@]}"; do
    log="${outroot}/sweep_${task}_g${g}_seed${seed}.log"
    echo "[sweep] g=${g} -> ${log}"
    TAG="g${g}" bash "${here}/train_robotwin2_sim.sh" "${task}" "${seed}" "${gpu_id}" \
        task.num_group="${g}" > "${log}" 2>&1 &
    running=$((running + 1))
    if [ "${running}" -ge "${parallel}" ]; then wait -n; running=$((running - 1)); fi
    sleep 60   # stagger start-up (dataset load + Uni3D init)
done
wait
echo "[sweep] done"
