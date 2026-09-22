#!/usr/bin/env bash
# Hybrid 2D/3D policy (DP3Hybrid, r3d_hybrid.yaml). Same launcher, same
# arguments and defaults as the 3D and 2D runs -- only the algorithm config
# differs. Plan: ~/ros2_ws/hybrid_2d3d_ablations_plan.txt
#
#   bash scripts/train_real_bimanual_hybrid.sh <task_config> <seed> <gpu_id> [hydra overrides...]
#
#   task_config: real_bimanual_hybrid_top3d | real_bimanual_hybrid_wrist3d |
#                real_bimanual_hybrid_wrist3d_fused
#   e.g. task.dataset.pose_source_3d=proprio task.dataset.pose_source_2d=none
#
# Give every variant of the same task config its OWN seed: run_dir is built
# from task + seed, not from the overrides, so a shared seed overwrites.
set -euo pipefail
exec env ALG="${ALG:-r3d_hybrid}" bash "$(dirname "$0")/train_real_bimanual.sh" "$@"
