#!/usr/bin/env bash
# R1 robot-cut run (2026-10-05, ~/ros2_ws/robot_cut_r1_plan.txt) with EXACTLY
# the settings K ran with (read off K's .hydra/overrides.yaml): seed 52, batch
# 256 x accum 1, 301 epochs (cosine to 0 there), checkpoint every 150, val every
# 10, 8 + 1 persistent workers, wandb online. Only the task differs.
#
#   bash scripts/train_r1_robotcut.sh [gpu_id]          # default GPU 0
#   BATCH_SIZE=32 GRAD_ACCUM=8 bash scripts/train_r1_robotcut.sh 0   # 24 GB card
#
# Needs R3D/data/real_bimanual_mixed150_robotcut_default.zarr next to the
# dataset (scripts/precompute_robot_cut.py; the dataset refuses a missing,
# incomplete or param-mismatched one).
set -euo pipefail
cd "$(dirname "$0")/.."
gpu=${1:-0}
labels=R3D/data/real_bimanual_mixed150_robotcut_default.zarr
[ -d "$labels" ] || { echo "missing $labels -- run scripts/precompute_robot_cut.py" >&2; exit 1; }

env BATCH_SIZE="${BATCH_SIZE:-256}" GRAD_ACCUM="${GRAD_ACCUM:-1}" \
    CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-150}" WANDB_MODE="${WANDB_MODE:-online}" \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    bash scripts/train_real_bimanual.sh real_bimanual_robotcut 52 "$gpu" \
    training.num_epochs=301 training.val_every=10 \
    dataloader.num_workers=8 val_dataloader.num_workers=1 \
    dataloader.persistent_workers=True val_dataloader.persistent_workers=True
