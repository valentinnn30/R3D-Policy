#!/usr/bin/env bash
# Fused baseline + camera one-hot, 512 tokens ("K-512", 2026-10-07,
# ~/ros2_ws/robot_cut_r1_ablations.txt) with EXACTLY the settings K and R1 ran
# with: seed 52, batch 256 x accum 1, 301 epochs (cosine to 0 there), checkpoint
# every 150, val every 10, 8 + 1 persistent workers, wandb online. Only the task
# differs.
#
#   bash scripts/train_fused_onehot.sh [gpu_id]          # default GPU 0
#   BATCH_SIZE=32 GRAD_ACCUM=8 bash scripts/train_fused_onehot.sh 0   # 24 GB card
set -euo pipefail
cd "$(dirname "$0")/.."
gpu=${1:-0}

env BATCH_SIZE="${BATCH_SIZE:-256}" GRAD_ACCUM="${GRAD_ACCUM:-1}" \
    CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-150}" WANDB_MODE="${WANDB_MODE:-online}" \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    bash scripts/train_real_bimanual.sh real_bimanual_onehot 52 "$gpu" \
    training.num_epochs=301 training.val_every=10 \
    dataloader.num_workers=8 val_dataloader.num_workers=1 \
    dataloader.persistent_workers=True val_dataloader.persistent_workers=True
