#!/usr/bin/env bash
# Train the sim baseline (CLAUDE.md "Sim baseline") in the r3d_sim env.
#
#   bash scripts/train_robotwin2_sim.sh <robotwin_task> <seed> <gpu_id> [hydra overrides...]
#   bash scripts/train_robotwin2_sim.sh place_empty_cup_right 42 0
#   SETTING=sim_stock_p1024 bash scripts/train_robotwin2_sim.sh stack_blocks_two_fov 42 0
#   TAG=g64 bash scripts/train_robotwin2_sim.sh place_empty_cup_right 42 0 task.num_group=64
#   TASK_CFG=robotwin2_sim EPOCHS=1001 bash scripts/train_robotwin2_sim.sh ...
#
# No in-loop rollouts (rollout_every > epochs): RoboTwin rollouts block training.
# Score checkpoints afterwards with scripts/eval_robotwin2_ckpt.py.
#
# run_dir is built from task config + RoboTwin task + TAG + seed, NOT from the
# hydra overrides -- two variants need different TAGs (or seeds), or the second
# overwrites the first one's checkpoints.

set -euo pipefail

robotwin_task=${1:?usage: train_robotwin2_sim.sh <robotwin_task> <seed> <gpu_id> [overrides]}
seed=${2:?missing seed}
gpu_id=${3:?missing gpu id}
extra_args=("${@:4}")

task_cfg=${TASK_CFG:-robotwin2_sim}
setting=${SETTING:-sim_head_p1024}      # RoboTwin task config: collection == evaluation
alg_name=${ALG:-r3d_robotwin2}
# 601, not 600: train.py counts epochs 0..N-1 and saves at epoch % checkpoint_every
# == 0, so N=600 never writes 600.ckpt (last is 500) and the final weights are
# lost. The real-data runs use 601 for the same reason.
epochs=${EPOCHS:-601}
checkpoint_every=${CHECKPOINT_EVERY:-100}
# Effective batch 256 = R3D's RoboTwin setting. A micro-batch of 256 does NOT fit
# a 24 GB card even at 1024 points / 128 tokens (OOM at 18.6 GiB in patch_embed,
# 2026-10-17); train.py scales the loss by 1/accum and steps optimizer + LR
# schedule on accumulation boundaries, so 64 x 4 is the same run.
batch_size=${BATCH_SIZE:-64}
grad_accum=${GRAD_ACCUM:-4}
wandb_mode=${WANDB_MODE:-offline}
# TAG names the variant (e.g. g64) -- without it, variants at one seed share a run_dir
exp_name=${task_cfg}-${robotwin_task}${TAG:+-${TAG}}-${seed}

eval "$(micromamba shell hook -s bash)"
micromamba activate r3d_sim
unset PYTHONPATH   # after activation -- see CLAUDE.md "Environment"

export HYDRA_FULL_ERROR=1
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=${gpu_id}

cd "$(dirname "$0")/../R3D"
run_dir="$(pwd)/data/outputs/sim/${exp_name}"

python train.py \
    --config-name="${alg_name}.yaml" \
    task="${task_cfg}" \
    task_name="${robotwin_task}" \
    setting="${setting}" \
    hydra.run.dir="${run_dir}" \
    training.seed="${seed}" \
    training.device="cuda:0" \
    training.use_ddp=false \
    training.num_epochs="${epochs}" \
    training.checkpoint_every="${checkpoint_every}" \
    training.rollout_every=$((epochs + 1)) \
    training.gradient_accumulate_every="${grad_accum}" \
    dataloader.batch_size="${batch_size}" \
    val_dataloader.batch_size="${batch_size}" \
    exp_name="${exp_name}" \
    logging.mode="${wandb_mode}" \
    checkpoint.save_ckpt=True \
    "${extra_args[@]}"
