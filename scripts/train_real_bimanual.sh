#!/usr/bin/env bash
# Train R3D on real bimanual Franka data (no simulator, no rollout metrics).
#
# Examples:
#   bash scripts/train_real_bimanual.sh real_bimanual 42 0
#   bash scripts/train_real_bimanual.sh real_bimanual 42 0,1        # DDP
#   EPOCHS=1200 bash scripts/train_real_bimanual.sh real_bimanual 42 0
#   BATCH_SIZE=64 GRAD_ACCUM=4 bash scripts/train_real_bimanual.sh ... # if OOM
#
# Args: <task_config_name> <seed> <gpu_id[,gpu_id...]>
#
# The defaults below are sized for real data, not for the sim benchmarks the
# repo ships with: r3d_robotwin2.yaml defaults to 3001 epochs with
# rollout_every 200, which only makes sense when a simulator is scoring
# rollouts. Here `env_runner` is null, so test_mean_score is just -train_loss
# and checkpoints are what you actually keep.

set -euo pipefail

task_config=${1:?usage: train_real_bimanual.sh <task_config> <seed> <gpu_id[,gpu_id]>}
seed=${2:?missing seed}
gpu_id=${3:?missing gpu id}

alg_name=${ALG:-r3d_robotwin2}
# The R3D paper trains for 1000 epochs. The config's own 3001 is the outlier --
# it is sized around `rollout_every: 200` sim evaluations we do not run.
epochs=${EPOCHS:-601}
checkpoint_every=${CHECKPOINT_EVERY:-100}
# 256 is the paper's RoboTwin setting, which uses this same policy config, so it
# is the tested pairing with lr 1e-4 / 1000 epochs / cosine. (They use 2048 for
# ManiSkill; no real-world figure is published.) Batch size and epoch count
# interact through total optimizer steps -- do not change one alone.
#
# Point count is NOT the reason to lower this: `Group` reduces any cloud to
# num_group(512) x group_size(32), so the Uni3D transformer and the 250M-param
# UNet are independent of it. Only FPS/kNN scale with N.
#
# If 256 does not fit in 24 GB, raise GRAD_ACCUM rather than lowering BATCH_SIZE:
# train.py scales the loss by 1/accum and steps the optimizer AND the LR
# scheduler only on accumulation boundaries, so the effective batch and the
# cosine schedule are preserved. BATCH_SIZE=64 GRAD_ACCUM=4 == batch 256.
batch_size=${BATCH_SIZE:-64}
grad_accum=${GRAD_ACCUM:-4}
wandb_mode=${WANDB_MODE:-offline}
save_ckpt=${SAVE_CKPT:-True}
# Any 4th and later argument is passed straight to hydra, e.g.
#   ... real_bimanual_unfused 42 0 task.dataset.pose_source=proprio
# Without this, such an override is SILENTLY DROPPED and you get the
# un-overridden run under the right-looking name.
#
# `run_dir` is derived from `exp_name`, which carries the seed -- so two
# variants of the same task config must differ in seed, or the second will
# overwrite the first one's checkpoints.
extra_args=("${@:4}")
exp_name=${task_config}-${alg_name}-${seed}

# The ros2_ws stack exports a PYTHONPATH pointing at ros_env's python3.12
# site-packages; on this python3.10 interpreter that produces undefined-symbol
# crashes far from the cause. `micromamba activate` re-introduces it, so this
# has to come after activation, not before -- the env's activate.d guard does
# the same thing for interactive shells.
unset PYTHONPATH

export HYDRA_FULL_ERROR=1
export TOKENIZERS_PARALLELISM=false

echo -e "\033[33mgpu id (to use): ${gpu_id}\033[0m"
if [[ $gpu_id == *","* ]]; then
    IFS=',' read -ra GPU_ARRAY <<< "$gpu_id"
    num_gpus=${#GPU_ARRAY[@]}
    echo -e "\033[32mMulti-GPU DDP mode: ${num_gpus} GPUs\033[0m"
    export CUDA_VISIBLE_DEVICES=${gpu_id}
    export WORLD_SIZE=${num_gpus}
    export MASTER_ADDR="localhost"
    export MASTER_PORT="12358"
    export NCCL_IB_DISABLE=1
    export NCCL_P2P_DISABLE=1
    USE_DDP=true
else
    echo -e "\033[32mSingle GPU mode\033[0m"
    export CUDA_VISIBLE_DEVICES=${gpu_id}
    USE_DDP=false
fi

cd "$(dirname "$0")/../R3D"
run_dir="$(pwd)/data/outputs/${exp_name}_seed${seed}"

common_args=(
    --config-name="${alg_name}.yaml"
    task="${task_config}"
    task_name="${task_config}"
    hydra.run.dir="${run_dir}"
    training.seed="${seed}"
    training.num_epochs="${epochs}"
    training.checkpoint_every="${checkpoint_every}"
    training.gradient_accumulate_every="${grad_accum}"
    dataloader.batch_size="${batch_size}"
    val_dataloader.batch_size="${batch_size}"
    exp_name="${exp_name}"
    logging.mode="${wandb_mode}"
    checkpoint.save_ckpt="${save_ckpt}"
)

if [ $USE_DDP = true ]; then
    torchrun --nproc_per_node="${num_gpus}" --master_port=12358 \
        train.py "${common_args[@]}" training.device="cuda" training.use_ddp=true \
        "${extra_args[@]}"
else
    python train.py "${common_args[@]}" training.device="cuda:0" \
        training.use_ddp=false "${extra_args[@]}"
fi
