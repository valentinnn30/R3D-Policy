#!/usr/bin/env bash
# Train the 2D DINOv2 baseline on real bimanual Franka data.
#
# A thin wrapper around train_real_bimanual.sh -- everything it does (env
# guard, DDP setup, run_dir naming, hydra passthrough) is identical, and
# duplicating it would give two scripts to keep in step. The only difference is
# the algorithm config.
#
# The three planned runs, all from this one file:
#
#   2D-N  headline   bash scripts/train_real_bimanual_2d.sh real_bimanual_2d 42 0
#   2D-A' ablation   bash scripts/train_real_bimanual_2d.sh real_bimanual_2d 43 0 \
#                        task.dataset.pose_source=proprio
#   2D-B  ablation   bash scripts/train_real_bimanual_2d.sh real_bimanual_2d 44 0 \
#                        task.dataset.pose_source=extrinsic
#   2D-U  ablation   bash scripts/train_real_bimanual_2d.sh real_bimanual_2d 45 0 \
#                        policy.image_encoder_cfg.freeze=false \
#                        policy.image_encoder_cfg.train_last_n_blocks=3 \
#                        policy.image_encoder_cfg.drop_path_rate=0.2
#
# 2D-U unfreezes the top of the DINOv2 trunk. The headline run's trainable
# visual pathway is 0.099 M parameters against the 3D encoder's 6.087 M, so
# 2D-N vs the 3D run is a frozen-vs-finetuned comparison, not an
# encoder-vs-encoder one. Unfreezing the last 3 blocks + norm gives 5.326 M,
# within 3% of Uni3D's 5.502 M transformer -- that is where the 3 comes from.
#
# NOTHING ELSE CHANGES. lr stays 1.0e-4 flat over every trainable parameter,
# weight_decay 1.0e-6, lr_warmup_steps 500 -- the 3D run fine-tunes its own
# PRETRAINED Uni3D trunk on exactly those (train.py:110 builds one flat param
# group), so a discriminative backbone LR would be a second, unjustified
# difference. drop_path_rate 0.2 is likewise the Uni3D trunk's own value.
#
# Costs no wall-clock: measured on the 4090 at batch 64 / bf16 the GPU step
# goes 0.48 -> 0.77 ms/sample while the dataloader delivers only
# 2.70 ms/sample, so training stays dataloader-bound. Peak VRAM 1.74 -> 3.53
# GiB. Re-time before raising num_workers.
#
# The headline carries NO pose tag: `agent_pos` already holds both arms' EE
# poses, and the camera pose is that pose composed with a constant, so the tag
# adds routing rather than information. See the task config's header.
#
# Note the differing SEEDS above, and not for statistical reasons: run_dir is
# derived from exp_name, which carries the seed, so two variants of one task
# config that share a seed will overwrite each other's checkpoints.
#
# Scale-up (2D-L), when the point trunk scales too:
#   ... policy.image_encoder_cfg.model_name=vit_large_patch14_reg4_dinov2.lvd142m
#
# GRAD_ACCUM defaults to 4 in the parent, i.e. an effective batch of 256 at
# BATCH_SIZE=64 -- the same effective batch the point-cloud runs use. The
# encoder is frozen, so activations for the trunk are not retained and 64 is
# comfortable; raise GRAD_ACCUM rather than lowering BATCH_SIZE if it is not.

set -euo pipefail
exec env ALG="${ALG:-r3d_2d}" bash "$(dirname "$0")/train_real_bimanual.sh" "$@"
