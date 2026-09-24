#!/usr/bin/env bash
# Launch all six fingertip anchor-set variants (A B C D F K) at once, three per
# GPU, each as an independent single-GPU run of train_real_bimanual.sh.
#
#   micromamba activate r3d && unset PYTHONPATH
#   cd ~/R3D-Policy
#   DRY_RUN=1 bash scripts/sweep_fingertip_variants.sh     # print, launch nothing
#   bash scripts/sweep_fingertip_variants.sh               # GPUs 0 and 1
#   GPUS="2 3" bash scripts/sweep_fingertip_variants.sh
#   OVERWRITE=1 bash scripts/sweep_fingertip_variants.sh   # restart over old run_dirs
#
# The variants and what each one asks: "Fingertip anchor-set variants" in
# CLAUDE.md.
#
# Split: F and K are the heaviest (0.36 GiB/sample vs 0.33) and are each other's
# control, so they go on DIFFERENT GPUs -- otherwise one card carries both big
# runs and finishes last. GPU a: A B F, GPU b: C D K.
#
# Batch 64 / accum 4, not the 32 / 8 the notes give for a 24 GB card: same
# effective batch 256, same LR schedule (train.py steps optimizer and scheduler
# on accumulation boundaries), just fewer, larger steps. Measured 0.33-0.36
# GiB/sample -> ~23-24 GiB per run, ~72 GiB per H200 for three.
#
# SEED 52 is the reference run's (fingertip_150/300.ckpt: seed 52, 601 epochs),
# so every variant differs from the reference in the anchor set only. The six
# task names differ, so their run_dirs cannot collide at one seed.
#
# The node's CPU and RAM are SHARED with other users, and no cap is applied, so
# the budget is held by the settings: 4+1 dataloader workers and 2 math threads
# per run, ~30 cores in total. The zarr is read through the OS page cache
# (create_from_path), so the six runs share one cached copy.

set -euo pipefail

variants=(A B F C D K)                # first three -> GPU a, last three -> GPU b
read -ra gpus <<< "${GPUS:-0 1}"
seed=${SEED:-52}
batch_size=${BATCH_SIZE:-64}
grad_accum=${GRAD_ACCUM:-4}
workers=${NUM_WORKERS:-4}
val_workers=${VAL_NUM_WORKERS:-1}
threads=${THREADS:-2}
alg_name=${ALG:-r3d_robotwin2}
# train_real_bimanual.sh defaults to offline; the sweep logs live.
wandb_mode=${WANDB_MODE:-online}
dry_run=${DRY_RUN:-0}
overwrite=${OVERWRITE:-0}
# 300 + 601 epochs -> exactly 300.ckpt and 600.ckpt per run (~2.1 GB each,
# ~25 GB for the sweep); train.py skips epoch 0.
checkpoint_every=${CHECKPOINT_EVERY:-300}

cd "$(dirname "$0")/.."
repo=$(pwd)

if [ ${#gpus[@]} -ne 2 ]; then
    echo "GPUS must name exactly two GPUs, got: '${gpus[*]}'" >&2; exit 1
fi

# --- preflight ---------------------------------------------------------------
fail=0
for v in "${variants[@]}"; do
    cfg="R3D/r3d/config/task/real_bimanual_fingertip_${v}.yaml"
    [ -f "$cfg" ] || { echo "missing task config: $cfg" >&2; fail=1; }
done

# An existing run_dir is NOT resumed: train.py resumes only from latest.ckpt,
# which this setup never writes. A rerun starts from epoch 0 and overwrites
# that run's 300/600.ckpt, so refuse unless asked.
for v in "${variants[@]}"; do
    task="real_bimanual_fingertip_${v}"
    run_dir="R3D/data/outputs/${task}-${alg_name}-${seed}_seed${seed}"
    if [ -d "$run_dir/checkpoints" ] && [ "$overwrite" != 1 ]; then
        echo "$run_dir already has checkpoints -- a rerun would OVERWRITE them." >&2
        echo "  Move it away, pick another SEED, or OVERWRITE=1." >&2
        fail=1
    fi
done

if [ "$dry_run" != 1 ]; then
    python -c "import torch, hydra" 2>/dev/null \
        || { echo "python cannot import torch/hydra -- activate the r3d env" >&2; fail=1; }
    # A detached run cannot answer wandb's login prompt: it would fail or
    # stall at init, six times over. Check the key exists before launching.
    if [ "$wandb_mode" = online ] && [ -z "${WANDB_API_KEY:-}" ] \
            && ! grep -q "api.wandb.ai" ~/.netrc 2>/dev/null; then
        echo "wandb online but not logged in -- run 'wandb login' (or WANDB_MODE=offline)" >&2
        fail=1
    fi
    n_gpu=$(nvidia-smi -L 2>/dev/null | wc -l)
    for g in "${gpus[@]}"; do
        [ "$g" -lt "$n_gpu" ] || { echo "GPU $g not present ($n_gpu visible)" >&2; fail=1; }
    done
fi

# The dataset reads off disk and needs one chunk per frame, else every sample
# decompresses ~50x what it uses. A copy made from the old 100-frame zarr still
# trains correctly, just CPU-bound -- so warn, do not refuse.
zarr=$(awk '/zarr_path:/ {print $2; exit}' R3D/r3d/config/task/real_bimanual_fingertip.yaml)
for f in "$zarr"/data/point_cloud_cam*/.zarray; do
    [ -f "$f" ] || { echo "WARNING: no point clouds found under $zarr" >&2; break; }
    if ! grep -A1 '"chunks"' "$f" | grep -q '^ *1,$'; then
        echo "WARNING: $f is not one chunk per frame -- run scripts/rechunk_zarr_per_frame.py" >&2
    fi
done

[ $fail = 0 ] || exit 1

# --- launch ------------------------------------------------------------------
stamp=$(date +%Y%m%d_%H%M%S)
log_dir="$repo/logs/sweep_fingertip_${stamp}"
[ "$dry_run" = 1 ] || mkdir -p "$log_dir"

for i in "${!variants[@]}"; do
    v=${variants[$i]}
    gpu=${gpus[$(( i / 3 ))]}
    task="real_bimanual_fingertip_${v}"
    cmd=(env OMP_NUM_THREADS="$threads" MKL_NUM_THREADS="$threads"
         BATCH_SIZE="$batch_size" GRAD_ACCUM="$grad_accum" ALG="$alg_name"
         WANDB_MODE="$wandb_mode" CHECKPOINT_EVERY="$checkpoint_every"
         PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
         bash scripts/train_real_bimanual.sh "$task" "$seed" "$gpu"
         dataloader.num_workers="$workers"
         val_dataloader.num_workers="$val_workers")
    if [ "$dry_run" = 1 ]; then
        echo "[GPU $gpu] ${cmd[*]}"
        continue
    fi
    # setsid + nohup: the runs outlive this shell and an SSH disconnect.
    setsid nohup "${cmd[@]}" > "$log_dir/${v}.log" 2>&1 < /dev/null &
    echo "$v $gpu $!" >> "$log_dir/pids"
    echo "[GPU $gpu] $task  pid $!  log $log_dir/${v}.log"
done

[ "$dry_run" = 1 ] && exit 0
cat <<EOF

Six runs launched. Logs and PIDs: $log_dir
  tail -f $log_dir/*.log
  watch -n 5 nvidia-smi          # utilisation near 100% = GPU-bound, as planned
Stop all six:
  while read v g p; do kill -- -\$p; done < $log_dir/pids
EOF
