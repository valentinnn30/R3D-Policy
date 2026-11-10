#!/usr/bin/env bash
# New task: MimicGen Threading D1 with the tripod window narrowed (r3d/common/robosuite_tasks.py WINDOWS
# 'threading_d1n'; docs/sim_token_allocation_learnings.md "Next task candidate: MimicGen Threading").
# 4096 points / 64 tokens, the FIRST 200 demos (file order) inside the window, ids in the zarr attrs. Strictly one
# GPU job at a time, detached:
#
#   setsid nohup bash scripts/sim/run_threading_chain.sh > R3D/data/outputs/sim/mimicgen/threading/chain.log 2>&1 &
#
#   0 wait until no training / eval / sweep driver has run for 10 min
#   1 convert (re-render 200 demos) + action-replay check (--verify 5) + link labels (bit-identity checked)
#   2 data check: needle / tripod points per frame, window of every demo
#   3 smoke: uncut and GObal, 2 short epochs each + 2-scene eval (narrowed eval env); smoke dirs deleted
#   4 uncut (seed 42) -> eval 250; arm cut [5] -> eval 250; GObal (sweep driver: train, eval 250, then 200)
#     GObal with gates for ALL arm links (region_groups=all_links: link0..link4 + the 8 default regions)
#   5 eval 200 of uncut and arm cut
# All runs: 251 epochs, 64 x 2, seed 42; the plain runs write only 200.ckpt + 250.ckpt.
# Any failure stops the chain (set -e). Steps whose output exists are skipped.
set -euo pipefail
here="$(cd "$(dirname "$0")/../.." && pwd)"
cd "${here}"
out=R3D/data/outputs/sim/mimicgen/threading
mkdir -p "${out}"
log() { echo "[$(date '+%F %T')] $*"; }
env_on() { eval "$(micromamba shell hook -s bash)"; micromamba activate "$1"; unset PYTHONPATH; }
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
task=threading_d1n
setting=agentview_p4096-200
zarr=R3D/data/${task}-${setting}.zarr
sim=R3D/data/outputs/sim
rd() { echo "${sim}/mimicgen_sim-${task}-p4096-g64$1-42"; }

busy() { ps -eo cmd | awk '($1=="bash" && $2 ~ /run_round[0-9]+_chain|run_remover_sweep|run_finetune_branches|run_mask_override|run_branch_test|run_early_branch|run_cut_keep7|run_cut_pair/) || ($1 ~ /python/ && ($0 ~ /train\.py/ || $0 ~ /eval_mimicgen_ckpt/ || $0 ~ /convert_mimicgen_to_zarr/ || $0 ~ /add_link_labels/))' | grep -q .; }
log "waiting for the GPU to be free for 10 min"
free=0
while [ "${free}" -lt 10 ]; do
    if busy; then free=0; else free=$((free + 1)); fi
    sleep 60
done
log "GPU free for 10 min"

# ---- 1 data ---------------------------------------------------------------------------------
if [ ! -f "${zarr}/.zattrs" ] && [ ! -f "${zarr}/meta/.zattrs" ]; then
    log "convert"
    ( env_on r3d_robosuite; python -W ignore scripts/sim/convert_mimicgen_to_zarr.py --task threading_d1 \
        --window threading_d1n --demos 200 --workers 6 --output "${zarr}" ) > "${out}/convert.log" 2>&1
    grep -E "^window|^rendered|^wrote" "${out}/convert.log"
fi
if [ ! -f "${out}/verify.ok" ]; then
    log "action replay check"
    ( env_on r3d_robosuite; python -W ignore scripts/sim/convert_mimicgen_to_zarr.py --task threading_d1 --verify 5 ) \
        > "${out}/verify.log" 2>&1
    grep -E "^demo|^replayed" "${out}/verify.log"
    grep -q "^replayed success 5/5" "${out}/verify.log" || { log "action replay failed"; exit 1; }
    touch "${out}/verify.ok"
fi
if [ ! -d "${zarr}/data/point_link" ]; then
    log "link labels"
    ( env_on r3d_robosuite; python -W ignore scripts/sim/add_link_labels_mimicgen.py --zarr "${zarr}" --workers 6 ) \
        > "${out}/link_labels.log" 2>&1
    tail -3 "${out}/link_labels.log"
    [ -d "${zarr}/data/point_link" ] || { log "no point_link written"; exit 1; }
fi

# ---- 2 data check ---------------------------------------------------------------------------
if [ ! -f "${out}/data_check.ok" ]; then
    log "data check"
    ( env_on r3d_sim; python - "${zarr}" <<'EOF'
import sys, numpy as np, zarr
z = zarr.open(sys.argv[1], "r")
a = dict(z["meta"].attrs)
lab, ends = z["data/point_label"], z["meta/episode_ends"][:]
print("demos", len(a["demos"]), "first", a["demos"][:5], "last", a["demos"][-1], "window", a["init_window"], "eval env", a["eval_env_name"])
print("frames", int(ends[-1]), "points in box min", a["points_in_box_min"], "p10", a["points_in_box_p10"])
mn = {8: 10 ** 9, 9: 10 ** 9}; mean = {8: 0.0, 9: 0.0}; n = 0
for s in range(0, lab.shape[0], 5000):
    L = lab[s:s + 5000]
    for c in (8, 9):
        k = (L == c).sum(1); mn[c] = min(mn[c], int(k.min())); mean[c] += float(k.sum())
    n += len(L)
print({("needle", "tripod")[c - 8]: dict(min=mn[c], mean=round(mean[c] / n)) for c in (8, 9)})
lay = z["meta/init_layout"][:]
if not (len(lay) == len(a["demos"]) == len(ends) == 200):
    raise SystemExit("demo count mismatch")
if mn[9] == 0:
    raise SystemExit("a frame has no tripod point")
EOF
    ) > "${out}/data_check.log" 2>&1
    cat "${out}/data_check.log"
    touch "${out}/data_check.ok"
fi

# ---- 3 smoke --------------------------------------------------------------------------------
smoke() {  # name alg task_cfg overrides...
    local name=$1 alg=$2 tcfg=$3; shift 3
    local run="${sim}/${tcfg}-${task}-${name}-42"
    rm -rf "${run}" "${run}.train.log"
    ALG=${alg} TASK_CFG=${tcfg} SETTING=${setting} EPOCHS=2 CHECKPOINT_EVERY=125 BATCH_SIZE=64 GRAD_ACCUM=2 \
        TAG="${name}" bash scripts/train_robotwin2_sim.sh "${task}" 42 0 task.num_group=64 task.num_points=4096 \
        '+training.checkpoint_epochs=[1]' training.max_train_steps=6 training.max_val_steps=3 "$@" > "${run}.train.log" 2>&1
    [ "$(ls "${run}/checkpoints")" = "1.ckpt" ] || { log "smoke ${name}: expected exactly 1.ckpt"; exit 1; }
    ( env_on r3d_robosuite; python -W ignore scripts/sim/eval_mimicgen_ckpt.py "${run}/checkpoints/1.ckpt" \
        --episodes 2 --workers 1 ) > "${out}/smoke_${name}_eval.log" 2>&1
    grep "^ckpt:" "${out}/smoke_${name}_eval.log"
    cp "${run}.train.log" "${out}/smoke_${name}_train.log"
    rm -rf "${run}" "${run}.train.log"
}
if [ ! -f "${out}/smoke.ok" ]; then
    log "smoke runs"
    smoke p4096-g64-smoke r3d_robotwin2 mimicgen_sim
    smoke p4096-g64-rm-GObalsmoke r3d_robotwin2_remover mimicgen_sim_remover \
        policy.link_remover.learn_density=false policy.link_remover.init_keep=0.5 policy.link_remover.theta_keep=0.5 policy.link_remover.region_groups=all_links
    log "smoke ok, smoke run dirs deleted"; touch "${out}/smoke.ok"
fi

# ---- 4 runs ---------------------------------------------------------------------------------
train() {  # suffix overrides...
    local sfx=$1 run; shift; run=$(rd "${sfx}")
    [ -f "${run}/checkpoints/250.ckpt" ] && return 0
    log "train $(basename "${run}") $*"
    TASK_CFG=mimicgen_sim SETTING=${setting} EPOCHS=251 CHECKPOINT_EVERY=125 BATCH_SIZE=64 GRAD_ACCUM=2 \
        TAG="p4096-g64${sfx}" bash scripts/train_robotwin2_sim.sh "${task}" 42 0 \
        task.num_group=64 task.num_points=4096 '+training.checkpoint_epochs=[200,250]' "$@" > "${run}.train.log" 2>&1
    log "train exit: $(ls "${run}/checkpoints" | tr '\n' ' ')"
}
evaluate() {  # suffix epoch
    local run; run=$(rd "$1")
    [ -f "${run}/eval/$2_200ep_block1.json" ] && return 0
    log "eval $(basename "${run}") $2"
    ( env_on r3d_robosuite; python -W ignore scripts/sim/eval_mimicgen_ckpt.py "${run}/checkpoints/$2.ckpt" \
        --episodes 200 --workers 5 ) >> "${run}/eval.log" 2>&1
    log "eval $2: $(grep '^ckpt:' "${run}/eval.log" | tail -1 | sed 's|.*checkpoints/||; s|  [0-9.]* min.*||')"
}
train "";      evaluate "" 250
train "-cut5" "task.drop_point_labels=[5]"; evaluate "-cut5" 250
queue="${out}/gobal_queue.txt"
echo "4096 42 GObalAll policy.link_remover.learn_density=false policy.link_remover.init_keep=0.5 policy.link_remover.theta_keep=0.5 policy.link_remover.region_groups=all_links" > "${queue}"
log "GObalAll (sweep driver)"
TASK=${task} SETTING=${setting} KEEP_ONLY_EVALUATED=1 MID_CKPTS="200" SWEEP_OUT="${out}/gobal_sweep" \
    bash scripts/sim/run_remover_sweep.sh "${queue}" > "${out}/gobal_driver.log" 2>&1
grep -E "train|eval" "${out}/gobal_driver.log" | tail -4

# ---- 5 ---------------------------------------------------------------------------------------
evaluate "" 200
evaluate "-cut5" 200
log "threading chain done"
