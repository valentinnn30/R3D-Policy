#!/usr/bin/env bash
# Threading D1 narrowed at 1024 points / 64 tokens (user, 2026-11-12): completes the point-count comparison. On Square
# and hanging_mug 1024/64 uncut was the best plain policy; on Threading the arm cut did not help at 4096 (37.5 uncut vs
# 35.0 cut). If 1024 helps here, sparser clouds and the cut are separate phenomena. Recipe = the Threading 4096 uncut
# run except num_points (the first 1024 points of the stored FPS-ordered cloud; eval builds the 1024 cloud directly):
# same zarr, seed 42, 251 epochs, 64 x 2, ckpt 250 only, 200 scenes of seed block 1 (Threading_D1_Narrow).
#
#   setsid nohup bash scripts/sim/run_thr_p1024.sh > R3D/data/outputs/sim/mimicgen/threading/p1024_chain.log 2>&1 &
#
# Named so that it does NOT match the busy patterns of run_table_eval.sh (`run_threading`), which would deadlock.
# Waits until the token-dropout and table-eval chains have exited and the GPU has been free for 10 min.
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
S=R3D/data/outputs/sim
run=${S}/mimicgen_sim-${task}-p1024-g64-42

busy() { ps -eo cmd | awk '($1=="bash" && $2 ~ /run_round[0-9]+_chain|run_remover_sweep|run_finetune_branches|run_mask_override|run_branch_test|run_early_branch|run_cut_keep7|run_cut_pair|run_mug_drop|run_threading|run_token_dropout|run_table_eval/) || ($1 ~ /python/ && ($0 ~ /train\.py/ || $0 ~ /eval_mimicgen_ckpt/ || $0 ~ /eval_robotwin2_ckpt/))' | grep -q .; }
log "waiting for the GPU to be free for 10 min (after the token-dropout and table-eval chains)"
free=0
while [ "${free}" -lt 10 ]; do
    if busy; then free=0; else free=$((free + 1)); fi
    sleep 60
done
log "GPU free for 10 min; disk free: $(df -h --output=avail R3D/data | tail -1)"

if [ ! -f "${run}/checkpoints/250.ckpt" ]; then
    log "train threading p1024 g64 seed 42"
    TASK_CFG=mimicgen_sim SETTING=${setting} EPOCHS=251 CHECKPOINT_EVERY=125 BATCH_SIZE=64 GRAD_ACCUM=2 \
        TAG="p1024-g64" bash scripts/train_robotwin2_sim.sh "${task}" 42 0 \
        task.num_group=64 task.num_points=1024 '+training.checkpoint_epochs=[250]' > "${run}.train.log" 2>&1
    log "train exit: $(ls "${run}/checkpoints" | tr '\n' ' ')"
fi
if [ ! -f "${run}/eval/250_200ep_block1.json" ]; then
    log "eval 250"
    ( env_on r3d_robosuite; python -W ignore scripts/sim/eval_mimicgen_ckpt.py "${run}/checkpoints/250.ckpt" \
        --episodes 200 --workers 5 ) >> "${run}/eval.log" 2>&1
    log "  $(grep '^ckpt:' "${run}/eval.log" | tail -1 | sed 's|.*checkpoints/||; s|  [0-9.]* min.*||')"
fi

python3 - "${run}" > "${out}/p1024_report.txt" <<'PY'
import json, math, sys, pathlib
S = pathlib.Path("R3D/data/outputs/sim")
def load(p):
    return {s["seed"]: bool(s["success"]) for s in json.load(open(p))["scenes"]}
def mcnemar(b, c):
    n = b + c
    return 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n)
a = load(pathlib.Path(sys.argv[1]) / "eval/250_200ep_block1.json")
print(f"threading 1024/64 uncut @250: {100 * sum(a.values()) / len(a):.1f}%")
for name, r in (("4096/64 uncut", "mimicgen_sim-threading_d1n-p4096-g64-42"),
                ("4096/64 arm cut", "mimicgen_sim-threading_d1n-p4096-g64-cut5-42"),
                ("4096/64 GObalAll", "mimicgen_sim_remover-threading_d1n-p4096-g64-rm-GObalAll-42")):
    f = S / r / "eval/250_200ep_block1.json"
    if not f.exists(): continue
    b = load(f); k = sorted(set(a) & set(b))
    x = sum(a[s] and not b[s] for s in k); y = sum(b[s] and not a[s] for s in k)
    print(f"  vs {name:17s} {100 * sum(b[s] for s in k) / len(k):5.1f}%  1024 only / other only {x} / {y}, "
          f"p = {mcnemar(x, y):.3f}  (n = {len(k)})")
PY
cat "${out}/p1024_report.txt"
log "threading p1024 done"
