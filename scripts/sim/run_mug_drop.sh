#!/usr/bin/env bash
# hanging_mug robot-part DROPOUT (the Square DROP / GObal mechanism, class level -- GObal itself needs per-link labels
# the RoboTwin data does not have): 4096 points / 64 tokens, seed 42, 601 epochs, 64 x 4, ckpts 300 + 600 -- the
# arm cut run's exact recipe (run_mug_sweep.sh) plus task.dropout_point_labels=[4,5,6,7] (keep 0.5 per class per
# window, training only; validation and evaluation see the full cloud). Then all three 600.ckpts -- uncut, arms
# cut [4,5], dropout -- on the CURRENT shared 200-scene list (scene_seed_group sim_stock; the old 4096 evals used
# their own list, 180 of 200 shared), tag "shared", and a paired report. Detached:
#
#   setsid nohup bash scripts/sim/run_mug_drop.sh > R3D/data/outputs/sim/mug_drop/chain.log 2>&1 &
set -euo pipefail
here="$(cd "$(dirname "$0")/../.." && pwd)"
cd "${here}"
sim=R3D/data/outputs/sim
out=${sim}/mug_drop
mkdir -p "${out}"
log() { echo "[$(date '+%F %T')] $*"; }
env_on() { eval "$(micromamba shell hook -s bash)"; micromamba activate "$1"; unset PYTHONPATH; }
task=hanging_mug_labeled
rd() { echo "${sim}/robotwin2_sim-${task}-$1-42"; }
drop="task.dropout_point_labels=[4,5,6,7]"

busy() { ps -eo cmd | awk '($1=="bash" && $2 ~ /run_threading|run_remover_sweep|run_cut_pair|run_round[0-9]+_chain/) || ($1 ~ /python/ && ($0 ~ /train\.py/ || $0 ~ /eval_mimicgen_ckpt/ || $0 ~ /eval_robotwin2_ckpt/))' | grep -q .; }
log "waiting for the GPU to be free for 10 min"
free=0
while [ "${free}" -lt 10 ]; do
    if busy; then free=0; else free=$((free + 1)); fi
    sleep 60
done
log "GPU free for 10 min"

if [ ! -f "${out}/checks.ok" ]; then
    log "checks"
    ( env_on r3d_sim; python -W ignore scripts/sim/check_label_dropout.py ) > "${out}/check_label_dropout.log" 2>&1
    tail -1 "${out}/check_label_dropout.log"
    touch "${out}/checks.ok"
fi

if [ ! -f "${out}/smoke.ok" ]; then
    log "smoke"
    smoke=$(rd p4096-g64-dropsmoke)
    rm -rf "${smoke}"
    TAG=p4096-g64-dropsmoke SETTING=sim_stock_p4096 EPOCHS=2 CHECKPOINT_EVERY=300 BATCH_SIZE=64 GRAD_ACCUM=4 \
        bash scripts/train_robotwin2_sim.sh "${task}" 42 0 task.num_points=4096 task.num_group=64 "${drop}" \
        '+training.checkpoint_epochs=[1]' training.max_train_steps=6 training.max_val_steps=3 > "${out}/smoke_train.log" 2>&1
    [ "$(ls "${smoke}/checkpoints" | grep -c '\.ckpt$')" = 1 ] || { log "smoke: expected exactly 1.ckpt"; exit 1; }
    ( env_on r3d_sim; python -W ignore scripts/eval_robotwin2_ckpt.py "${smoke}/checkpoints/1.ckpt" --episodes 2 --seed 1 \
        --tag smoke ) > "${out}/smoke_eval.log" 2>&1
    grep -a "success" "${out}/smoke_eval.log" | tail -1
    rm -rf "${smoke}" R3D/r3d/env/robotwin2/eval_video/1-robotwin2_sim-${task}-p4096-g64-dropsmoke-42_seed1_smoke-*
    log "smoke ok, smoke run deleted"; touch "${out}/smoke.ok"
fi

run=$(rd p4096-g64-drop4567)
if [ ! -f "${run}/checkpoints/600.ckpt" ]; then
    log "train $(basename "${run}")"
    TAG=p4096-g64-drop4567 SETTING=sim_stock_p4096 EPOCHS=601 CHECKPOINT_EVERY=300 BATCH_SIZE=64 GRAD_ACCUM=4 \
        bash scripts/train_robotwin2_sim.sh "${task}" 42 0 task.num_points=4096 task.num_group=64 "${drop}" \
        > "${sim}/train_${task}_p4096-g64-drop4567.log" 2>&1
    log "train exit: $(ls "${run}/checkpoints" | tr '\n' ' ')"
fi

for t in p4096-g64-drop4567 p4096-g64 p4096-g64-cut45; do
    ck=$(rd "${t}")/checkpoints/600.ckpt
    [ -f "${ck%.ckpt}.eval_200ep_seed1_shared.json" ] && continue
    log "eval ${t} 600 on the shared list"
    ( env_on r3d_sim; python -W ignore scripts/eval_robotwin2_ckpt.py "${ck}" --episodes 200 --seed 1 --tag shared ) \
        > "$(rd "${t}")/eval_600_200ep_seed1_shared.log" 2>&1
    log "eval ${t}: $(python3 -c "import json;print(json.load(open('${ck%.ckpt}.eval_200ep_seed1_shared.json'))['success_rate'])")"
done

( env_on r3d_sim; python - "${sim}" <<'EOF'
import re, sys, json
from scipy.stats import binomtest
sim = sys.argv[1]
def eps(t):
    s = open(f"{sim}/robotwin2_sim-hanging_mug_labeled-{t}-42/eval_600_200ep_seed1_shared.log", errors="ignore").read()
    r = {int(k): v == "SUCCESS" for k, v in re.findall(r"Episode (\d+)/200: ([A-Z]+)", s)}
    assert len(r) == 200, (t, len(r))
    return r
R = {n: eps(t) for n, t in (("uncut", "p4096-g64"), ("arms cut", "p4096-g64-cut45"), ("dropout", "p4096-g64-drop4567"))}
out = {}
for n, r in R.items():
    print(f"{n}: {sum(r.values()) / 2:.1f}%")
for a, b in (("arms cut", "uncut"), ("dropout", "uncut"), ("dropout", "arms cut")):
    x = sum(R[a][k] and not R[b][k] for k in R[a]); y = sum(R[b][k] and not R[a][k] for k in R[a])
    p = binomtest(x, x + y, 0.5).pvalue
    print(f"{a} vs {b}: {x} / {y}  p = {p:.4f}")
    out[f"{a} vs {b}"] = [x, y, p]
json.dump({"success": {n: sum(r.values()) / 200 for n, r in R.items()}, "paired": out},
          open(f"{sim}/mug_drop/report.json", "w"), indent=1)
EOF
) | tee "${out}/report.txt"
log "mug dropout chain done"
