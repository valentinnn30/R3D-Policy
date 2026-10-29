#!/usr/bin/env bash
# Early-branch test for the branch-and-prune mask search (2026-10-29). Follows the late branch
# (run_branch_test.sh: uncut -> arm cut at epoch 125 recovered only half of the cut's benefit,
# 13.0% vs 20.0% from scratch, 8.5% uncut). Here the branch point is epoch 50:
#
#   trunk   uncut 4096/64, seed 42, the standard 251-epoch schedule, stopped after 50.ckpt
#   cut5    trunk 50.ckpt -> arm cut (task.drop_point_labels=[5]), epochs 51..250, same schedule
#   uncut   trunk 50.ckpt -> uncut, epochs 51..250: the matched control. Skipped when the trunk
#           reproduces the uncut baseline's first 50 epochs exactly (val loss bit-identical):
#           then the baseline IS this control.
#
# Then 250.ckpt of each branch on the 200 fixed scenes, val loss at short horizons after the
# branch (@75 = the first 25 epochs) via wandb_val_loss.py, paired McNemar at 250.
# Only evaluated checkpoints are kept (trunk 50.ckpt stays: it is the branch point).
#
#   setsid nohup bash scripts/sim/run_early_branch.sh > R3D/data/outputs/sim/mimicgen/early_branch_driver.log 2>&1 &
#
# Every step is skipped when its output exists, so the driver can be restarted.
set -uo pipefail
here="$(cd "$(dirname "$0")/../.." && pwd)"
out="${here}/R3D/data/outputs/sim"
base="${out}/mimicgen_sim-square_d2-p4096-g64-42"
scratch_cut="${out}/mimicgen_sim-square_d2-p4096-g64-cut5-42"
late="${out}/mimicgen_sim-square_d2-p4096-g64-cut5from125-42"
trunk="${out}/mimicgen_sim-square_d2-p4096-g64-trunk-42"
report="${out}/mimicgen/early_branch"
branch_at=50
log() { echo "[$(date '+%F %T')] $*"; }
act() { eval "$(micromamba shell hook -s bash)"; micromamba activate "$1"; unset PYTHONPATH; }
mkdir -p "${report}"

train() {  # tag, extra overrides...
    local tag=$1; shift
    TASK_CFG=mimicgen_sim SETTING=agentview_p4096-200 EPOCHS=251 CHECKPOINT_EVERY=50 BATCH_SIZE=64 GRAD_ACCUM=2 \
        TAG="p4096-g64-${tag}" bash "${here}/scripts/train_robotwin2_sim.sh" square_d2 42 0 \
        task.num_group=64 task.num_points=4096 "$@"
}

# ---- trunk: uncut, stopped once 50.ckpt is written ----------------------------------------
if [ ! -f "${trunk}/checkpoints/${branch_at}.ckpt" ]; then
    log "train trunk (uncut) to epoch ${branch_at}"
    train trunk > "${trunk}.train.log" 2>&1 < /dev/null &
    tpid=$!
    # save_checkpoint is synchronous: once epoch 51 starts, 50.ckpt is complete
    until grep -q "Training epoch $((branch_at + 1)):" "${trunk}.train.log" 2>/dev/null || ! kill -0 ${tpid} 2>/dev/null; do
        sleep 20
    done
    # the python sits under two bash levels (this subshell -> train_robotwin2_sim.sh): kill the
    # whole subtree by PID, deepest first -- never by pattern
    tree() { local c; for c in $(pgrep -P "$1"); do tree "${c}"; echo "${c}"; done; }
    kids=$(tree ${tpid})
    [ -n "${kids}" ] && kill ${kids} 2>/dev/null
    wait ${tpid} 2>/dev/null
    [ -f "${trunk}/checkpoints/${branch_at}.ckpt" ] || { log "trunk wrote no ${branch_at}.ckpt"; exit 1; }
    log "trunk done"
fi

# ---- is the trunk the baseline? (val loss epochs 0..50, bit for bit) -------------------------
same=$(act r3d_sim; cd "${here}" && python -W ignore - "${base}" "${trunk}" ${branch_at} <<'EOF'
import sys
sys.path.insert(0, "scripts/sim")
from wandb_val_loss import history
def vl(run):
    return {int(r["epoch"]): float(r["val_loss"]) for r in history(run) if "val_loss" in r and "epoch" in r}
a, b, e = vl(sys.argv[1]), vl(sys.argv[2]), int(sys.argv[3])
ks = [k for k in range(e + 1) if k in a and k in b]
d = max(abs(a[k] - b[k]) for k in ks) if ks else float("nan")
print("same" if ks and d == 0.0 else f"differs(max_abs_diff={d:.3g},n={len(ks)})")
EOF
)
log "trunk vs uncut baseline, val loss epochs 0..${branch_at}: ${same}"
branches="cut5"
[ "${same}" = same ] || branches="cut5 uncut"

# ---- branches -------------------------------------------------------------------------------
for b in ${branches}; do
    run="${out}/mimicgen_sim-square_d2-p4096-g64-${b}from${branch_at}-42"
    if [ ! -f "${run}/checkpoints/250.ckpt" ] && [ ! -f "${run}/eval/250_200ep_block1.json" ]; then
        mkdir -p "${run}/checkpoints"
        # a copy of the trunk checkpoint without its `_output_dir` pickle, so the resume can
        # never write into the trunk's directory
        (act r3d_sim; python - "${trunk}/checkpoints/${branch_at}.ckpt" "${run}/checkpoints/latest.ckpt" <<'EOF'
import sys, torch, dill
p = torch.load(open(sys.argv[1], "rb"), pickle_module=dill, map_location="cpu")
p["pickles"].pop("_output_dir", None)
torch.save(p, open(sys.argv[2], "wb"), pickle_module=dill)
EOF
        ) || { log "branch ${b}: checkpoint copy failed"; exit 1; }
        extra=()
        [ "${b}" = cut5 ] && extra=("task.drop_point_labels=[5]")
        log "train branch ${b} from trunk ${branch_at}.ckpt"
        train "${b}from${branch_at}" training.resume=True "${extra[@]}" > "${run}.train.log" 2>&1 < /dev/null
        log "train ${b} exit $?"
        [ -f "${run}/checkpoints/250.ckpt" ] || { log "branch ${b}: no 250.ckpt"; continue; }
    fi
    for f in "${run}"/checkpoints/*.ckpt; do          # keep only what gets evaluated
        [ "$(basename "${f}")" = 250.ckpt ] || { [ -f "${f}" ] && log "delete ${b}/$(basename "${f}")" && rm -f "${f}"; }
    done
    if [ ! -f "${run}/eval/250_200ep_block1.json" ]; then
        log "eval ${b} 250"
        (act r3d_robosuite; cd "${here}" && python -W ignore scripts/sim/eval_mimicgen_ckpt.py "${run}/checkpoints/250.ckpt" \
            --episodes 200 --workers 5 >> "${run}/eval.log" 2>&1 < /dev/null)
        log "eval ${b} exit $?: $(grep '^ckpt:' "${run}/eval.log" | tail -1 | sed 's|.*checkpoints/||; s|  [0-9.]* min.*||')"
    fi
done

# ---- report -----------------------------------------------------------------------------------
runs=("${base}" "${scratch_cut}" "${late}" "${trunk}")
for b in cut5 uncut; do
    r="${out}/mimicgen_sim-square_d2-p4096-g64-${b}from${branch_at}-42"
    [ -d "${r}" ] && runs+=("${r}")
done
(act r3d_sim; cd "${here}" && python -W ignore scripts/sim/wandb_val_loss.py "${runs[@]}" --at 50 75 100 150 200 250 \
    --json "${report}/val_loss.json" > "${report}/val_loss.txt" 2>&1
python3 - "${report}" "${runs[@]}" <<'EOF' > "${report}/paired.txt" 2>&1
import json, os, sys
from math import comb
rep, runs = sys.argv[1], sys.argv[2:]
def s(r):
    f = os.path.join(r, "eval", "250_200ep_block1.json")
    return {x["seed"]: x["success"] for x in json.load(open(f))["scenes"]} if os.path.exists(f) else None
res = {os.path.basename(r): s(r) for r in runs}
res = {k: v for k, v in res.items() if v}
names = list(res)
for i, a in enumerate(names):
    for b in names[i + 1:]:
        A, B = res[a], res[b]
        ks = A.keys() & B.keys()
        ab = sum(A[k] and not B[k] for k in ks); ba = sum(B[k] and not A[k] for k in ks); n = ab + ba
        p = min(1.0, 2 * sum(comb(n, j) for j in range(min(ab, ba) + 1)) / 2 ** n) if n else 1.0
        print(f"{a:45s} {100*sum(A[k] for k in ks)/len(ks):5.1f}%  vs  {b:45s} {100*sum(B[k] for k in ks)/len(ks):5.1f}%   {ab:3d} vs {ba:3d}  p={p:.3f}")
EOF
)
log "early branch done -> ${report}/{val_loss,paired}.txt"
