#!/usr/bin/env bash
# Token-dropout control (user, 2026-11-12): is the Square gain of robot-part dropout (DROP / FTcont / GObal) just
# "dropout", or does it need the structure? Standard ViT token dropout (PatchDropout, Liu et al. 2023 -- what Uni3D
# pretrains with): every training sample independently keeps a uniform random 75% of its encoder tokens (48 of 64),
# after the positional embedding, before the transformer; the head sees only the kept tokens + their centres;
# evaluation keeps all 64. Uninformed, unclustered. Everything else is the uncut baseline's recipe
# (mimicgen_sim-square_d2-p4096-g64-42: plain DP3, 4096 points, 64 tokens, seed 42, 251 epochs, 64 x 2):
#
#   setsid nohup bash scripts/sim/run_token_dropout.sh > R3D/data/outputs/sim/mimicgen/token_dropout/chain.log 2>&1 &
#
#   0 wait until no training / eval / chain has run for 10 min
#   1 checks: check_token_dropout.py
#   2 train (ckpt 250 only), eval 250 (200 scenes, seed block 1 -- paired with every Square run)
#   3 paired report vs uncut / DROP / FTcont -> token_dropout/report.txt
# Any failure stops the chain (set -e). Steps whose output exists are skipped.
set -euo pipefail
here="$(cd "$(dirname "$0")/../.." && pwd)"
cd "${here}"
seed=${SEED:-42}
p=${P_DROP:-0.25}
tag=${TAG:-tokdrop$(python3 -c "print(round(${p}*100))")}
out=R3D/data/outputs/sim/mimicgen/token_dropout
mkdir -p "${out}"
log() { echo "[$(date '+%F %T')] $*"; }
env_on() { eval "$(micromamba shell hook -s bash)"; micromamba activate "$1"; unset PYTHONPATH; }
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
run=R3D/data/outputs/sim/mimicgen_sim-square_d2-p4096-g64-${tag}-${seed}

busy() { ps -eo cmd | awk '($1=="bash" && $2 ~ /run_round[0-9]+_chain|run_remover_sweep|run_finetune_branches|run_mask_override|run_branch_test|run_early_branch|run_cut_keep7|run_cut_pair|run_mug_drop|run_threading/) || ($1 ~ /python/ && ($0 ~ /train\.py/ || $0 ~ /eval_mimicgen_ckpt/ || $0 ~ /eval_robotwin2_ckpt/))' | grep -q .; }
log "waiting for the GPU to be free for 10 min"
free=0
while [ "${free}" -lt 10 ]; do
    if busy; then free=0; else free=$((free + 1)); fi
    sleep 60
done
log "GPU free for 10 min; disk free: $(df -h --output=avail R3D/data | tail -1) (1 ckpt needs ~2.2 GB)"

if [ ! -f "${out}/checks.ok" ]; then
    log "checks"
    ( env_on r3d_sim; cd R3D; python -W ignore ../scripts/sim/check_token_dropout.py ) > "${out}/check_token_dropout.log" 2>&1
    grep -q "^ALL PASS" "${out}/check_token_dropout.log" || { log "check failed"; exit 1; }
    touch "${out}/checks.ok"
fi

if [ ! -f "${run}/checkpoints/250.ckpt" ]; then
    log "train ${tag} seed ${seed}: patch_dropout ${p}"
    TASK_CFG=mimicgen_sim SETTING=agentview_p4096-200 EPOCHS=251 CHECKPOINT_EVERY=125 BATCH_SIZE=64 GRAD_ACCUM=2 \
        TAG="p4096-g64-${tag}" bash scripts/train_robotwin2_sim.sh square_d2 "${seed}" 0 \
        task.num_group=64 task.num_points=4096 "policy.pointcloud_encoder_cfg.patch_dropout=${p}" \
        '+training.checkpoint_epochs=[250]' > "${run}.train.log" 2>&1
    log "train exit: $(ls "${run}/checkpoints" | tr '\n' ' ')"
fi

evaluate() {
    [ -f "${run}/eval/$1_200ep_block1.json" ] && return 0
    log "eval $1"
    ( env_on r3d_robosuite; python -W ignore scripts/sim/eval_mimicgen_ckpt.py "${run}/checkpoints/$1.ckpt" \
        --episodes 200 --workers 5 ) >> "${run}/eval.log" 2>&1
    log "eval $1: $(grep '^ckpt:' "${run}/eval.log" | tail -1 | sed 's|.*checkpoints/||; s|  [0-9.]* min.*||')"
}
evaluate 250

python3 - "${run}" > "${out}/report.txt" <<'EOF'
import json, math, sys, pathlib
S = pathlib.Path("R3D/data/outputs/sim")
def load(path):
    d = json.load(open(path)); return {s["seed"]: bool(s["success"]) for s in d["scenes"]}
def mcnemar(b, c):   # exact two-sided binomial on the discordant pairs
    n = b + c
    if n == 0: return 1.0
    k = min(b, c); return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)
run = pathlib.Path(sys.argv[1])
refs = {"uncut s42": "mimicgen_sim-square_d2-p4096-g64-42",
        "DROP s42": "mimicgen_sim_remover-square_d2-p4096-g64-rm-DROP-42",
        "DROP s43": "mimicgen_sim_remover-square_d2-p4096-g64-rm-DROP-43",
        "FTcont s42": "mimicgen_sim_remover-square_d2-p4096-g64-rm-FTcont-42",
        "arm cut s42": "mimicgen_sim-square_d2-p4096-g64-cut5-42"}
for ep in (250,):
    f = run / "eval" / f"{ep}_200ep_block1.json"
    if not f.exists(): continue
    a = load(f)
    print(f"token dropout @{ep}: {100 * sum(a.values()) / len(a):.1f}%  ({run.name})")
    for name, r in refs.items():
        g = S / r / "eval" / f"{ep}_200ep_block1.json"
        if not g.exists(): continue
        b = load(g); common = sorted(set(a) & set(b))
        only_a = sum(a[s] and not b[s] for s in common); only_b = sum(b[s] and not a[s] for s in common)
        print(f"  vs {name:12s} {100 * sum(b[s] for s in common) / len(common):5.1f}%  "
              f"tokdrop only / other only {only_a} / {only_b}, p = {mcnemar(only_a, only_b):.3f}  (n = {len(common)})")
EOF
cat "${out}/report.txt"
log "token dropout chain done"
