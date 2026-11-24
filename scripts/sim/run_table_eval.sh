#!/usr/bin/env bash
# Eval-only, out of interest (user, 2026-11-12): Square D2 policies on the cloud WITH the table (workspace box
# only: floor 2 cm below the table top, TABLE not filtered; `eval_mimicgen_ckpt.py --keep-table 0.02`).
# Known beforehand (scripts/sim/table_cloud_check.py, 10 eval resets): the table takes 3630 / 4096 points and
# 53 / 64 tokens, nut / peg fall from 11.3 / 9.9 to 1.7 / 1.0 tokens -- so this measures token starvation by
# background, NOT whether a policy ignores irrelevant content.
#
#   setsid nohup bash scripts/sim/run_table_eval.sh > R3D/data/outputs/sim/mimicgen/table_eval/chain.log 2>&1 &
#
# Waits until no training / eval / chain has run for 10 min (incl. the token-dropout chain), then evaluates
# 250.ckpt of DROP s42 and uncut s42 on the 200 scenes of seed block 1, and writes a paired report.
set -euo pipefail
here="$(cd "$(dirname "$0")/../.." && pwd)"
cd "${here}"
out=R3D/data/outputs/sim/mimicgen/table_eval
mkdir -p "${out}"
log() { echo "[$(date '+%F %T')] $*"; }
env_on() { eval "$(micromamba shell hook -s bash)"; micromamba activate "$1"; unset PYTHONPATH; }
S=R3D/data/outputs/sim
runs=(mimicgen_sim_remover-square_d2-p4096-g64-rm-DROP-42 mimicgen_sim-square_d2-p4096-g64-42)

busy() { ps -eo cmd | awk '($1=="bash" && $2 ~ /run_round[0-9]+_chain|run_remover_sweep|run_finetune_branches|run_mask_override|run_branch_test|run_early_branch|run_cut_keep7|run_cut_pair|run_mug_drop|run_threading|run_token_dropout/) || ($1 ~ /python/ && ($0 ~ /train\.py/ || $0 ~ /eval_mimicgen_ckpt/ || $0 ~ /eval_robotwin2_ckpt/))' | grep -q .; }
log "waiting for the GPU to be free for 10 min"
free=0
while [ "${free}" -lt 10 ]; do
    if busy; then free=0; else free=$((free + 1)); fi
    sleep 60
done
log "GPU free for 10 min"

for r in "${runs[@]}"; do
    [ -f "${S}/${r}/eval/250_200ep_block1_table.json" ] && continue
    log "eval ${r} 250 with table"
    ( env_on r3d_robosuite; python -W ignore scripts/sim/eval_mimicgen_ckpt.py "${S}/${r}/checkpoints/250.ckpt" \
        --episodes 200 --workers 5 --keep-table 0.02 ) >> "${S}/${r}/eval_table.log" 2>&1
    log "  $(grep '^ckpt:' "${S}/${r}/eval_table.log" | tail -1 | sed 's|.*checkpoints/||; s|  [0-9.]* min.*||')"
done

python3 - "${runs[@]}" > "${out}/report.txt" <<'PY'
import json, math, sys, pathlib
S = pathlib.Path("R3D/data/outputs/sim")
def load(p):
    return {s["seed"]: bool(s["success"]) for s in json.load(open(p))["scenes"]}
def mcnemar(b, c):
    n = b + c
    return 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n)
def pair(a, b):
    k = sorted(set(a) & set(b)); x = sum(a[s] and not b[s] for s in k); y = sum(b[s] and not a[s] for s in k)
    return f"{x} / {y}, p = {mcnemar(x, y):.3f}"
res = {}
for r in sys.argv[1:]:
    t, n = load(S / r / "eval/250_200ep_block1_table.json"), load(S / r / "eval/250_200ep_block1.json")
    res[r] = t
    print(f"{r}: with table {100 * sum(t.values()) / len(t):.1f}%  vs normal cloud {100 * sum(n.values()) / len(n):.1f}%"
          f"  (table only / normal only {pair(t, n)})")
a, b = sys.argv[1:3]
print(f"with table, DROP vs uncut (DROP only / uncut only): {pair(res[a], res[b])}")
PY
cat "${out}/report.txt"
log "table eval done"
