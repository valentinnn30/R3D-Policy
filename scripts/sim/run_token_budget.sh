#!/usr/bin/env bash
# Token-budget side runs (user, 2026-11-14): is 64 tokens a fair budget, and does the arm cut's gain survive more
# tokens? Square D2 4096 points, seed 42, 251 epochs, 64 x 2 (driver micro-batch 64 fits 4096/256: 19.3 GB),
# checkpoint 250 only, 200 paired scenes:  128 uncut | 128 arm cut [5] | 256 uncut. Queue file
# R3D/data/outputs/sim/mimicgen/token_budget/queue.txt (re-read by the driver). Waits for the claim sweep:
#
#   setsid nohup bash scripts/sim/run_token_budget.sh > R3D/data/outputs/sim/mimicgen/token_budget/chain.log 2>&1 &
set -uo pipefail
here="$(cd "$(dirname "$0")/../.." && pwd)"
cd "${here}"
out=R3D/data/outputs/sim/mimicgen/token_budget
log() { echo "[$(date '+%F %T')] $*"; }
busy() { ps -eo cmd | awk '($1=="bash" && $2 ~ /run_claim_sweep|run_remover_sweep|run_mimicgen_queue|run_round[0-9]+_chain|run_table_eval|run_thr_p1024|run_threading/) || ($1 ~ /python/ && ($0 ~ /train\.py/ || $0 ~ /eval_mimicgen_ckpt/ || $0 ~ /eval_robotwin2_ckpt/))' | grep -q .; }
log "waiting for the claim sweep to finish and the GPU to be free for 10 min"
free=0
while [ "${free}" -lt 10 ]; do
    if busy; then free=0; else free=$((free + 1)); fi
    sleep 60
done
log "GPU free; disk free $(df -h --output=avail R3D/data | tail -1)"
CKPT_EPOCHS='[250]' bash scripts/sim/run_mimicgen_queue.sh "${out}/queue.txt" >> "${out}/driver.log" 2>&1
tail -3 "${out}/driver.log"

python3 - > "${out}/report.txt" <<'PY'
import json, math, pathlib
S = pathlib.Path("R3D/data/outputs/sim")
def load(r):
    p = S / r / "eval/250_200ep_block1.json"
    return {s["seed"]: bool(s["success"]) for s in json.load(open(p))["scenes"]} if p.exists() else None
def mc(b, c):
    n = b + c
    return 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n)
R = {k: load(f"mimicgen_sim-square_d2-p4096-{k}-42") for k in ("g64", "g64-cut5", "g128", "g128-cut5", "g256")}
print("Square 4096, seed 42, 250.ckpt, 200 paired scenes")
for k, r in R.items():
    print(f"  {k:10s} {'missing' if r is None else f'{100 * sum(r.values()) / 200:.1f}%'}")
for a, b in (("g128", "g128-cut5"), ("g64", "g64-cut5"), ("g64", "g128"), ("g64", "g256"), ("g128", "g256"),
             ("g64-cut5", "g128"), ("g64-cut5", "g256"), ("g64-cut5", "g128-cut5")):
    if R[a] is None or R[b] is None: continue
    bo = sum(R[b][s] and not R[a][s] for s in R[a]); ao = sum(R[a][s] and not R[b][s] for s in R[a])
    print(f"  {a} -> {b}: {100 * (sum(R[b].values()) - sum(R[a].values())) / 200:+.1f} points, {b} only / {a} only {bo} / {ao}, p = {mc(bo, ao):.3g}")
PY
cat "${out}/report.txt"
log "token budget runs done"
