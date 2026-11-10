#!/usr/bin/env bash
# The hanging_mug dropout network (drop4567, 600.ckpt) evaluated WITH the arms cut [4,5] at test time, shared scene list
# -- the counterpart of Square's GObal mask-override evals. Waits for run_mug_drop.sh; then a paired report against
# uncut / arms cut / dropout-on-full-cloud (all on the shared list). Detached:
#   setsid nohup bash scripts/sim/run_mug_drop_cuteval.sh > R3D/data/outputs/sim/mug_drop/cuteval.log 2>&1 &
set -euo pipefail
here="$(cd "$(dirname "$0")/../.." && pwd)"
cd "${here}"
sim=R3D/data/outputs/sim
log() { echo "[$(date '+%F %T')] $*"; }
env_on() { eval "$(micromamba shell hook -s bash)"; micromamba activate "$1"; unset PYTHONPATH; }
run=${sim}/robotwin2_sim-hanging_mug_labeled-p4096-g64-drop4567-42
log "waiting for run_mug_drop.sh"
while grep -q "mug dropout chain done" ${sim}/mug_drop/chain.log; [ $? -ne 0 ]; do sleep 60; done
log "eval dropout 600 with the arms cut"
( env_on r3d_sim; python -W ignore scripts/eval_robotwin2_ckpt.py "${run}/checkpoints/600.ckpt" --episodes 200 --seed 1 \
    --tag shared --drop-point-labels 4,5 ) > "${run}/eval_600_200ep_seed1_shared_cut-45.log" 2>&1
( env_on r3d_sim; python - "${sim}" <<'PY'
import re, sys
from scipy.stats import binomtest
sim = sys.argv[1]
def eps(t, f):
    s = open(f"{sim}/robotwin2_sim-hanging_mug_labeled-{t}-42/{f}", errors="ignore").read()
    r = {int(k): v == "SUCCESS" for k, v in re.findall(r"Episode (\d+)/200: ([A-Z]+)", s)}
    assert len(r) == 200, (t, f, len(r))
    return r
R = {"uncut": eps("p4096-g64", "eval_600_200ep_seed1_shared.log"),
     "arms cut": eps("p4096-g64-cut45", "eval_600_200ep_seed1_shared.log"),
     "dropout, full cloud": eps("p4096-g64-drop4567", "eval_600_200ep_seed1_shared.log"),
     "dropout, arms cut at eval": eps("p4096-g64-drop4567", "eval_600_200ep_seed1_shared_cut-45.log")}
for n, r in R.items():
    print(f"{n}: {sum(r.values()) / 2:.1f}%")
for a, b in (("dropout, arms cut at eval", "dropout, full cloud"), ("dropout, arms cut at eval", "arms cut"),
             ("dropout, arms cut at eval", "uncut")):
    x = sum(R[a][k] and not R[b][k] for k in R[a]); y = sum(R[b][k] and not R[a][k] for k in R[a])
    print(f"{a} vs {b}: {x} / {y}  p = {binomtest(x, x + y, 0.5).pvalue:.4f}")
PY
) | tee ${sim}/mug_drop/cuteval_report.txt
log "done"
