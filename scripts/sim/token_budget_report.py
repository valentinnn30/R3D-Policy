#!/usr/bin/env python3
"""Token-budget side runs: success + paired McNemar (seed 42, 250.ckpt, 200 paired scenes) and the coverage grid
measured on the stored frames (767 frames, 2026-11-14). Rerunnable with partial results.
Writes R3D/data/outputs/sim/mimicgen/token_budget/report.txt."""
import json, math, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[2]
S = ROOT / "R3D/data/outputs/sim"
# nut / peg coverage and object patch (cm), measured 2026-11-14 (scripts/sim/coverage_grid.py, token_allocation, 64 x 32 etc.)
COV = {"p1024-g64": (1.00, 1.00, 4.5), "p1024-g64-cut5": (1.00, 1.00, 4.3), "p1024-g128": (1.00, 1.00, 4.4),
       "p2048-g64": (0.84, 0.88, 3.4), "p2048-g64-cut5": (0.99, 1.00, 3.3), "p2048-g128": (0.99, 1.00, 3.3),
       "p2048-g128-cut5": (1.00, 1.00, 3.1), "p2048-g256": (1.00, 1.00, 3.1),
       "p4096-g64": (0.48, 0.57, 2.5), "p4096-g64-cut5": (0.83, 0.93, 2.4), "p4096-g128": (0.81, 0.92, 2.4),
       "p4096-g128-cut5": (0.99, 1.00, 2.2), "p4096-g256": (0.99, 1.00, 2.3), "p4096-g256-cut5": (1.00, 1.00, 2.1),
       "p4096-g64-k128": (1.00, 1.00, 4.7)}   # group size 128 (k = 32 everywhere else)
def load(k):
    p = S / f"mimicgen_sim-square_d2-{k}-42" / "eval/250_200ep_block1.json"
    return {s["seed"]: bool(s["success"]) for s in json.load(open(p))["scenes"]} if p.exists() else None
def mc(b, c):
    n = b + c
    return 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n)
R = {k: load(k) for k in COV}
L = ["Square D2, seed 42, 250.ckpt, 200 paired scenes; coverage / patch from the stored training frames", "",
     f"{'config':18s} {'success':>8s}  nut cov  peg cov  object patch"]
for k, (n, p, c) in COV.items():
    L.append(f"{k:18s} {'-' if R[k] is None else f'{100 * sum(R[k].values()) / 200:.1f}%':>8s}  {n:7.2f}  {p:7.2f}  {c:6.1f} cm")
L += ["", "paired (b - a, b only / a only, p):"]
for a, b in (("p4096-g64", "p4096-g128"), ("p4096-g64", "p4096-g256"), ("p4096-g128", "p4096-g256"),
             ("p4096-g64-cut5", "p4096-g128"), ("p4096-g64-cut5", "p4096-g256"),
             ("p4096-g128", "p4096-g128-cut5"), ("p4096-g256", "p4096-g256-cut5"), ("p2048-g128", "p2048-g128-cut5"),
             ("p4096-g64-cut5", "p4096-g128-cut5"), ("p2048-g64-cut5", "p2048-g128-cut5"),
             ("p4096-g64", "p4096-g64-k128"), ("p4096-g256", "p4096-g64-k128"), ("p4096-g64-cut5", "p4096-g64-k128"),
             ("p1024-g64", "p4096-g64-k128")):
    if R[a] is None or R[b] is None:
        continue
    bo = sum(R[b][s] and not R[a][s] for s in R[a]); ao = sum(R[a][s] and not R[b][s] for s in R[a])
    L.append(f"  {a} -> {b}: {100 * (sum(R[b].values()) - sum(R[a].values())) / 200:+.1f}, {bo} / {ao}, p = {mc(bo, ao):.3g}")
out = S / "mimicgen/token_budget/report.txt"
out.write_text("\n".join(L) + "\n")
print("\n".join(L))
