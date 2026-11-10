#!/usr/bin/env python
"""Threading policies (trained in the narrowed tripod window) evaluated on STOCK Threading D1 (r3d_sim, CPU).

    python scripts/sim/ood_threading_report.py

Reads the `_env-Threading_D1` evals of the 250 checkpoints (uncut, arm cut, GObalAll; the same 200 D1 scenes for
all three) and splits the scenes by `in_train_window` (tripod inside the training window or not). Per split: success
of each policy and paired exact McNemar of arm cut / GObalAll vs uncut. If removal / dropout buys robustness to
distribution shift, the gain should appear out of the window and not inside it.
"""

import json
import pathlib

import numpy as np
from scipy.stats import binomtest

SIM = pathlib.Path(__file__).resolve().parents[2] / "R3D/data/outputs/sim"
RUNS = {"uncut": "mimicgen_sim-threading_d1n-p4096-g64-42",
        "arm cut": "mimicgen_sim-threading_d1n-p4096-g64-cut5-42",
        "GObalAll": "mimicgen_sim_remover-threading_d1n-p4096-g64-rm-GObalAll-42"}
NAME = "250_200ep_block1_env-Threading_D1.json"


def main():
    d = {k: {s["seed"]: s for s in json.load(open(SIM / r / "eval" / NAME))["scenes"]} for k, r in RUNS.items()}
    seeds = sorted(set.intersection(*[set(v) for v in d.values()]))
    inw = np.array([d["uncut"][s]["in_train_window"] for s in seeds])
    if not all(np.array_equal(inw, np.array([d[k][s]["in_train_window"] for s in seeds])) for k in d):
        raise RuntimeError("the runs saw different scenes")
    out = {}
    for split, sel in (("all", np.ones_like(inw)), ("inside window", inw), ("outside window", ~inw)):
        S = [s for s, m in zip(seeds, sel) if m]
        line = {k: float(np.mean([d[k][s]["success"] for s in S])) for k in d}
        print(f"\n== {split}: {len(S)} scenes  " + "  ".join(f"{k} {v:.1%}" for k, v in line.items()))
        for k in ("arm cut", "GObalAll"):
            x = sum(d[k][s]["success"] and not d["uncut"][s]["success"] for s in S)
            y = sum(d["uncut"][s]["success"] and not d[k][s]["success"] for s in S)
            p = binomtest(x, x + y, 0.5).pvalue if x + y else 1.0
            print(f"  {k} vs uncut: {x} / {y}  p = {p:.3f}")
            line[f"{k}_vs_uncut"] = [x, y, p]
        out[split] = {"n": len(S), **line}
    json.dump(out, open(SIM / "mimicgen/threading/ood_threading_report.json", "w"), indent=1)


if __name__ == "__main__":
    main()
