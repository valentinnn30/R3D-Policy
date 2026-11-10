#!/usr/bin/env python
"""Round 7 report: sensitivity table + frozen-robot success with paired McNemar.

    python scripts/sim/round7_report.py      # -> R3D/data/outputs/sim/mimicgen/round7/report.md (+ stdout)
"""

import json
import os
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "sim"))
from analyze_remover_run import mcnemar  # noqa: E402

SIM = ROOT / "R3D/data/outputs/sim"
OUT = SIM / "mimicgen/round7"
RUNS = {
    "frozen": "mimicgen_sim-square_d2-p4096-g64-frozen-42",
    "uncut": "mimicgen_sim-square_d2-p4096-g64-42",
    "arm cut": "mimicgen_sim-square_d2-p4096-g64-cut5-42",
    "SPEC": "mimicgen_sim_remover-square_d2-p4096-g64-rm-SPEC-42",
    "SPEC06": "mimicgen_sim_remover-square_d2-p4096-g64-rm-SPEC06-42",
    "FTcont": "mimicgen_sim_remover-square_d2-p4096-g64-rm-FTcont-42",
    "DROP": "mimicgen_sim_remover-square_d2-p4096-g64-rm-DROP-42",
    "GObal": "mimicgen_sim_remover-square_d2-p4096-g64-rm-GObal-42",
}


def load(run, ep):
    f = SIM / run / "eval" / f"{ep}_200ep_block1.json"
    if not f.exists():
        return None
    rows = json.load(open(f))["scenes"]
    seeds = [r["seed"] for r in rows]
    if seeds != [100000 + k for k in range(200)]:
        raise ValueError(f"{f}: unexpected scene list")
    return np.array([r["success"] for r in rows])


def main():
    lines = []
    p = lambda s="": (lines.append(s), print(s))
    sens = OUT / "sensitivity.json"
    if sens.exists():
        S = json.load(open(sens))
        perts = ["objects_2cm", "objects_5cm", "arm57_2cm", "arm57_5cm", "gripper_2cm", "gripper_5cm"]
        for key, title in (("ratio", "effect / noise floor (fresh noise; 1.0 = no reaction)"),
                           ("crn_ratio", "pure effect / noise floor (same seed as the base call; 0 = no reaction)")):
            p(f"### Sensitivity, {S['windows']} held-out windows (val demos {S['val_episodes']}), {S['ckpt']}.ckpt: {title}")
            p()
            p("| policy | floor mm | " + " | ".join(perts) + " |")
            p("|---|---|" + "---|" * len(perts))
            for name, r in S["policies"].items():
                p(f"| {name} | {r['floor_mm']:.2f} | " + " | ".join(f"{r['perts'][k][key]:.2f}" for k in perts) + " |")
            p()
    succ = {(n, e): load(r, e) for n, r in RUNS.items() for e in (200, 250)}
    p("### Success, 200 paired scenes (seed block 1), seed 42")
    p()
    p("| run | 200 | 250 |")
    p("|---|---|---|")
    for n in RUNS:
        cell = lambda e: "–" if succ[(n, e)] is None else f"{100 * succ[(n, e)].mean():.1f}"
        p(f"| {n} | {cell(200)} | {cell(250)} |")
    p()
    p("### Paired McNemar (a only / b only, exact two-sided p), frozen vs …")
    p()
    p("| frozen @ | vs | a only / b only | p |")
    p("|---|---|---|---|")
    for e in (200, 250):
        a = succ[("frozen", e)]
        if a is None:
            continue
        # uncut / arm cut / FTcont exist at 250 only; the others at the same epoch
        for n, eb in (("uncut", 250), ("arm cut", 250), ("SPEC", e), ("SPEC06", e), ("FTcont", 250),
                      ("DROP", e), ("GObal", e)):
            b = succ[(n, eb)]
            if b is not None:
                n01, n10, pv = mcnemar(a, b)
                p(f"| {e} | {n} @{eb} | {n01} / {n10} | {pv:.3g} |")
    s6 = [succ[("SPEC06", e)] for e in (200, 250)]
    if all(x is not None for x in s6):
        p()
        p("### SPEC06 vs …")
        p()
        p("| SPEC06 @ | vs | a only / b only | p |")
        p("|---|---|---|---|")
        for e in (200, 250):
            for n, eb in (("SPEC", e), ("arm cut", 250), ("uncut", 250), ("FTcont", 250)):
                b = succ[(n, eb)]
                if b is not None:
                    n01, n10, pv = mcnemar(succ[("SPEC06", e)], b)
                    p(f"| {e} | {n} @{eb} | {n01} / {n10} | {pv:.3g} |")
    os.makedirs(OUT, exist_ok=True)
    open(OUT / "report.md", "w").write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
