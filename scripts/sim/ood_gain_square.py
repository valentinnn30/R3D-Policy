#!/usr/bin/env python
"""Does the Square arm-cut / dropout gain concentrate in out-of-distribution scenes?  (r3d_sim, CPU, eval jsons only)

    python scripts/sim/ood_gain_square.py

Per eval scene (the 200 fixed scenes, epoch 250) the success of three groups of 4096 / 64 runs:
  U  uncut s42 (the only uncut run)
  C  arm cut, no dropout: cut5-42, cut5-43, cutL07-44 (== arm cut, s44)
  D  trained with robot-part dropout: GObal-42, GObal-43, FTcont-42, DROP-43, GObalAll-42
and three per-scene measures of how unusual the scene is:
  nn_dist   distance (m) of (nut xy, peg xy) at reset to the nearest of the 200 training layouts
  reach     max xy distance (m) of nut and peg from the robot base (-0.56, 0): how far the arm must extend
  link0_4   link0-4 points per frame in GObal-42's rollout on the link0-6-removed override eval (a proxy
            for how much of the near-base arm the camera sees; policy-dependent)
Scenes are split into tertiles of each measure; per tertile the success of U / C / D and the gains C-U, D-U,
with a bootstrap 95% CI (over scenes) of gain(far) - gain(near).
"""

import json
import pathlib

import numpy as np
import zarr

ROOT = pathlib.Path(__file__).resolve().parents[2]
SIM = ROOT / "R3D/data/outputs/sim"
BASE_XY = np.array([-0.56, 0.0])          # robot0_base, read from the simulator
GROUPS = {
    "U": ["mimicgen_sim-square_d2-p4096-g64-42"],
    "C": ["mimicgen_sim-square_d2-p4096-g64-cut5-42", "mimicgen_sim-square_d2-p4096-g64-cut5-43",
          "mimicgen_sim-square_d2-p4096-g64-cutL07-44"],
    "D": [f"mimicgen_sim_remover-square_d2-p4096-g64-rm-{t}" for t in
          ("GObal-42", "GObal-43", "FTcont-42", "DROP-43", "GObalAll-42")],
}


def load(run, name="250_200ep_block1.json"):
    return {s["seed"]: s for s in json.load(open(SIM / run / "eval" / name))["scenes"]}


def main():
    runs = {g: [load(r) for r in rs] for g, rs in GROUPS.items()}
    seeds = sorted(set.intersection(*[set(d) for rs in runs.values() for d in rs]))
    succ = {g: np.array([[float(d[s]["success"]) for s in seeds] for d in rs]).mean(0) for g, rs in runs.items()}
    lay = np.array([load(GROUPS["U"][0])[s]["layout"] for s in seeds])          # nut pos quat | peg pos quat
    train = zarr.open(str(ROOT / "R3D/data/square_d2-agentview_p4096-200.zarr"), "r")["meta/init_layout"][:]
    key = lambda L: np.concatenate([L[:, 0:2], L[:, 7:9]], 1)
    nn = np.linalg.norm(key(lay)[:, None] - key(train)[None], axis=-1).min(1)
    reach = np.maximum(np.linalg.norm(lay[:, 0:2] - BASE_XY, axis=1), np.linalg.norm(lay[:, 7:9] - BASE_XY, axis=1))
    ov = load("mimicgen_sim_remover-square_d2-p4096-g64-rm-GObal-42",
              "250_200ep_block1_mask-link0+link1+link2+link3+link4+link5+link6.json")
    l04 = np.array([ov[s]["override_cells_cut_mean"] for s in seeds])
    print(f"{len(seeds)} scenes; overall U {succ['U'].mean():.1%}  C {succ['C'].mean():.1%}  D {succ['D'].mean():.1%}")
    print(f"correlations (Spearman): nn_dist-reach {spear(nn, reach):+.2f}, nn_dist-link0_4 {spear(nn, l04):+.2f}, "
          f"reach-link0_4 {spear(reach, l04):+.2f}")
    rng = np.random.default_rng(0)
    out = {}
    for name, m in (("nn_dist", nn), ("reach", reach), ("link0_4", l04)):
        t = np.quantile(m, [1 / 3, 2 / 3])
        bins = np.digitize(m, t)
        print(f"\n== {name} tertiles")
        print("| tertile | range | n | U | C | D | C-U | D-U |\n|---|---|---|---|---|---|---|---|")
        rows = []
        for b, lab in enumerate(("near", "mid", "far")):
            s = bins == b
            u, c, d = (succ[g][s].mean() for g in "UCD")
            rows.append((lab, float(m[s].min()), float(m[s].max()), int(s.sum()), u, c, d))
            print(f"| {lab} | {m[s].min():.3f}-{m[s].max():.3f} | {s.sum()} | {u:.1%} | {c:.1%} | {d:.1%} | {c - u:+.1%} | {d - u:+.1%} |")
        res = {}
        for g in "CD":
            gain = succ[g] - succ["U"]
            far, near = np.flatnonzero(bins == 2), np.flatnonzero(bins == 0)
            boots = [gain[rng.choice(far, len(far))].mean() - gain[rng.choice(near, len(near))].mean() for _ in range(5000)]
            lo, hi = np.percentile(boots, [2.5, 97.5])
            res[g] = (float(gain[far].mean() - gain[near].mean()), float(lo), float(hi))
            print(f"  {g}-U gain far - near: {res[g][0]:+.1%}  95% CI [{lo:+.1%}, {hi:+.1%}]")
        out[name] = {"rows": rows, "far_minus_near": res}
    json.dump(out, open(SIM / "mimicgen/ood_gain_square.json", "w"), indent=1)
    print("\nwrote", SIM / "mimicgen/ood_gain_square.json")


def spear(a, b):
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


if __name__ == "__main__":
    main()
