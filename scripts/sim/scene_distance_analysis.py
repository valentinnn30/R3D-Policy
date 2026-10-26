#!/usr/bin/env python
"""Memorization check: does an eval scene succeed because it is CLOSE to a training layout?

    python scripts/sim/scene_distance_analysis.py                     # hanging_mug, all 200-scene evals

Inputs: the layouts from `dump_scene_layouts.py` (object poses at t=0 for every
training and eval seed) and the per-episode SUCCESS/FAIL lines of each run's
`eval_600_200ep_seed1.log` (episode k <-> k-th seed of the "Found 200 valid task
seeds" list; the runner iterates that list in order).

Distances (cm) from an eval scene to its NEAREST training layout:
  mug    [mug centre xy, handle xy]   -- the handle point carries the mug's yaw
  rack   [rack origin xy, hook xy]    -- the hook point carries the rack's yaw
  joint  the whole layout (mug and rack of the SAME training scene)
  mug_same_model  mug distance restricted to training scenes with the same mug mesh

Per run: AUC = P(a failed scene is farther than a successful one); 0.5 = distance
does not matter, 1.0 = every failure is farther than every success. One-sided
Mann-Whitney p. The same AUC is reported for plain layout variables (mug x, handle
angle, ...) so "far from training" can be told apart from "hard place".

Output: R3D/data/outputs/sim/scene_layouts/<task>_distance.json + .png
"""

import argparse
import glob
import json
import os
import re

import numpy as np
from scipy.stats import mannwhitneyu, spearmanr

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SIM_OUT = os.path.join(ROOT, "R3D", "data", "outputs", "sim")
LAYOUT_DIR = os.path.join(SIM_OUT, "scene_layouts")

SEEDS_RE = re.compile(r"Found \d+ valid task seeds: \[([0-9, ]+)\]")
EP_RE = re.compile(r"Episode (\d+)/(\d+): (SUCCESS|FAIL)")


def features(rec):
    mug, rack = rec["object0"], rec["target0"]
    m, h = np.array(mug["p"][:2]), np.array(mug["fp0"][:2])
    r, k = np.array(rack["p"][:2]), np.array(rack["fp0"][:2])
    d = h - m
    return {
        "mug": np.r_[m, h], "rack": np.r_[r, k],
        "mug_x": m[0], "mug_y": m[1], "handle_angle": np.degrees(np.arctan2(d[1], d[0])),
        "rack_x": r[0], "hook_x": k[0], "hook_y": k[1], "mug_id": rec["mug_id"],
    }


def nn_dists(ev, tr, exclude_self=False):
    """cm to the nearest training layout, per object and jointly (same training scene)."""
    out = {}
    dm = np.linalg.norm(ev["mug"][:, None] - tr["mug"][None], axis=-1)
    dr = np.linalg.norm(ev["rack"][:, None] - tr["rack"][None], axis=-1)
    dj = np.sqrt(dm ** 2 + dr ** 2)
    same = ev["mug_id"][:, None] == tr["mug_id"][None]
    dms = np.where(same, dm, np.inf)
    if exclude_self:
        for a in (dm, dr, dj, dms):
            np.fill_diagonal(a, np.inf)
    out["mug"], out["rack"], out["joint"] = dm.min(1), dr.min(1), dj.min(1)
    out["joint_k3"] = np.sort(dj, 1)[:, :3].mean(1)
    out["mug_same_model"] = dms.min(1)
    return {k: 100 * v for k, v in out.items()}


def stack(recs):
    f = [features(r) for r in recs]
    return {k: np.array([x[k] for x in f]) for k in f[0]}


def auc_fail_higher(x, success):
    """P(x_fail > x_success) and one-sided Mann-Whitney p."""
    xs, xf = x[success], x[~success]
    if len(xs) == 0 or len(xf) == 0:
        return float("nan"), float("nan")
    u, p = mannwhitneyu(xf, xs, alternative="greater")
    return float(u / (len(xs) * len(xf))), float(p)


def parse_log(path):
    txt = open(path, errors="replace").read()
    seeds = [int(s) for s in SEEDS_RE.search(txt).group(1).split(",")]
    eps = {int(i): r == "SUCCESS" for i, _, r in EP_RE.findall(txt)}
    assert sorted(eps) == list(range(1, len(seeds) + 1)), f"{path}: {len(eps)} episodes vs {len(seeds)} seeds"
    return seeds, np.array([eps[i + 1] for i in range(len(seeds))])


def run_label(run_dir, task):
    return os.path.basename(run_dir).replace(f"robotwin2_sim-{task}-", "").replace("-42", "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="hanging_mug_labeled")
    ap.add_argument("--log-name", default="eval_600_200ep_seed1.log")
    a = ap.parse_args()

    L = json.load(open(os.path.join(LAYOUT_DIR, f"{a.task}.json")))
    tr_seeds = list(L["train"])
    tr = stack([L["train"][s] for s in tr_seeds])
    ev_all = {int(s): r for s, r in L["eval"].items()}

    # scale reference: training layouts to their nearest OTHER training layout
    loo = nn_dists(tr, tr, exclude_self=True)
    res = {"train_loo_nn_cm": {k: float(np.median(v)) for k, v in loo.items()},
           "train_ranges": {k: [float(tr[k].min()), float(tr[k].max())]
                            for k in ("mug_x", "mug_y", "handle_angle", "rack_x", "hook_x", "hook_y")},
           "runs": {}}

    VARS = ("mug_x", "mug_y", "handle_angle", "rack_x", "hook_x", "hook_y")
    per_scene = {}  # seed -> list of outcomes over runs
    runs = sorted(glob.glob(os.path.join(SIM_OUT, f"robotwin2_sim-{a.task}-*", a.log_name)))
    print(f"{len(runs)} runs; training LOO nn (cm): "
          + ", ".join(f"{k} {v:.1f}" for k, v in res["train_loo_nn_cm"].items()))
    hdr = f"{'run':<16}{'succ':>6}" + "".join(f"{k:>16}" for k in ("joint", "mug", "rack", "mug_same_model"))
    print("\nAUC (P[fail farther than success]), one-sided p in ()\n" + hdr)
    for log in runs:
        name = run_label(os.path.dirname(log), a.task)
        seeds, succ = parse_log(log)
        ev = stack([ev_all[s] for s in seeds])
        d = nn_dists(ev, tr)
        r = {"success": float(succ.mean()), "n": len(seeds), "dist": {}, "vars": {}, "terciles": {}}
        for k, v in d.items():
            auc, p = auc_fail_higher(v, succ)
            q = np.quantile(v, [1 / 3, 2 / 3])
            bins = np.digitize(v, q)
            r["dist"][k] = {"auc": auc, "p": p, "median_cm": float(np.median(v))}
            r["terciles"][k] = [float(succ[bins == b].mean()) for b in range(3)]
        for k in VARS:
            auc, _ = auc_fail_higher(ev[k], succ)
            r["vars"][k] = float(auc)
        r["by_mug_id"] = {int(m): [int(succ[ev["mug_id"] == m].sum()), int((ev["mug_id"] == m).sum())]
                          for m in np.unique(ev["mug_id"])}
        res["runs"][name] = r
        for s, ok in zip(seeds, succ):
            per_scene.setdefault(s, []).append(bool(ok))
        print(f"{name:<16}{succ.mean():>6.2f}" + "".join(
            f"{r['dist'][k]['auc']:>9.2f} ({r['dist'][k]['p']:.2f})" for k in ("joint", "mug", "rack", "mug_same_model")))

    print("\nsuccess by distance tercile (near / mid / far), joint distance")
    for n, r in res["runs"].items():
        print(f"  {n:<16}" + " / ".join(f"{x:.2f}" for x in r["terciles"]["joint"]))

    print("\nAUC of plain layout variables (0.5 = no effect; <0.5 = failures at LOW values)")
    print(f"{'run':<16}" + "".join(f"{k:>13}" for k in VARS))
    for n, r in res["runs"].items():
        print(f"{n:<16}" + "".join(f"{r['vars'][k]:>13.2f}" for k in VARS))

    # pooled: scene difficulty = fraction of runs that solved it
    seeds = sorted(per_scene)
    frac = np.array([np.mean(per_scene[s]) for s in seeds])
    ev = stack([ev_all[s] for s in seeds])
    d = nn_dists(ev, tr)
    res["pooled"] = {"n_scenes": len(seeds), "spearman": {}}
    print(f"\npooled over runs: {len(seeds)} scenes, Spearman(solve fraction, x)")
    for k in list(d) + list(VARS):
        x = d[k] if k in d else ev[k]
        rho, p = spearmanr(frac, x)
        res["pooled"]["spearman"][k] = [float(rho), float(p)]
        print(f"  {k:<16} rho {rho:+.2f}  p {p:.3f}")
    mug_tab = {int(m): float(frac[ev["mug_id"] == m].mean()) for m in np.unique(ev["mug_id"])}
    res["pooled"]["solve_fraction_by_mug_id"] = mug_tab
    res["pooled"]["train_count_by_mug_id"] = {int(m): int((tr["mug_id"] == m).sum()) for m in np.unique(tr["mug_id"])}
    print("  solve fraction by mug model:", {m: round(v, 2) for m, v in mug_tab.items()})
    print("  training scenes per mug model:", res["pooled"]["train_count_by_mug_id"])

    out = os.path.join(LAYOUT_DIR, f"{a.task}_distance")
    with open(out + ".json", "w") as f:
        json.dump(res, f, indent=1)
    plot(out + ".png", tr, ev, frac, d)
    print("wrote", out + ".json", out + ".png")


def plot(path, tr, ev, frac, d):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ink, muted, grid = "#0b0b0b", "#52514e", "#e4e3df"
    seq = matplotlib.colors.LinearSegmentedColormap.from_list("seq", ["#cfe0f6", "#2a78d6", "#0d3a75"])
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": muted, "axes.labelcolor": ink,
                         "xtick.color": muted, "ytick.color": muted, "axes.grid": True,
                         "grid.color": grid, "grid.linewidth": 0.6, "axes.axisbelow": True})
    fig, ax = plt.subplots(1, 3, figsize=(14, 4.4), facecolor="#fcfcfb")
    for a_ in ax:
        a_.set_facecolor("#fcfcfb")
        for s in ("top", "right"):
            a_.spines[s].set_visible(False)

    # A: mug handle position, eval coloured by solve fraction, training as hollow diamonds
    sc = ax[0].scatter(ev["mug"][:, 2], ev["mug"][:, 3], c=frac, cmap=seq, vmin=0, vmax=1, s=22,
                       edgecolor="#fcfcfb", linewidth=0.6)
    ax[0].scatter(tr["mug"][:, 2], tr["mug"][:, 3], marker="D", facecolor="none", edgecolor=ink, s=30,
                  linewidth=1.1, label="training layout")
    ax[0].set(title="Mug handle position (world xy)", xlabel="x (m)", ylabel="y (m)")
    ax[0].legend(loc="upper left", frameon=False)
    # B: rack hook position
    ax[1].scatter(ev["rack"][:, 2], ev["rack"][:, 3], c=frac, cmap=seq, vmin=0, vmax=1, s=22,
                  edgecolor="#fcfcfb", linewidth=0.6)
    ax[1].scatter(tr["rack"][:, 2], tr["rack"][:, 3], marker="D", facecolor="none", edgecolor=ink, s=30,
                  linewidth=1.1)
    ax[1].set(title="Rack hook position (world xy)", xlabel="x (m)", ylabel="y (m)")
    cb = fig.colorbar(sc, ax=ax[:2], orientation="horizontal", fraction=0.05, pad=0.16, aspect=50)
    cb.set_label("fraction of runs that solved the scene", color=ink)
    cb.outline.set_visible(False)
    # C: solve fraction vs joint distance to nearest training layout
    ax[2].scatter(d["joint"], frac, s=18, color="#2a78d6", edgecolor="#fcfcfb", linewidth=0.6, alpha=0.85)
    bins = np.quantile(d["joint"], np.linspace(0, 1, 6))
    idx = np.clip(np.digitize(d["joint"], bins[1:-1]), 0, 4)
    cx = [np.median(d["joint"][idx == b]) for b in range(5)]
    cy = [frac[idx == b].mean() for b in range(5)]
    ax[2].plot(cx, cy, color=ink, linewidth=2, marker="o", markersize=5, label="quintile mean")
    ax[2].set(title="Scene solve fraction vs. distance to nearest training layout",
              xlabel="joint distance (cm)", ylabel="fraction of runs that solved it", ylim=(-0.03, 1.03))
    ax[2].legend(loc="upper right", frameon=False)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())


if __name__ == "__main__":
    main()
