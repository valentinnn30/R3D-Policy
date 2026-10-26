#!/usr/bin/env python
"""Figure of the MimicGen sweep results: final success with 95% Wilson intervals per
points x tokens, mid -> final on the first 100 scenes, success by nearest-demo distance.

    python scripts/sim/plot_mimicgen_sweep.py --points 1024 2048
Output: R3D/data/outputs/sim/mimicgen/sweep1_results.png
"""
import argparse, json, os
import numpy as np
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUT = os.path.join(ROOT, "R3D", "data", "outputs", "sim")
TOK = {1024: (32, 64, 128), 2048: (64, 128, 256), 4096: (128, 256, 512)}


def wilson(k, n, z=1.96):
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d; h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def load(p, g, f):
    path = os.path.join(OUT, f"mimicgen_sim-square_d2-p{p}-g{g}-42", "eval", f)
    return json.load(open(path))["scenes"] if os.path.exists(path) else None


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--points", type=int, nargs="+", default=[1024, 2048]); a = ap.parse_args()
    import zarr, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    ink, muted, bg, grid = "#0b0b0b", "#52514e", "#fcfcfb", "#e4e3df"
    col = {1024: "#2a78d6", 2048: "#eb6834", 4096: "#1baf7a"}
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.8), facecolor=bg, gridspec_kw={"width_ratios": [1.2, 1, 1]})
    for x in ax:
        x.set_facecolor(bg); [x.spines[s].set_visible(False) for s in ("top", "right")]
        x.grid(axis="y", color=grid, lw=0.6); x.set_axisbelow(True); x.tick_params(colors=muted)
    # A: final success by points-per-token
    ppt = (32, 16, 8); w = 0.8 / len(a.points)
    for i, p in enumerate(a.points):
        for j, r in enumerate(ppt):
            g = p // r; sc = load(p, g, "250_200ep_block1.json")
            if sc is None: continue
            k, n = sum(s["success"] for s in sc), len(sc); lo, hi = wilson(k, n); x = j + (i - (len(a.points) - 1) / 2) * w
            ax[0].bar(x, k / n, w * 0.92, color=col[p], label=f"{p} points" if j == 0 else None)
            ax[0].errorbar(x, k / n, yerr=[[k / n - lo], [hi - k / n]], color=ink, lw=1, capsize=3)
            ax[0].text(x, hi + 0.008, f"{k / n:.1%}\n{g} tok", ha="center", va="bottom", fontsize=7.5, color=ink)
    ax[0].set_xticks(range(3)); ax[0].set_xticklabels([f"{r} points / token" for r in ppt])
    ax[0].set_ylabel("success (final ckpt, 200 scenes)"); ax[0].set_ylim(0, 0.42)
    ax[0].set_title("Square D2: success by density and tokens (95% CI)", fontsize=10, color=ink); ax[0].legend(frameon=False, fontsize=8, loc="upper right")
    # B: mid -> final, first 100 scenes; end labels spread so equal values stay readable
    ends = []
    for i, p in enumerate(a.points):
        for g in TOK[p]:
            m, f = load(p, g, "125_100ep_block1.json"), load(p, g, "250_200ep_block1.json")
            if m is None or f is None: continue
            y = [np.mean([s["success"] for s in m]), np.mean([s["success"] for s in f[:100]])]
            ax[1].plot([0, 1], y, color=col[p], lw=2, marker="o", ms=5)
            ends.append([y[1], f"{p}/{g} tokens"])
    ends.sort()
    for k in range(1, len(ends)):
        ends[k][0] = max(ends[k][0], ends[k - 1][0] + 0.014)
    for y_, t in ends:
        ax[1].text(1.04, y_, t, fontsize=7.5, va="center", color=muted)
    ax[1].set_xticks([0, 1]); ax[1].set_xticklabels(["epoch 125", "epoch 250"]); ax[1].set_xlim(-0.2, 1.35); ax[1].set_ylim(0, 0.42)
    ax[1].set_ylabel("success (first 100 scenes)"); ax[1].set_title("Convergence: mid vs final checkpoint", fontsize=10, color=ink)
    # C: success by nearest-demo distance tercile, pooled over the tokens of each density
    z = zarr.open(os.path.join(ROOT, "R3D", "data", "square_d2-agentview_p4096-200.zarr"), "r"); tr = z["meta/init_layout"][:][:, [0, 1, 7, 8]]
    ref = load(1024, 64, "250_200ep_block1.json"); ev = np.array([s["layout"] for s in ref])[:, [0, 1, 7, 8]]
    d = np.sqrt(((ev[:, None] - tr[None]) ** 2).sum(-1)).min(1) * 100; q = np.quantile(d, [1 / 3, 2 / 3]); b = np.digitize(d, q)
    labels = [f"< {q[0]:.1f} cm", f"{q[0]:.1f}-{q[1]:.1f} cm", f"> {q[1]:.1f} cm"]
    for i, p in enumerate(a.points):
        runs = [load(p, g, "250_200ep_block1.json") for g in TOK[p]]; runs = [r for r in runs if r]
        if not runs: continue
        fr = np.mean([[s["success"] for s in r] for r in runs], 0)
        x = np.arange(3) + (i - (len(a.points) - 1) / 2) * w
        ax[2].bar(x, [fr[b == t].mean() for t in range(3)], w * 0.92, color=col[p], label=f"{p} points")
    ax[2].set_xticks(range(3)); ax[2].set_xticklabels(labels); ax[2].set_xlabel("eval scene -> nearest training demo (nut + peg)")
    ax[2].set_ylabel("success (mean over token counts)"); ax[2].set_ylim(0, 0.42); ax[2].legend(frameon=False, fontsize=8)
    ax[2].set_title("Success falls away from the demos", fontsize=10, color=ink)
    out = os.path.join(OUT, "mimicgen", "sweep1_results.png"); fig.savefig(out, dpi=140, bbox_inches="tight", facecolor=bg); print(out)


if __name__ == "__main__":
    main()
