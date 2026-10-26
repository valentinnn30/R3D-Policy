#!/usr/bin/env python
"""Analytics of learned-link-remover runs (docs/learned_remover_design.md).  Run in r3d_sim.

    python scripts/sim/analyze_remover_run.py <run_dir> [<run_dir> ...] [--sweep-out <dir>]

Reads each run's `remover_log.jsonl` (written by r3d/common/link_remover_training.py), its
`.hydra/config.yaml` and `eval/*_200ep_block1.json`, and the matching baselines (same point
count, 64 tokens, seed: `mimicgen_sim-square_d2-p<N>-g64[-cut5]-<seed>`), and writes to
<run_dir>/analysis/:

  phi.png          gate keep probability, density rho (+ deterministic rho) per region vs
                   training progress; phases shaded (warm-up | phi | fine-tune)
  logits.png       gate logits a and density logits mu (drift), their phi gradients
  reward.png       sigma, reward mean / std, advantage std, phi windows per step
  effects.png      per-region removal effect (mean advantage gate closed - open; < 0 =
                   removing helps) and density slope, cumulative mean +- 95% CI
  diagnostics.png  held-out check (rank corr, grad cosine, sign agreement: floor vs
                   held-out) and diagnostic B (proxy vs fully sampled rank corr) per epoch
  mechanism.png    token centres per class, coverage of nut / peg, patch radius per class
                   on the uncut / deterministic / sampled cloud, per diagnostic epoch
  training.png     train / val loss, kept fraction per region (theta pass), points kept,
                   fill-up rate per epoch
  success.png      success vs checkpoint epoch (200 fixed scenes) against the baselines
  report.md        all final numbers as tables, incl. paired McNemar tests vs the baselines
  summary.json     the same, machine-readable

With several runs, `--sweep-out` (default R3D/data/outputs/sim/mimicgen/remover_sweep/) gets
sweep_report.md + sweep_success.png + sweep_gates.png across runs.
"""

import argparse
import glob
import json
import math
import os
import re
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SIM_OUT = os.path.join(ROOT, "R3D", "data", "outputs", "sim")
PHASE_COLOURS = {"warmup": "#e8e8e8", "finetune": "#fde8c8"}


def load_jsonl(path):
    phi, ep = [], []
    with open(path) as fh:
        for line in fh:
            r = json.loads(line)
            (phi if r["kind"] == "phi" else ep).append(r)
    return phi, ep


def read_cfg(run):
    import yaml
    with open(os.path.join(run, ".hydra", "config.yaml")) as fh:
        cfg = yaml.safe_load(fh)
    return cfg


def mcnemar(a, b):
    """exact two-sided McNemar on paired booleans -> (a only, b only, p)."""
    n01 = int(np.sum(a & ~b))
    n10 = int(np.sum(~a & b))
    n = n01 + n10
    if n == 0:
        return n01, n10, 1.0
    k = min(n01, n10)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n * 2
    return n01, n10, min(1.0, p)


def eval_results(run):
    out = {}
    for f in glob.glob(os.path.join(run, "eval", "*_200ep_block1.json")):
        ep = int(os.path.basename(f).split("_")[0])
        d = json.load(open(f))
        out[ep] = {"success_rate": d["success_rate"], "succ": {r["seed"]: r["success"] for r in d["scenes"]},
                   "remover_rho": d.get("remover_rho")}
    return dict(sorted(out.items()))


def baselines(N, seed):
    res = {}
    for name, suffix in (("uncut", ""), ("arm cut", "-cut5")):
        for s in sorted({seed, 42, 43}):
            d = os.path.join(SIM_OUT, f"mimicgen_sim-square_d2-p{N}-g64{suffix}-{s}")
            r = eval_results(d) if os.path.isdir(d) else {}
            if 250 in r:
                res[f"{name} s{s}"] = r[250]
    return res


def shade(ax, cfg, x_max):
    E = cfg["training"]["num_epochs"]
    rl = cfg["training"]["link_remover"]
    w = round(rl["warmup_frac"] * E)
    f = E - round(rl["finetune_frac"] * E)
    ax.axvspan(0, w, color=PHASE_COLOURS["warmup"], zorder=0)
    ax.axvspan(f, max(x_max, E), color=PHASE_COLOURS["finetune"], zorder=0)
    return w, f


def cum_ci(vals):
    v = np.asarray(vals, float)
    n = np.arange(1, len(v) + 1)
    m = np.cumsum(v) / n
    sq = np.cumsum(v ** 2) / n
    se = np.sqrt(np.maximum(sq - m ** 2, 0) / np.maximum(n - 1, 1))
    return m, 1.96 * se


def analyze(run):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cfg = read_cfg(run)
    N = int(cfg["task"]["num_points"])
    seed = int(cfg["training"]["seed"])
    phi, ep = load_jsonl(os.path.join(run, "remover_log.jsonl"))
    if not ep:
        raise RuntimeError(f"{run}: remover_log.jsonl has no epoch records")
    out = os.path.join(run, "analysis")
    os.makedirs(out, exist_ok=True)
    regions = list(ep[-1]["phi"].keys())
    E = cfg["training"]["num_epochs"]
    cmap10 = plt.get_cmap("tab10")
    cmap = lambda i: cmap10(i) if len(regions) + 4 <= 10 else plt.get_cmap("tab20")(i % 20)
    xs = [r["epoch_frac"] for r in phi]
    summary = {"run": run, "N": N, "seed": seed, "regions": regions, "phi_steps": len(phi), "epochs_logged": len(ep)}

    # ---- phi trajectories ---------------------------------------------------------------
    fig, axs = plt.subplots(1, 3, figsize=(20, 5), sharex=True)
    for ax, key, title in zip(axs, ("keep_prob", "rho", "det_rho"),
                              ("gate keep probability", "density rho = sigmoid(mu)", "deterministic rho (inference)")):
        shade(ax, cfg, E)
        ex = [r["epoch"] for r in ep]
        for i, nm in enumerate(regions):
            ax.plot(ex, [r["phi"][nm][key] for r in ep], color=cmap(i), label=nm)
        ax.set_title(title)
        ax.set_xlabel("epoch")
        ax.set_ylim(-0.02, 1.02)
    axs[0].axhline(0.5, color="k", lw=0.6, ls="--")
    axs[-1].legend(fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "phi.png"), dpi=90)
    plt.close(fig)

    # ---- logits + gradients -------------------------------------------------------------
    if phi:
        fig, axs = plt.subplots(2, 2, figsize=(18, 9), sharex=True)
        for i, nm in enumerate(regions):
            c = cmap(i)
            axs[0, 0].plot(xs, [r["phi"][nm]["gate"] for r in phi], color=c, label=nm)
            axs[0, 1].plot(xs, [r["phi"][nm]["mu"] for r in phi], color=c, label=nm)
            for ax, key in ((axs[1, 0], "remover_grad_gate/"), (axs[1, 1], "remover_grad_mu/")):
                g = [r.get(key + nm, np.nan) for r in phi]
                k = max(1, len(g) // 100)
                ax.plot(xs[k - 1:], np.convolve(g, np.ones(k) / k, "valid"), color=c, lw=0.8)
        for ax, t in zip(axs.flat, ("gate logit a", "density logit mu", "d surrogate / d a (smoothed)",
                                    "d surrogate / d mu (smoothed)")):
            shade(ax, cfg, E)
            ax.set_title(t)
        axs[0, 0].legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(out, "logits.png"), dpi=90)
        plt.close(fig)

        # ---- reward / advantage -------------------------------------------------------------
        fig, axs = plt.subplots(1, 4, figsize=(22, 4.5), sharex=True)
        for ax, key in zip(axs, ("remover_sigma", "remover_reward_mean", "remover_adv_std", "remover_reward_std")):
            shade(ax, cfg, E)
            ax.plot(xs, [r.get(key, np.nan) for r in phi], lw=0.7)
            ax.set_title(key.replace("remover_", ""))
        fig.tight_layout()
        fig.savefig(os.path.join(out, "reward.png"), dpi=90)
        plt.close(fig)

        # ---- per-region effects -------------------------------------------------------------
        fig, axs = plt.subplots(1, 2, figsize=(18, 5), sharex=True)
        eff = {}
        for i, nm in enumerate(regions):
            c = cmap(i)
            for ax, key, kind in ((axs[0], "remover_effect_gate/", "gate"), (axs[1], "remover_effect_density/", "density")):
                pts = [(r["epoch_frac"], r[key + nm]) for r in phi if key + nm in r]
                if len(pts) < 3:
                    continue
                x, v = zip(*pts)
                m, ci = cum_ci(v)
                ax.plot(x, m, color=c, label=nm)
                ax.fill_between(x, m - ci, m + ci, color=c, alpha=0.15)
                late = np.asarray(v[len(v) // 2:])
                eff.setdefault(nm, {})[kind] = {
                    "mean_all": float(np.mean(v)), "ci95_all": float(ci[-1]), "n_steps": len(v),
                    "mean_2nd_half": float(late.mean()),
                    "t_2nd_half": float(late.mean() / (late.std(ddof=1) / math.sqrt(len(late)))) if len(late) > 2 and late.std() > 0 else None}
        for ax, t in zip(axs, ("removal effect: mean adv closed - open (< 0: removing helps)",
                               "density slope d adv / d z (< 0: denser helps)")):
            shade(ax, cfg, E)
            ax.axhline(0, color="k", lw=0.6)
            ax.set_title(t, fontsize=10)
            ax.set_xlabel("epoch")
        axs[0].legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(out, "effects.png"), dpi=90)
        plt.close(fig)
        summary["effects"] = eff

    # ---- diagnostics ------------------------------------------------------------------------
    diag = [r for r in ep if "remover_check/n_batches" in r]
    if diag:
        fig, axs = plt.subplots(1, 4, figsize=(22, 4.5), sharex=True)
        dx = [r["epoch"] for r in diag]
        for ax, m in zip(axs[:3], ("mask_rank_spearman", "grad_cosine", "grad_sign_agreement")):
            shade(ax, cfg, E)
            ax.plot(dx, [r.get(f"remover_check/floor_{m}", np.nan) for r in diag], "o-", label="floor (same windows, new seeds)")
            ax.plot(dx, [r.get(f"remover_check/heldout_{m}", np.nan) for r in diag], "s-", label="held-out demos")
            ax.set_title(m)
            ax.set_ylim(-1.05, 1.05)
        axs[0].legend(fontsize=8)
        shade(axs[3], cfg, E)
        axs[3].plot(dx, [r.get("remover_check/ab_rank_spearman", np.nan) for r in diag], "o-")
        axs[3].set_title("diagnostic B: proxy vs sampled-action rank corr")
        axs[3].set_ylim(-1.05, 1.05)
        fig.tight_layout()
        fig.savefig(os.path.join(out, "diagnostics.png"), dpi=90)
        plt.close(fig)
        summary["diagnostics_last"] = {k: v for k, v in diag[-1].items() if k.startswith("remover_check/")}
        summary["diagnostics_mean_phi_phase"] = {
            k: float(np.nanmean([r.get(k, np.nan) for r in diag if r["phase"] == "phi"]))
            for k in diag[-1] if k.startswith("remover_check/")} if any(r["phase"] == "phi" for r in diag) else {}

        # ---- mechanism ----------------------------------------------------------------------
        classes = regions + ["link0-4", "nut", "peg", "other"]
        fig, axs = plt.subplots(1, 3, figsize=(22, 5), sharex=True)
        for mask, ls in (("none", ":"), ("deterministic", "-"), ("sampled", "--")):
            for i, c in enumerate(classes):
                y = [r.get(f"mech/{mask}/tokens/{c}", np.nan) for r in diag]
                axs[0].plot(dx, y, ls, color=cmap(i), label=c if mask == "deterministic" else None)
            for c, col in (("nut", "#d62728"), ("peg", "#2ca02c")):
                axs[1].plot(dx, [r.get(f"mech/{mask}/coverage/{c}", np.nan) for r in diag], ls, color=col,
                            label=f"{c} ({mask})")
            for i, c in enumerate(classes):
                axs[2].plot(dx, [r.get(f"mech/{mask}/patch_cm/{c}", np.nan) for r in diag], ls, color=cmap(i))
        for ax, t in zip(axs, ("token centres per class (: uncut, - deterministic, -- sampled)",
                               "coverage of nut / peg", "median patch radius [cm] per class")):
            shade(ax, cfg, E)
            ax.set_title(t, fontsize=10)
        axs[0].legend(fontsize=7, ncol=2)
        axs[1].legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(os.path.join(out, "mechanism.png"), dpi=90)
        plt.close(fig)
        summary["mechanism_last"] = {k: v for k, v in diag[-1].items() if k.startswith("mech/")}

    # ---- training curves ----------------------------------------------------------------------
    fig, axs = plt.subplots(1, 3, figsize=(20, 4.5), sharex=True)
    ex = [r["epoch"] for r in ep]
    for k in ("train_loss", "val_loss"):
        axs[0].plot(ex, [r.get(k, np.nan) for r in ep], label=k)
    axs[0].set_yscale("log")
    axs[0].legend()
    for i, nm in enumerate(regions):
        axs[1].plot(ex, [r.get(f"remover_kept_frac/{nm}_epoch", np.nan) for r in ep], color=cmap(i), label=nm)
    axs[1].legend(fontsize=7)
    axs[2].plot(ex, [r.get("remover_points_kept_epoch", np.nan) for r in ep], label="points kept (theta pass)")
    ax2 = axs[2].twinx()
    ax2.plot(ex, [r.get("remover_filled_frac_epoch", np.nan) for r in ep], color="C3", label="fill-up rate")
    for ax, t in zip(axs, ("loss", "kept fraction per region (theta pass)", "points kept / fill-up rate")):
        shade(ax, cfg, E)
        ax.set_title(t)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "training.png"), dpi=90)
    plt.close(fig)

    # ---- success ------------------------------------------------------------------------------
    ev = eval_results(run)
    base = baselines(N, seed)
    summary["success"] = {str(k): v["success_rate"] for k, v in ev.items()}
    summary["baselines_250"] = {k: v["success_rate"] for k, v in base.items()}
    fig, ax = plt.subplots(figsize=(8, 5))
    shade(ax, cfg, E)
    if ev:
        ax.plot(list(ev), [v["success_rate"] for v in ev.values()], "o-", color="k", label="learned remover")
    for i, (k, v) in enumerate(base.items()):
        ax.axhline(v["success_rate"], color=cmap(i), ls="--", label=f"{k} (250)")
    ax.set_xlabel("checkpoint epoch")
    ax.set_ylabel("success (200 scenes)")
    ax.set_title(f"N = {N}, 64 tokens, seed {seed}")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "success.png"), dpi=90)
    plt.close(fig)

    tests = {}
    if 250 in ev:
        a = ev[250]["succ"]
        for k, v in base.items():
            common = sorted(set(a) & set(v["succ"]))
            x = np.array([a[s] for s in common])
            y = np.array([v["succ"][s] for s in common])
            n01, n10, p = mcnemar(x, y)
            tests[k] = {"scenes": len(common), "remover": float(x.mean()), "baseline": float(y.mean()),
                        "remover_only": n01, "baseline_only": n10, "p": p}
    summary["paired_250"] = tests
    summary["final_phi"] = ep[-1]["phi"]

    # ---- report -------------------------------------------------------------------------------
    L = [f"# Learned link remover: {os.path.basename(run)}", "",
         f"N = {N}, 64 tokens, seed {seed}; {len(phi)} phi steps, {len(ep)} epochs logged.", "",
         "## Final phi (deterministic = what evaluation uses)", "",
         "| region | keep prob | gate open | density rho | deterministic rho |", "|---|---|---|---|---|"]
    for nm, v in ep[-1]["phi"].items():
        L.append(f"| {nm} | {v['keep_prob']:.3f} | {'yes' if v['keep_prob'] >= 0.5 else '**no**'} | {v['rho']:.3f} | {v['det_rho']:.3f} |")
    if summary.get("effects"):
        L += ["", "## What the reward says per region (phi phase)", "",
              "removal effect = mean advantage gate closed - open (< 0: removing the region lowers the loss); "
              "t over the second half of the phi steps.", "",
              "| region | removal effect (all) +- 95% CI | 2nd half | t | density slope (all) |", "|---|---|---|---|---|"]
        for nm, v in summary["effects"].items():
            g, d = v.get("gate", {}), v.get("density", {})
            t = g.get("t_2nd_half")
            L.append(f"| {nm} | {g.get('mean_all', float('nan')):+.2e} +- {g.get('ci95_all', float('nan')):.1e} | "
                     f"{g.get('mean_2nd_half', float('nan')):+.2e} | {'' if t is None else f'{t:+.1f}'} | "
                     f"{d.get('mean_all', float('nan')):+.2e} |")
    if diag:
        L += ["", "## Diagnostics (mean over the phi phase | last)", "", "| metric | phi phase | last |", "|---|---|---|"]
        for k, v in summary["diagnostics_last"].items():
            L.append(f"| {k.split('/')[1]} | {summary['diagnostics_mean_phi_phase'].get(k, float('nan')):.2f} | {v:.2f} |")
        L += ["", "## Mechanism at the last diagnostic", "", "| class | tokens uncut | tokens det. | coverage uncut | coverage det. | patch cm uncut | patch cm det. |",
              "|---|---|---|---|---|---|---|"]
        mm = summary["mechanism_last"]
        for c in regions + ["link0-4", "nut", "peg", "other"]:
            g = lambda m, s: mm.get(f"mech/{m}/{s}/{c}", float("nan"))
            L.append(f"| {c} | {g('none', 'tokens'):.1f} | {g('deterministic', 'tokens'):.1f} | {g('none', 'coverage'):.2f} | "
                     f"{g('deterministic', 'coverage'):.2f} | {g('none', 'patch_cm'):.1f} | {g('deterministic', 'patch_cm'):.1f} |")
    L += ["", "## Success (200 fixed scenes, block 1)", "", "| checkpoint | success |", "|---|---|"]
    for k, v in ev.items():
        L.append(f"| {k} | {100 * v['success_rate']:.1f}% |")
    if tests:
        L += ["", "## Paired vs baselines at 250 (McNemar, exact)", "",
              "| baseline | scenes | remover | baseline | remover only | baseline only | p |", "|---|---|---|---|---|---|---|"]
        for k, v in tests.items():
            L.append(f"| {k} | {v['scenes']} | {100 * v['remover']:.1f}% | {100 * v['baseline']:.1f}% | {v['remover_only']} | "
                     f"{v['baseline_only']} | {v['p']:.3f} |")
    L += ["", "Plots: phi.png, logits.png, reward.png, effects.png, diagnostics.png, mechanism.png, training.png, success.png"]
    with open(os.path.join(out, "report.md"), "w") as fh:
        fh.write("\n".join(L) + "\n")
    with open(os.path.join(out, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=1)
    print(f"{run}: analysis -> {out}")
    return summary


def sweep(summaries, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(out, exist_ok=True)
    summaries = sorted(summaries, key=lambda s: (s["N"], s["seed"]))
    L = ["# Learned link remover sweep", "", "## Success at 250 (200 fixed scenes)", "",
         "| N | seed | remover | uncut | arm cut | vs uncut (rem only / base only, p) | vs arm cut |", "|---|---|---|---|---|---|---|"]
    for s in summaries:
        sc = s["success"].get("250")
        b = s["baselines_250"]
        pt = s.get("paired_250", {})
        f = lambda k: f"{pt[k]['remover_only']}/{pt[k]['baseline_only']}, p={pt[k]['p']:.3f}" if k in pt else "-"
        L.append(f"| {s['N']} | {s['seed']} | {'-' if sc is None else f'{100 * sc:.1f}%'} | "
                 f"{b.get('uncut s' + str(s['seed']), float('nan')) * 100:.1f}% | {b.get('arm cut s' + str(s['seed']), float('nan')) * 100:.1f}% | "
                 f"{f('uncut s' + str(s['seed']))} | {f('arm cut s' + str(s['seed']))} |")
    regions = summaries[0]["regions"]
    L += ["", "## Learned gates (keep prob) and densities at the end", "",
          "| N | " + " | ".join(regions) + " |", "|---|" + "---|" * len(regions)]
    for s in summaries:
        L.append(f"| {s['N']} | " + " | ".join(f"{s['final_phi'][r]['keep_prob']:.2f} / {s['final_phi'][r]['rho']:.2f}" for r in regions) + " |")
    with open(os.path.join(out, "sweep_report.md"), "w") as fh:
        fh.write("\n".join(L) + "\n")
    fig, ax = plt.subplots(figsize=(8, 5))
    for i, s in enumerate(summaries):
        ev = {int(k): v for k, v in s["success"].items()}
        if ev:
            ax.plot(list(ev), list(ev.values()), "o-", color=plt.get_cmap("tab10")(i), label=f"remover N={s['N']}")
        for j, (k, v) in enumerate(s["baselines_250"].items()):
            if k.endswith(f"s{s['seed']}"):
                ax.axhline(v, color=plt.get_cmap("tab10")(i), ls="--" if k.startswith("uncut") else ":", lw=0.8)
    ax.set_xlabel("checkpoint epoch")
    ax.set_ylabel("success")
    ax.set_title("remover (solid) vs uncut (--) and arm cut (:) at 250")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "sweep_success.png"), dpi=90)
    plt.close(fig)
    fig, axs = plt.subplots(1, 2, figsize=(14, 4.5))
    w = 0.8 / len(summaries)
    for i, s in enumerate(summaries):
        x = np.arange(len(regions)) + i * w
        axs[0].bar(x, [s["final_phi"][r]["keep_prob"] for r in regions], w, label=f"N={s['N']}")
        axs[1].bar(x, [s["final_phi"][r]["rho"] for r in regions], w, label=f"N={s['N']}")
    for ax, t in zip(axs, ("gate keep probability", "density rho")):
        ax.set_xticks(np.arange(len(regions)) + 0.4 - w / 2)
        ax.set_xticklabels(regions, rotation=30, fontsize=8)
        ax.set_title(t)
        ax.set_ylim(0, 1)
    axs[0].axhline(0.5, color="k", lw=0.6, ls="--")
    axs[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "sweep_gates.png"), dpi=90)
    plt.close(fig)
    print(f"sweep report -> {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--sweep-out", default=os.path.join(SIM_OUT, "mimicgen", "remover_sweep"))
    a = ap.parse_args()
    sums = []
    for r in a.runs:
        try:
            sums.append(analyze(os.path.abspath(r)))
        except Exception as e:      # one broken run must not hide the others
            print(f"{r}: analysis FAILED: {type(e).__name__}: {e}", file=sys.stderr)
    if len(sums) > 1 or (sums and a.sweep_out):
        sweep(sums, a.sweep_out)


if __name__ == "__main__":
    main()
