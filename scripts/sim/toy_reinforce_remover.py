#!/usr/bin/env python
"""Step 0.3 of the learned link remover (docs/learned_remover_v0.md): toy REINFORCE test.
No policy, no GPU, seconds.  Run in r3d_sim.

    python scripts/sim/toy_reinforce_remover.py                    # full sweep + pass/fail
    python scripts/sim/toy_reinforce_remover.py --lrs 1e-2 --noise 3

The diffusion loss is replaced by a synthetic per-draw reward with a KNOWN answer:

    L_ik = b_i + sum_l w_l * f_l(rho_ik) + eps_ik
    f_l   = round(rho_l * n_l) / n_l       (the kept FRACTION, discretised as the real remover)
    b_i   ~ N(0, window_spread^2)          per window, shared by its K draws (RLOO cancels it)
    eps   ~ N(0, (noise * effect)^2)       per draw (the part of the real loss that is not the
                                            mask: encoder FPS start, thinning FPS start, ...)

w = +effect on --cut (default link6: keeping it RAISES the loss -> thin it to 0) and
w = -effect on --keep (default the four finger regions) (thinning them raises the loss -> keep everything); all other
regions w = 0. n_l are the region sizes measured by link_stats.py (mean at --num-points)
when the report exists, else a stated guess.

Everything else is the real training machinery: LinkRemover (logit-normal, clamp to
[-8, 6]), K = 2 draws per window, rloo_surrogate, Adam on mu, sigma annealed 1.0 -> 0.2.

Two criteria per run (neutral regions' mu must stay within 1 logit for both):
  PATTERN  rho_cut < 0.1 and every finger rho > 0.9
  COMPLETE the --cut region keeps 0 points (deterministic) and every finger keeps >= 98%
COMPLETE is expected to be slow: the score-function signal for region l scales with
w_l * rho_l (1 - rho_l), which vanishes as the sigmoid saturates. PASS = the PATTERN is
learned on every seed at some Adam lr, the mirrored weights give the mirrored result, and
plain REINFORCE (no baseline: adv = L) does worse. A last run over a real run's phi budget
(--budget) reports how far the --cut region gets and how far the zero-signal regions random-walk
(Adam steps ~lr per update whatever the gradient's size).

The brief proposes lr_phi = 0.1 * lr_theta = 1e-5. With Adam, mu moves at most ~lr per
update, and a Square run gives ~14k phi updates (251 epochs x 224 optimizer steps / 4):
<= 0.14 logit in total. The sweep therefore reports updates-to-pass per lr.
"""

import argparse
import json
import math
import os
import sys

import numpy as np
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "R3D"))
from r3d.common import robosuite_pcd as RP  # noqa: E402
from r3d.model.vision.link_remover import LinkRemover, rloo_surrogate  # noqa: E402

GOOD_TO_CUT = "link6"      # set by --cut: present in every frame, the largest region
FINGERS = ("finger1_prox", "finger1_dist", "finger2_prox", "finger2_dist")   # set by --keep


def region_sizes(names, num_points):
    p = os.path.join(ROOT, "R3D", "data", "outputs", "sim", "link_labels", "square_d2-agentview_p4096-200_stats.json")
    if os.path.exists(p):
        reg = json.load(open(p))["per_N"][str(num_points)]["regions"]
        return np.array([max(1.0, round(reg[nm]["mean"])) for nm in names]), p
    return np.full(len(names), 100.0), "guess: 100 points per region"


def run(names, n_l, w, lr, noise, effect, updates, B=64, K=2, baseline=True, seed=0,
        sigma0=1.0, sigma1=0.2, anneal=None, window_spread=None, opt="adam"):
    """-> (first update with the PATTERN, first update with COMPLETE removal, final state, history)"""
    torch.manual_seed(seed)
    rem = LinkRemover(RP.CELL_NAMES, RP.default_region_groups(), init_rho=0.8, sigma=sigma0)
    assert rem.region_names == names
    if opt == "adam":
        optim = torch.optim.Adam([rem.mu], lr=lr)
    elif opt == "sgd_norm":      # SGD on the surrogate with the advantage divided by its batch std
        optim = torch.optim.SGD([rem.mu], lr=lr)
    else:
        raise ValueError(opt)
    n_t, w_t = torch.as_tensor(n_l, dtype=torch.float64), torch.as_tensor(w, dtype=torch.float64)
    spread = 10 * effect if window_spread is None else window_spread
    anneal = min(updates // 2, 2000) if anneal is None else anneal
    mu0 = rem.mu.detach().clone()
    pattern_at = complete_at = None
    hist = []
    for u in range(updates):
        rem.set_sigma(sigma0 + (sigma1 - sigma0) * min(1.0, u / max(1, anneal)))
        z = rem.sample(B * K).view(B, K, -1)
        rho = torch.sigmoid(z).double()
        frac = torch.floor(rho * n_t + 0.5) / n_t
        loss = (spread * torch.randn(B, 1, dtype=torch.float64) + (frac * w_t).sum(-1)
                + noise * effect * torch.randn(B, K, dtype=torch.float64))
        lp = rem.log_prob(z)
        if baseline:
            surr, adv = rloo_surrogate(lp, loss.float())
        else:
            adv = loss.float()
        if opt == "sgd_norm":
            adv = adv / adv.std().clamp(min=1e-12)
        surr = (adv * lp).mean()
        optim.zero_grad()
        surr.backward()
        optim.step()
        rem.clamp_()
        st = state(rem, names, n_t, mu0)
        hist.append(st)
        if pattern_at is None and st["pattern"]:
            pattern_at = u + 1
        if complete_at is None and st["complete"]:
            complete_at = u + 1
    return pattern_at, complete_at, hist[-1], hist


def state(rem, names, n_t, mu0):
    """PATTERN: rho_cut < 0.1, every finger rho > 0.9, neutral |d mu| <= 1.
    COMPLETE: the --cut region keeps 0 points (deterministic), fingers >= 98%, neutral |d mu| <= 1."""
    rho = rem.rho().double()
    keep = torch.floor(rho * n_t + 0.5)
    i3 = names.index(GOOD_TO_CUT)
    fi = [names.index(f) for f in FINGERS]
    neutral = [i for i in range(len(names)) if i != i3 and i not in fi]
    drift = (rem.mu.detach() - mu0)[neutral].abs().max().item()
    pattern = rho[i3].item() < 0.1 and all(rho[i].item() > 0.9 for i in fi) and drift <= 1.0
    complete = keep[i3].item() == 0 and all(keep[i].item() >= 0.98 * n_t[i].item() for i in fi) and drift <= 1.0
    return {"pattern": pattern, "complete": complete, "mu": rem.mu.detach().tolist(),
            "rho_cut": rho[i3].item(), "keep_cut": keep[i3].item(),
            "finger_rho_min": min(rho[i].item() for i in fi), "neutral_max_drift": drift}


def main():
    global GOOD_TO_CUT, FINGERS
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lrs", type=float, nargs="+", default=[1e-5, 1e-3, 3e-3, 1e-2, 3e-2])
    ap.add_argument("--noise", type=float, nargs="+", default=[1.0, 3.0, 10.0],
                    help="per-draw noise std in units of the full effect size")
    ap.add_argument("--effect", type=float, default=0.01, help="loss change for rho 0 -> 1 of one region")
    ap.add_argument("--updates", type=int, default=1000)
    ap.add_argument("--num-points", type=int, default=4096)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--cut", default=GOOD_TO_CUT, help="region whose removal lowers the loss")
    ap.add_argument("--keep", nargs="+", default=list(FINGERS), help="regions whose thinning raises it")
    ap.add_argument("--sgd-lrs", type=float, nargs="+", default=[0.1, 0.3, 1.0],
                    help="SGD with the advantage normalised by its batch std (comparison to Adam)")
    ap.add_argument("--budget", type=int, default=14000, help="phi updates of a real Square run")
    a = ap.parse_args()
    GOOD_TO_CUT, FINGERS = a.cut, tuple(a.keep)

    names = [g["name"] for g in RP.default_region_groups()]
    n_l, src = region_sizes(names, a.num_points)
    print(f"{len(names)} regions; sizes from {src}:\n  " + ", ".join(f"{nm} {int(n)}" for nm, n in zip(names, n_l)))
    w = np.zeros(len(names))
    w[names.index(GOOD_TO_CUT)] = a.effect
    for f in FINGERS:
        w[names.index(f)] = -a.effect

    # 0. gradient sign, analytically: d/dmu of the surrogate = mean(adv * (z - mu) / sigma^2)
    rem = LinkRemover(RP.CELL_NAMES, RP.default_region_groups(), sigma=0.5)
    z = rem.sample(8).view(4, 2, -1)
    loss = torch.randn(4, 2)
    surr, adv = rloo_surrogate(rem.log_prob(z), loss)
    surr.backward()
    expect = (adv[..., None] * (z - rem.mu.detach()) / 0.25).mean((0, 1))
    assert torch.allclose(rem.mu.grad, expect, atol=1e-6), "surrogate gradient != (z - mu) / sigma^2 * adv"
    assert torch.allclose(adv[:, 0], loss[:, 0] - loss[:, 1]) and torch.allclose(adv[:, 1], -adv[:, 0])
    print("gradient check: d surrogate / d mu = mean(adv (z - mu) / sigma^2); K=2 adv = L1 - L2: OK")

    results = {"regions": names, "sizes": n_l.tolist(), "effect": a.effect, "updates": a.updates, "runs": []}
    fmt = lambda v: "-" if v is None else str(v)
    print(f"\nupdates to PATTERN / COMPLETE (of {a.updates}; '-' = never), {a.seeds} seeds, RLOO, B=64, K=2;"
          f" then final rho_cut | finger rho min | neutral drift (seed 0)")
    for opt, lrs in (("adam", a.lrs), ("sgd_norm", a.sgd_lrs)):
        for lr in lrs:
            for nz in a.noise:
                pats, comps = [], []
                for s in range(a.seeds):
                    p, c, last, _ = run(names, n_l, w, lr, nz, a.effect, a.updates, seed=s, opt=opt)
                    pats.append(p)
                    comps.append(c)
                    results["runs"].append({"opt": opt, "lr": lr, "noise": nz, "seed": s, "pattern_at": p,
                                            "complete_at": c, "final": last})
                    if s == 0:
                        last0 = last
                print(f"{opt:8s} lr {lr:7.0e} noise {nz:4g}x | pattern {' '.join(map(fmt, pats)):18s} | complete "
                      f"{' '.join(map(fmt, comps)):18s} | {last0['rho_cut']:.3f} | {last0['finger_rho_min']:.3f} | "
                      f"{last0['neutral_max_drift']:.2f}", flush=True)

    # choose the Adam lr whose PATTERN arrives on every seed at the highest noise, fastest
    by = {}
    for r in results["runs"]:
        if r["opt"] == "adam" and r["pattern_at"] is not None:
            by.setdefault((r["lr"], r["noise"]), []).append(r["pattern_at"])
    good = sorted([k for k, v in by.items() if len(v) == a.seeds], key=lambda k: (-k[1], np.median(by[k])))
    if not good:
        raise SystemExit("toy REINFORCE FAILED: no Adam lr learns the pattern on every seed")
    lr, nz = good[0]
    print(f"\ncontrols at Adam lr {lr:g}, noise {nz:g}x:")
    _, _, last_m, _ = run(names, n_l, -w, lr, nz, a.effect, a.updates, seed=0)
    i3 = names.index(GOOD_TO_CUT)
    fi = [names.index(f) for f in FINGERS]
    rho_m = 1 / (1 + np.exp(-np.array(last_m["mu"])))
    mirror_ok = rho_m[i3] > 0.9 and all(rho_m[i] < 0.1 for i in fi)
    print(f"  mirrored weights: rho {GOOD_TO_CUT} {rho_m[i3]:.3f}, keep regions {np.round(rho_m[fi], 3).tolist()} -> "
          f"{'mirrored OK' if mirror_ok else 'NOT MIRRORED'}")
    p_b, _, last_b, _ = run(names, n_l, w, lr, nz, a.effect, a.updates, seed=0, baseline=False)
    p_r = int(np.median(by[(lr, nz)]))
    base_ok = p_b is None or p_b > p_r
    print(f"  plain REINFORCE (no baseline): pattern at {p_b} vs RLOO median {p_r}; final rho_cut "
          f"{last_b['rho_cut']:.3f}, drift {last_b['neutral_max_drift']:.2f} -> "
          f"{'baseline helps OK' if base_ok else 'BASELINE DOES NOT HELP'}")
    # a real run's phi budget: neutral drift and how far the --cut region gets
    long = {}
    for opt, l in (("adam", lr), ("sgd_norm", a.sgd_lrs[len(a.sgd_lrs) // 2])):
        p, c, last_l, h = run(names, n_l, w, l, nz, a.effect, a.budget, seed=0, opt=opt)
        long[opt] = {"lr": l, "pattern_at": p, "complete_at": c, "final": last_l,
                     "neutral_drift_at": {str(u): h[u - 1]["neutral_max_drift"] for u in (1000, 5000, a.budget) if u <= a.budget}}
        print(f"  {a.budget} updates ({opt}, lr {l:g}): pattern {fmt(p)}, complete {fmt(c)}, final rho_cut "
              f"{last_l['rho_cut']:.4f} (keeps {int(last_l['keep_cut'])}), fingers min {last_l['finger_rho_min']:.3f}, "
              f"neutral drift " + ", ".join(f"{k}: {v:.2f}" for k, v in long[opt]["neutral_drift_at"].items()), flush=True)
    results["controls"] = {"lr": lr, "noise": nz, "mirrored_ok": bool(mirror_ok), "mirrored_final": last_m,
                           "no_baseline_pattern_at": p_b, "rloo_pattern_at_median": p_r, "baseline_ok": bool(base_ok),
                           "long_run": long}
    out = os.path.join(ROOT, "R3D", "data", "outputs", "sim", "link_labels", "toy_reinforce.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as fh:
        json.dump(results, fh, indent=1)
    print(f"report: {out}")
    if not (mirror_ok and base_ok):
        raise SystemExit("toy REINFORCE controls FAILED")
    print("toy REINFORCE: PASS (pattern learned, mirrored, baseline helps)")


if __name__ == "__main__":
    main()
