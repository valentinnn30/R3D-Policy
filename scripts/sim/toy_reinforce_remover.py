#!/usr/bin/env python
"""Step 0.3 of the learned link remover (docs/learned_remover_v0.md): toy REINFORCE test.
No policy, no GPU, seconds.  Run in r3d_sim.

    python scripts/sim/toy_reinforce_remover.py                    # sweep + controls + pass/fail
    python scripts/sim/toy_reinforce_remover.py --compare          # drift vs learning over a run's budget

The remover draws, per region l, a removal gate g_l ~ Bern(sigmoid(a_l)) and a density
z_l ~ N(mu_l, sigma^2); the region keeps k_l = g_l * round(sigmoid(z_l) * n_l) points. The
diffusion loss is replaced by a synthetic per-draw reward with a KNOWN answer, shaped like
the measured mechanism (FPS puts tokens by surface extent, so only REMOVAL moves tokens;
thinning only changes patch size -- link_remover.py header):

    L_ik = b_i + sum_l w_l * 1[k_l > 0] + sum_l v_l * k_l / n_l + eps_ik

    w = +effect on --cut (default link6: keeping it at all RAISES the loss -> remove it),
        -effect on --keep (default both fingers: removing them raises it), 0 elsewhere
    v = --density-effect * effect on every region (default 0: density is a pure nuisance)
    b_i ~ N(0, (10 effect)^2) per window, shared by its K draws (RLOO cancels it)
    eps ~ N(0, (noise * effect)^2) per draw

n_l are the region sizes measured by link_stats.py (mean at --num-points). Everything else
is the real machinery: LinkRemover (bounds mu [-9, 6], gate [-5, 5]), K draws per window,
rloo_surrogate, Adam on (gate, mu), sigma annealed 1.0 -> 0.2.

PATTERN = keep prob of --cut < 0.1, of every --keep > 0.9, and no neutral region's gate
logit moved by more than 1. PASS = the PATTERN is learned on every seed at some Adam lr,
the mirrored weights give the mirrored result, and plain REINFORCE (no baseline) does
worse. Drift is reported for both logits: a neutral GATE that drifts closed removes a
region by noise; the density logits are all neutral here (v = 0) and show pure drift.
"""

import argparse
import json
import os
import sys

import numpy as np
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "R3D"))
from r3d.common import robosuite_pcd as RP  # noqa: E402
from r3d.model.vision.link_remover import LinkRemover, rloo_surrogate  # noqa: E402

OUT = os.path.join(ROOT, "R3D", "data", "outputs", "sim", "link_labels")


def region_sizes(names, num_points):
    p = os.path.join(OUT, "square_d2-agentview_p4096-200_stats_v0.json")
    if os.path.exists(p):
        reg = json.load(open(p))["per_N"][str(num_points)]["regions"]
        return np.array([max(1.0, round(reg[nm]["mean"])) for nm in names]), p
    return np.full(len(names), 100.0), "guess: 100 points per region"


class Toy:
    def __init__(self, names, n_l, cut, keep, effect, density_effect):
        self.names, self.cut, self.keep = names, cut, keep
        self.ic, self.ik = names.index(cut), [names.index(k) for k in keep]
        self.neutral = [i for i in range(len(names)) if i != self.ic and i not in self.ik]
        self.n = torch.as_tensor(n_l, dtype=torch.float64)
        self.w = torch.zeros(len(names), dtype=torch.float64)
        self.w[self.ic] = effect
        self.w[self.ik] = -effect
        self.v = torch.full((len(names),), density_effect * effect, dtype=torch.float64)
        self.effect = effect

    def loss(self, rem, draw, noise, spread):
        B, K = draw.shape[:2]
        k = torch.floor(rem.rho(draw).double() * self.n + 0.5)
        return (spread * torch.randn(B, 1, dtype=torch.float64) + ((k > 0).double() * self.w).sum(-1)
                + (k / self.n * self.v).sum(-1) + noise * self.effect * torch.randn(B, K, dtype=torch.float64))

    def state(self, rem, a0, mu0):
        kp = rem.keep_prob().double()
        dg = (rem.gate.detach() - a0)[self.neutral].abs()
        dm = (rem.mu.detach() - mu0).abs()
        pattern = kp[self.ic] < 0.1 and all(kp[i] > 0.9 for i in self.ik) and dg.max() <= 1.0
        return {"pattern": bool(pattern), "keep_cut": kp[self.ic].item(),
                "keep_kept": [kp[i].item() for i in self.ik],
                "neutral_gate_drift_max": dg.max().item(), "neutral_gate_drift_med": dg.median().item(),
                "neutral_keep_prob": kp[self.neutral].tolist(),
                "mu_drift_max": dm.max().item(), "gate": rem.gate.detach().tolist(), "mu": rem.mu.detach().tolist()}


def run(toy, lr, noise, updates, B=64, K=2, baseline=True, seed=0, sigma0=1.0, sigma1=0.2,
        anneal=None, accum=1, mirror=False):
    """-> (first step with the PATTERN, final state, history)."""
    torch.manual_seed(seed)
    rem = LinkRemover(RP.CELL_NAMES, RP.default_region_groups(), init_rho=0.8, init_keep=0.8, sigma=sigma0)
    assert rem.region_names == toy.names
    opt = torch.optim.Adam([rem.gate, rem.mu], lr=lr)
    a0, mu0 = rem.gate.detach().clone(), rem.mu.detach().clone()
    anneal = min(updates // 2, 2000) if anneal is None else anneal
    sign = -1.0 if mirror else 1.0
    pattern_at, hist = None, []
    for u in range(updates):
        rem.set_sigma(sigma0 + (sigma1 - sigma0) * min(1.0, u / max(1, anneal)))
        draw = rem.sample(B * K).view(B, K, -1)
        loss = sign * toy.loss(rem, draw, noise, 10 * toy.effect)
        lp = rem.log_prob(draw)
        if baseline:
            surr, _ = rloo_surrogate(lp, loss.float())
        else:
            surr = (loss.float() * lp).mean()
        (surr / accum).backward()
        if (u + 1) % accum == 0:
            opt.step()
            opt.zero_grad()
            rem.clamp_()
        st = toy.state(rem, a0, mu0)
        hist.append(st)
        if pattern_at is None and st["pattern"]:
            pattern_at = u + 1
    return pattern_at, hist[-1], hist


VARIANTS = [   # name, kwargs -- reward compute per phi step relative to B=64, K=2 in brackets
    ("A adam 3e-2 B64 K2", dict(lr=3e-2, B=64, K=2)),                  # [1x]
    ("B adam 1e-2 B64 K2", dict(lr=1e-2, B=64, K=2)),                  # [1x]
    ("C adam 1e-2 B128 K2", dict(lr=1e-2, B=128, K=2)),                # [2x]
    ("D adam 1e-2 B64 K4", dict(lr=1e-2, B=64, K=4)),                  # [2x]
    ("F adam 1e-2 B128 K4", dict(lr=1e-2, B=128, K=4)),                # [4x]
    ("G adam 3e-3 B128 K4", dict(lr=3e-3, B=128, K=4)),                # [4x]  config default
    ("I adam 1e-3 B128 K4", dict(lr=1e-3, B=128, K=4)),                # [4x]
]


def compare(a, toy):
    print(f"\nover {a.budget} phi steps, {a.seeds} seeds; cut {toy.cut} / keep {toy.keep} / "
          f"{len(toy.neutral)} neutral gates; density effect {a.density_effect}")
    print(f"{'variant':24s} {'noise':>5s} | {'pattern@':>16s} | {'keep prob cut':>17s} | {'keep prob kept min':>18s} | "
          f"{'neutral gate drift max/med':>27s} | {'neutral gates closed':>20s} | {'mu drift max':>14s}")
    rows = []
    for nm, kw in VARIANTS:
        for nz in a.noise:
            res = [run(toy, noise=nz, updates=a.budget, seed=sd, **kw) for sd in range(a.seeds)]
            last = [r[1] for r in res]
            f = lambda vals, p=3: " ".join(f"{v:.{p}f}" for v in vals)
            closed = [sum(k < 0.5 for k in L["neutral_keep_prob"]) for L in last]
            print(f"{nm:24s} {nz:4g}x | {' '.join('-' if r[0] is None else str(r[0]) for r in res):>16s} | "
                  f"{f([L['keep_cut'] for L in last]):>17s} | {f([min(L['keep_kept']) for L in last], 2):>18s} | "
                  f"{f([L['neutral_gate_drift_max'] for L in last], 1)} / {f([L['neutral_gate_drift_med'] for L in last], 1)} | "
                  f"{' '.join(map(str, closed)):>20s} | {f([L['mu_drift_max'] for L in last], 1):>14s}", flush=True)
            rows.append({"variant": nm, "noise": nz, "pattern_at": [r[0] for r in res], "final": last})
    with open(os.path.join(OUT, "toy_reinforce_compare.json"), "w") as fh:
        json.dump({"regions": toy.names, "budget": a.budget, "rows": rows}, fh, indent=1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lrs", type=float, nargs="+", default=[1e-5, 1e-3, 3e-3, 1e-2, 3e-2])
    ap.add_argument("--noise", type=float, nargs="+", default=[1.0, 3.0, 10.0],
                    help="per-draw noise std in units of the effect size")
    ap.add_argument("--effect", type=float, default=0.01, help="loss change when a region is removed")
    ap.add_argument("--density-effect", type=float, default=0.0,
                    help="loss change for density 0 -> 1, in units of --effect (0: density is a nuisance)")
    ap.add_argument("--updates", type=int, default=1000)
    ap.add_argument("--num-points", type=int, default=1024)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--cut", default="link6", help="region whose removal lowers the loss")
    ap.add_argument("--keep", nargs="+", default=["finger1", "finger2"], help="regions whose removal raises it")
    ap.add_argument("--budget", type=int, default=14000, help="phi steps of a real Square run")
    ap.add_argument("--compare", action="store_true", help="only the drift-vs-learning comparison of VARIANTS")
    a = ap.parse_args()

    names = [g["name"] for g in RP.default_region_groups()]
    n_l, src = region_sizes(names, a.num_points)
    print(f"{len(names)} regions; sizes from {src}:\n  " + ", ".join(f"{nm} {int(n)}" for nm, n in zip(names, n_l)))
    toy = Toy(names, n_l, a.cut, a.keep, a.effect, a.density_effect)
    os.makedirs(OUT, exist_ok=True)
    if a.compare:
        return compare(a, toy)

    # gradient sign, analytically: d/da = mean(adv (g - pi)), d/dmu = mean(adv g (z - mu)) / sigma^2
    rem = LinkRemover(RP.CELL_NAMES, RP.default_region_groups(), sigma=0.5)
    draw = rem.sample(8).view(4, 2, -1)
    loss = torch.randn(4, 2)
    surr, adv = rloo_surrogate(rem.log_prob(draw), loss)
    surr.backward()
    z, g = rem.split(draw)
    pi = torch.sigmoid(rem.gate.detach())
    assert torch.allclose(rem.gate.grad, (adv[..., None] * (g - pi)).mean((0, 1)), atol=1e-6)
    assert torch.allclose(rem.mu.grad, (adv[..., None] * g * (z - rem.mu.detach()) / 0.25).mean((0, 1)), atol=1e-6)
    assert torch.allclose(adv[:, 0], loss[:, 0] - loss[:, 1]) and torch.allclose(adv[:, 1], -adv[:, 0])
    print("gradient check: d/da = mean(adv (g - pi)), d/dmu = mean(adv g (z - mu)) / sigma^2, "
          "K=2 adv = L1 - L2: OK")

    results = {"regions": names, "sizes": n_l.tolist(), "effect": a.effect, "updates": a.updates, "runs": []}
    print(f"\nsteps to PATTERN (of {a.updates}; '-' = never), {a.seeds} seeds, RLOO, B=64, K=2; then final "
          f"keep prob cut | keep prob kept min | neutral gate drift max (seed 0)")
    by = {}
    for lr in a.lrs:
        for nz in a.noise:
            ps, last0 = [], None
            for s in range(a.seeds):
                p, last, _ = run(toy, lr, nz, a.updates, seed=s)
                ps.append(p)
                last0 = last0 or last
                results["runs"].append({"lr": lr, "noise": nz, "seed": s, "pattern_at": p, "final": last})
            if all(p is not None for p in ps):
                by[(lr, nz)] = ps
            print(f"adam lr {lr:7.0e} noise {nz:4g}x | {' '.join('-' if p is None else str(p) for p in ps):18s} | "
                  f"{last0['keep_cut']:.3f} | {min(last0['keep_kept']):.3f} | {last0['neutral_gate_drift_max']:.2f}",
                  flush=True)
    if not by:
        raise SystemExit("toy REINFORCE FAILED: no lr learns the pattern on every seed")
    lr, nz = sorted(by, key=lambda k: (-k[1], np.median(by[k])))[0]
    print(f"\ncontrols at Adam lr {lr:g}, noise {nz:g}x:")
    _, last_m, _ = run(toy, lr, nz, a.updates, mirror=True)
    mirror_ok = last_m["keep_cut"] > 0.9 and max(last_m["keep_kept"]) < 0.1
    print(f"  mirrored weights: keep prob {toy.cut} {last_m['keep_cut']:.3f}, {toy.keep} "
          f"{np.round(last_m['keep_kept'], 3).tolist()} -> {'mirrored OK' if mirror_ok else 'NOT MIRRORED'}")
    p_b, last_b, _ = run(toy, lr, nz, a.updates, baseline=False)
    p_r = int(np.median(by[(lr, nz)]))
    base_ok = p_b is None or p_b > p_r
    print(f"  plain REINFORCE (no baseline): pattern at {p_b} vs RLOO median {p_r}; final keep prob cut "
          f"{last_b['keep_cut']:.3f} -> {'baseline helps OK' if base_ok else 'BASELINE DOES NOT HELP'}")
    results["controls"] = {"lr": lr, "noise": nz, "mirrored_ok": bool(mirror_ok), "no_baseline_pattern_at": p_b,
                           "rloo_pattern_at_median": p_r, "baseline_ok": bool(base_ok)}
    with open(os.path.join(OUT, "toy_reinforce.json"), "w") as fh:
        json.dump(results, fh, indent=1)
    if not (mirror_ok and base_ok):
        raise SystemExit("toy REINFORCE controls FAILED")
    print("toy REINFORCE: PASS (pattern learned, mirrored, baseline helps)")


if __name__ == "__main__":
    main()
