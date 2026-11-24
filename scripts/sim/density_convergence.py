#!/usr/bin/env python
"""Has a density-only remover run converged? (2026-11-19, density specialist chain.)

Reads <run>/remover_log.jsonl and, per region, reports the deterministic rho at the start of
the phi phase, `window` epochs before its end and at its end, the change over that last
window, and the reward's own density slope (d adv / d z, per phi step) over the last window
with its t value. Writes the final deterministic rho as the fixed_rho table for the specialist.

Criterion (stated before the run): a region has CONVERGED if over the last `window` phi
epochs the reward no longer pushes it in one direction -- |t| of its density slope < `t_max`
(default 2) -- OR it sits at a bound (scene floor scene_rho_min, or snapped to 0). The rho
range is reported but not used: under Adam a region with no signal random-walks by ~lr per
phi step whatever the noise (toy, 2026-10-26), so a range test flags pure drift. The run
converged if every region did.

    python scripts/sim/density_convergence.py <run_dir> [--window 100] [--t-max 2] [--out <dir>]
    -> <out>/convergence.json, <out>/convergence.md; prints the hydra fixed_rho override last.
"""

import argparse
import json
import os

import numpy as np
import yaml


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--window", type=int, default=100)
    ap.add_argument("--t-max", type=float, default=2.0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out = a.out or os.path.join(a.run, "analysis")
    os.makedirs(out, exist_ok=True)
    cfg = yaml.safe_load(open(os.path.join(a.run, ".hydra", "config.yaml")))
    rl = cfg["policy"]["link_remover"]
    floor = float(rl.get("scene_rho_min", 0.1))
    ep, phi = [], []
    with open(os.path.join(a.run, "remover_log.jsonl")) as fh:
        for line in fh:
            r = json.loads(line)
            (ep if r["kind"] == "epoch" else phi).append(r)
    phi_ep = [r for r in ep if r["phase"] == "phi"]
    if not phi_ep:
        raise SystemExit("no phi-phase epochs logged")
    first, last = phi_ep[0]["epoch"], phi_ep[-1]["epoch"]
    w0 = last - a.window
    at = {r["epoch"]: r for r in ep}
    if w0 < first:
        raise SystemExit(f"phi phase {first}..{last} shorter than the window {a.window}")
    regions = list(phi_ep[-1]["phi"].keys())
    rows, final = [], {}
    for nm in regions:
        r_start = at[first]["phi"][nm]["det_rho"]
        r_w0 = at[w0]["phi"][nm]["det_rho"]
        r_end = at[last]["phi"][nm]["det_rho"]
        traj = np.array([at[e]["phi"][nm]["det_rho"] for e in range(w0, last + 1)])
        moved = float(traj.max() - traj.min())
        slopes = np.array([p[f"remover_effect_density/{nm}"] for p in phi
                           if p["epoch_frac"] >= w0 and p.get(f"remover_effect_density/{nm}") is not None
                           and np.isfinite(p[f"remover_effect_density/{nm}"])])
        t = float(slopes.mean() / (slopes.std(ddof=1) / np.sqrt(len(slopes)))) if len(slopes) > 2 else float("nan")
        at_bound = (nm == "scene" and r_end <= floor + 1e-4) or r_end == 0.0
        conv = bool(at_bound or (np.isfinite(t) and abs(t) < a.t_max))
        rows.append(dict(region=nm, rho_phi_start=r_start, rho_window_start=r_w0, rho_end=r_end,
                         range_in_window=moved, net_change_in_window=r_end - r_w0, slope_t_window=t,
                         at_bound=bool(at_bound), converged=bool(conv)))
        final[nm] = round(float(r_end), 4)
    res = dict(run=a.run, phi_epochs=[first, last], window=[w0, last], t_max=a.t_max, regions=rows,
               converged=all(r["converged"] for r in rows), fixed_rho=final)
    json.dump(res, open(os.path.join(out, "convergence.json"), "w"), indent=1)
    L = [f"# Density convergence: {os.path.basename(a.run.rstrip('/'))}", "",
         f"phi phase epochs {first}-{last}; window = last {a.window} phi epochs ({w0}-{last}); "
         f"converged = |density slope t| < {a.t_max} in the window (no consistent push left) or at a bound "
         f"(scene floor {floor}, snapped 0). Range = max - min of rho in the window (drift included).", "",
         "| region | rho phi start | rho window start | rho end | range in window | net change | slope t (window) | converged |",
         "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        L.append(f"| {r['region']} | {r['rho_phi_start']:.3f} | {r['rho_window_start']:.3f} | {r['rho_end']:.3f} | "
                 f"{r['range_in_window']:.3f} | {r['net_change_in_window']:+.3f} | {r['slope_t_window']:+.1f} | "
                 f"{'yes' + (' (bound)' if r['at_bound'] else '') if r['converged'] else '**no**'} |")
    L += ["", f"**Run converged: {'YES' if res['converged'] else 'NO'}**", "",
          "Slope t > 0 = the reward still prefers sparser, < 0 = denser (|t| < 2: no consistent push left).", ""]
    open(os.path.join(out, "convergence.md"), "w").write("\n".join(L))
    print("\n".join(L))
    print("FIXED_RHO={" + ",".join(f"{k}:{v}" for k, v in final.items()) + "}")


if __name__ == "__main__":
    main()
