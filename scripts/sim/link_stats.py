#!/usr/bin/env python
"""Step 0.1 of the learned link remover (docs/learned_remover_v0.md): per-link statistics.
Run in r3d_sim, after scripts/sim/add_link_labels_mimicgen.py.

    python scripts/sim/link_stats.py                       # Square D2 p4096 zarr, N = 1024/2048/4096

For every point budget N (the first N stored points = the FPS-N cloud the policy sees):

- points per stored cell and per remover region (default grouping, robosuite_pcd):
  mean / p50 / p90 / max over ALL frames, and the fraction of frames the region is present
- the robot share of the cloud, and the non-robot minimum (what a complete removal keeps)
  against the token budget M (N_kept >= M must hold)
- the mu_min check: at mu = mu_min the deterministic keep count round(sigmoid(mu) * n) must
  be 0 for every region in every frame; plus how often a SAMPLED draw (sigma 0.2 / 1.0)
  at mu_min still keeps >= 1 point
- token allocation at M tokens, k = 32 (`token_allocation`, random start as in training)
  on every `--token-stride`-th frame: token centres and coverage per region

Report: R3D/data/outputs/sim/link_labels/<zarr stem>_stats.json (+ printed tables).
"""

import argparse
import json
import os
import sys

import numpy as np
import zarr

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "R3D"))
from r3d.common import robosuite_pcd as RP  # noqa: E402

OBJ_NAMES = {RP.OBJECT: "nut", RP.TARGET: "peg"}


def pct(x):
    return {"mean": float(np.mean(x)), "p50": float(np.percentile(x, 50)), "p90": float(np.percentile(x, 90)),
            "max": int(np.max(x)), "present": float(np.mean(x > 0))}


def region_of_cells(groups):
    cell_to_region = np.full(RP.N_CELLS, -1, np.int64)
    for r, g in enumerate(groups):
        for c in g["cells"]:
            cell_to_region[RP.CELL_NAMES.index(c)] = r
    return cell_to_region


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--zarr", default=os.path.join(ROOT, "R3D", "data", "square_d2-agentview_p4096-200.zarr"))
    ap.add_argument("--num-points", type=int, nargs="+", default=[1024, 2048, 4096])
    ap.add_argument("--tokens", type=int, default=64)
    ap.add_argument("--group-size", type=int, default=32)
    ap.add_argument("--token-stride", type=int, default=10)
    ap.add_argument("--mu-min", type=float, default=-8.0)
    a = ap.parse_args()

    z = zarr.open(a.zarr, "r")
    if "point_link" not in z["data"]:
        raise KeyError(f"{a.zarr} has no data/point_link -- run scripts/sim/add_link_labels_mimicgen.py")
    if list(z["meta"].attrs["point_link_names"]) != list(RP.CELL_NAMES):
        raise ValueError("zarr point_link_names differ from robosuite_pcd.CELL_NAMES")
    bbox = z["meta"].attrs["pcd_crop_bbox"]
    groups = RP.default_region_groups()
    rnames = [g["name"] for g in groups]
    c2r = region_of_cells(groups)
    print("loading point_link / point_label ...", flush=True)
    cells_all = z["data/point_link"][:]
    labels_all = z["data/point_label"][:]
    T = len(cells_all)
    rng = np.random.default_rng(0)
    rep = {"zarr": a.zarr, "frames": T, "tokens": a.tokens, "group_size": a.group_size,
           "regions": groups, "per_N": {}}

    for N in a.num_points:
        cells, labels = cells_all[:, :N].astype(np.int64), labels_all[:, :N]
        cell_n = np.stack([(cells == c).sum(1) for c in range(RP.N_CELLS)], 1)            # (T, 22)
        reg_n = np.stack([cell_n[:, c2r == r].sum(1) for r in range(len(groups))], 1)     # (T, L)
        robot = (cells >= 0).sum(1)
        thinned = reg_n.sum(1)
        nonrobot_kept = N - thinned
        r = {"cells": {RP.CELL_NAMES[c]: pct(cell_n[:, c]) for c in range(RP.N_CELLS)},
             "regions": {rnames[l]: pct(reg_n[:, l]) for l in range(len(groups))},
             "robot_share": pct(robot / N), "robot_points": pct(robot),
             "objects": {nm: pct((labels == c).sum(1)) for c, nm in OBJ_NAMES.items()},
             "kept_at_full_removal_min": int(nonrobot_kept.min()),
             "full_removal_ok": bool(nonrobot_kept.min() >= a.tokens)}
        rho_min = 1.0 / (1.0 + np.exp(-a.mu_min))
        det_keep = np.floor(rho_min * reg_n + 0.5)
        r["mu_min_check"] = {"rho_at_mu_min": rho_min, "max_kept_deterministic": int(det_keep.max()),
                             "ok": bool(det_keep.max() == 0)}
        for sig in (0.2, 1.0):
            zs = a.mu_min + sig * rng.standard_normal(reg_n.shape)
            k = np.floor(reg_n / (1.0 + np.exp(-zs)) + 0.5)
            r["mu_min_check"][f"sampled_sigma{sig}_frac_regions_keeping_any"] = float((k[reg_n > 0] > 0).mean())
            r["mu_min_check"][f"sampled_sigma{sig}_max_kept"] = int(k.max())

        # token allocation (needs the GPU encoder grouping)
        import torch
        from r3d.common.token_allocation import box_normalize, token_allocation
        ts = np.arange(0, T, a.token_stride)
        pc = z["data/point_cloud"].get_orthogonal_selection((ts, slice(0, N), slice(0, 3)))
        # one label space: regions 0..L-1, then nut, peg, everything else
        L = len(groups)
        lab = np.where(cells[ts] >= 0, c2r[np.clip(cells[ts], 0, None)], -1)
        lab = np.where(lab >= 0, lab, np.where(labels[ts] == RP.OBJECT, L, np.where(labels[ts] == RP.TARGET, L + 1, L + 2)))
        names = rnames + ["nut", "peg", "other"]
        acc = {"centres": [], "coverage": [], "points": []}
        for i in range(0, len(ts), 256):
            out = token_allocation(box_normalize(pc[i:i + 256], bbox), lab[i:i + 256], L + 3,
                                   num_group=a.tokens, group_size=a.group_size, random_start=True)
            for kk in acc:
                acc[kk].append(np.asarray(out[kk]))
        cen, cov, pts = (np.concatenate(acc[k]) for k in ("centres", "coverage", "points"))
        r["tokens_per_region"] = {
            names[l]: {"centres_mean": float(cen[:, l].mean()),
                       "coverage_mean_when_present": float(cov[pts[:, l] > 0, l].mean()) if (pts[:, l] > 0).any() else None}
            for l in range(L + 3)}
        rep["per_N"][str(N)] = r

        print(f"\n=== N = {N}  (robot share mean {r['robot_share']['mean']:.2f}; complete removal keeps >= "
              f"{r['kept_at_full_removal_min']} points vs M = {a.tokens}: {'ok' if r['full_removal_ok'] else 'VIOLATED'};"
              f" mu_min check {'ok' if r['mu_min_check']['ok'] else 'FAILED'}) ===")
        print(f"{'region':16s} {'mean':>7s} {'p50':>6s} {'p90':>6s} {'max':>6s} {'present':>8s} {'tokens':>7s} {'cov':>5s}")
        for l, nm in enumerate(names):
            st = r["regions"].get(nm) or r["objects"].get(nm)
            tk = r["tokens_per_region"][nm]
            if st is None:
                continue
            cv = tk["coverage_mean_when_present"]
            print(f"{nm:16s} {st['mean']:7.1f} {st['p50']:6.0f} {st['p90']:6.0f} {st['max']:6d} {st['present']:8.3f} "
                  f"{tk['centres_mean']:7.2f} {'' if cv is None else f'{cv:5.2f}'}")

    out = os.path.join(ROOT, "R3D", "data", "outputs", "sim", "link_labels",
                       os.path.basename(a.zarr.rstrip("/")).replace(".zarr", "") + "_stats.json")
    with open(out, "w") as fh:
        json.dump(rep, fh, indent=1)
    print(f"\nreport: {out}")


if __name__ == "__main__":
    main()
