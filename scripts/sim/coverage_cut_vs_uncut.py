#!/usr/bin/env python
"""Token coverage per robot part / object, uncut vs arm cut, on the stored training clouds (r3d_sim, CPU).

    python scripts/sim/coverage_cut_vs_uncut.py --stride 20

The encoder's grouping as in training (`token_allocation`: FPS with a random start to 64 centres on the
box-normalized xyz, 32-NN patches). Arm cut = every class-5 point (link0..7) deleted before the grouping.
Per part and phase of the episode: points per frame, token centres per frame, coverage (fraction of the
part's points in at least one patch) and the median patch radius in cm of the tokens centred on the part.
Datasets: Threading narrowed at 4096 points, and Square D2 at 4096 and 1024 points (first N = FPS-N) for
reference -- on Square the arm cut helped at 4096 / 64 and the analogous cut hurt at 1024 points.
"""

import argparse
import json
import os
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "R3D"))
PARTS = ["link0-4", "link5", "link6", "link7", "hand", "fingers", "object", "target", "other"]
PHASES = [(0.0, 0.3), (0.3, 0.7), (0.7, 1.0)]


def part_ids(label, link):
    p = np.full(label.shape, 8, np.int64)
    p[(link >= 0) & (link <= 4)] = 0
    for i, l in ((1, 5), (2, 6), (3, 7)):
        p[link == l] = i
    p[(link >= 8) & (link <= 13)] = 4
    p[link >= 14] = 5
    p[label == 8] = 6
    p[label == 9] = 7
    return p


def analyse(zarr_path, N, stride, num_group=64, group_size=32, seed=0):
    import torch
    import zarr
    from r3d.common.token_allocation import box_normalize, token_allocation
    torch.manual_seed(seed)
    z = zarr.open(str(zarr_path), "r")
    bbox = z["meta"].attrs["pcd_crop_bbox"]
    ends = z["meta/episode_ends"][:]
    starts = np.concatenate([[0], ends[:-1]])
    frames = np.arange(0, int(ends[-1]), stride)
    ep = np.searchsorted(ends, frames, side="right")
    phase = (frames - starts[ep]) / (ends[ep] - starts[ep])
    acc = {m: {k: [] for k in ("points", "centres", "coverage", "radius")} for m in ("uncut", "cut")}
    for f in frames:
        pc = z["data/point_cloud"][f, :N]
        lab = z["data/point_label"][f, :N]
        link = z["data/point_link"][f, :N]
        part = part_ids(lab, link)
        for m in ("uncut", "cut"):
            keep = np.ones(N, bool) if m == "uncut" else lab != 5
            xyz, pt = pc[keep, :3], part[keep]
            r = token_allocation(box_normalize(xyz, bbox)[None], pt[None], len(PARTS), num_group, group_size,
                                 random_start=True, device="cpu", return_idx=True)
            c_idx, knn = r["centre_idx"][0], r["knn_idx"][0]
            rad = np.linalg.norm(xyz[knn] - xyz[c_idx][:, None], axis=-1).max(1) * 100   # cm
            rad_part = np.full(len(PARTS), np.nan)
            for c in range(len(PARTS)):
                sel = pt[c_idx] == c
                if sel.any():
                    rad_part[c] = np.median(rad[sel])
            for k in ("points", "centres", "coverage"):
                acc[m][k].append(r[k][0])
            acc[m]["radius"].append(rad_part)
    out = {}
    for m in acc:
        out[m] = {}
        for pi, (lo, hi) in enumerate([(0.0, 1.0)] + PHASES):
            sel = (phase >= lo) & (phase < hi) if hi < 1 else (phase >= lo)
            d = {k: np.array(v)[sel] for k, v in acc[m].items()}
            out[m][f"{lo:.1f}-{hi:.1f}"] = {
                p: {"points": float(d["points"][:, c].mean()), "tokens": float(d["centres"][:, c].mean()),
                    "coverage": float(np.nanmean(d["coverage"][:, c])) if np.isfinite(d["coverage"][:, c]).any() else None,
                    "patch_cm": float(np.nanmedian(d["radius"][:, c])) if np.isfinite(d["radius"][:, c]).any() else None}
                for c, p in enumerate(PARTS)}
    return out, len(frames)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stride", type=int, default=20)
    ap.add_argument("--out", default=str(ROOT / "R3D/data/outputs/sim/mimicgen/threading/coverage_cut_vs_uncut.json"))
    a = ap.parse_args()
    import torch
    torch.set_num_threads(2)
    sets = [("threading 4096", ROOT / "R3D/data/threading_d1n-agentview_p4096-200.zarr", 4096),
            ("square 4096", ROOT / "R3D/data/square_d2-agentview_p4096-200.zarr", 4096),
            ("square 1024", ROOT / "R3D/data/square_d2-agentview_p4096-200.zarr", 1024)]
    res = {}
    for name, path, N in sets:
        res[name], n = analyse(path, N, a.stride)
        print(f"\n== {name}: {n} frames (stride {a.stride}), 64 tokens")
        print(f"{'part':9s} | {'pts':>5s} | {'tok uncut':>9s} {'tok cut':>7s} | {'cov uncut':>9s} {'cov cut':>7s} | {'patch cm u/c':>12s}")
        for p in PARTS:
            u, c = res[name]["uncut"]["0.0-1.0"][p], res[name]["cut"]["0.0-1.0"][p]
            f = lambda v, s="{:.0%}": "-" if v is None else s.format(v)
            print(f"{p:9s} | {u['points']:5.0f} | {u['tokens']:9.1f} {c['tokens']:7.1f} | {f(u['coverage']):>9s} {f(c['coverage']):>7s} | "
                  f"{f(u['patch_cm'], '{:.1f}')}/{f(c['patch_cm'], '{:.1f}')}")
        for ph in ("0.0-0.3", "0.3-0.7", "0.7-1.0"):
            line = []
            for p in ("object", "target"):
                u, c = res[name]["uncut"][ph][p], res[name]["cut"][ph][p]
                line.append(f"{p} cov {u['coverage']:.0%}->{c['coverage']:.0%}, tok {u['tokens']:.1f}->{c['tokens']:.1f}")
            print(f"  phase {ph}: " + "; ".join(line))
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1)
    print("\nwrote", a.out)


if __name__ == "__main__":
    main()
