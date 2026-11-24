#!/usr/bin/env python
"""Claim sweep, mechanism table BEFORE training (r3d_sim, CPU), and the armsparse gate.

    python scripts/sim/claim_sweep_mechanism.py [--stride 20]

Stored Square D2 training frames, the encoder's grouping as in training (`token_allocation`: FPS with a random
start to 64 centres on the box-normalized xyz, 32-NN patches; as coverage_cut_vs_uncut.py). Clouds: uncut 4096,
armsparse, scenesparse (r3d/common/prefix_thin.py, prefix 1024), and the 1024 cloud (first 1024 points). Per
part (arm = link0..7, gripper, nut, peg): points kept, tokens, coverage (fraction of the part's KEPT points in
at least one patch), median patch radius in cm of the tokens centred on the part.

GATE (plan section 4): armsparse arm tokens < 28 or nut coverage > 0.60 -> exit 2 (training must not start).
Writes R3D/data/outputs/sim/mimicgen/claim_sweep/mechanism.{json,md}.
"""

import argparse
import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "R3D"))
ZARR = ROOT / "R3D/data/square_d2-agentview_p4096-200.zarr"
OUT = ROOT / "R3D/data/outputs/sim/mimicgen/claim_sweep"
PARTS = ["arm", "gripper", "nut", "peg", "other"]
CLOUDS = ["uncut 4096", "armsparse", "scenesparse", "1024"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stride", type=int, default=20)
    a = ap.parse_args()
    import torch
    import zarr
    from r3d.common.prefix_thin import arm_cell_ids, thin_drop_labels
    from r3d.common.token_allocation import box_normalize, token_allocation
    torch.set_num_threads(4)
    torch.manual_seed(0)
    z = zarr.open(str(ZARR), "r")
    bbox = z["meta"].attrs["pcd_crop_bbox"]
    arm = arm_cell_ids()
    frames = np.arange(0, z["data/point_cloud"].shape[0], a.stride)
    acc = {c: {k: [] for k in ("points", "centres", "coverage", "radius", "kept")} for c in CLOUDS}
    for f in frames:
        pc, lab, link = z["data/point_cloud"][f], z["data/point_label"][f], z["data/point_link"][f]
        part = np.full(len(lab), 4)
        part[np.isin(link, arm)] = 0
        part[(link >= 8)] = 1                       # hand + fingers cells = the gripper
        part[lab == 8] = 2
        part[lab == 9] = 3
        keeps = {"uncut 4096": np.ones(len(lab), bool),
                 "armsparse": thin_drop_labels(link, "arm", 1024, arm) == 0,
                 "scenesparse": thin_drop_labels(link, "non_arm", 1024, arm) == 0,
                 "1024": np.arange(len(lab)) < 1024}
        for c, keep in keeps.items():
            xyz, pt = pc[keep, :3], part[keep]
            r = token_allocation(box_normalize(xyz, bbox)[None], pt[None], len(PARTS), 64, 32,
                                 random_start=True, device="cpu", return_idx=True)
            ci, knn = r["centre_idx"][0], r["knn_idx"][0]
            rad = np.linalg.norm(xyz[knn] - xyz[ci][:, None], axis=-1).max(1) * 100
            rp = np.array([np.median(rad[pt[ci] == p]) if (pt[ci] == p).any() else np.nan for p in range(len(PARTS))])
            for k in ("points", "centres", "coverage"):
                acc[c][k].append(r[k][0])
            acc[c]["radius"].append(rp)
            acc[c]["kept"].append(int(keep.sum()))
    res = {}
    for c in CLOUDS:
        d = {k: np.array(v) for k, v in acc[c].items()}
        res[c] = {"points_kept": float(d["kept"].mean())}
        for i, p in enumerate(PARTS[:4]):
            res[c][p] = {"points": float(d["points"][:, i].mean()), "tokens": float(d["centres"][:, i].mean()),
                         "coverage": float(np.nanmean(d["coverage"][:, i])),
                         "patch_cm": float(np.nanmedian(d["radius"][:, i]))}
    lines = [f"Mechanism table: {len(frames)} stored Square D2 frames (stride {a.stride}), 64 tokens, k = 32", "",
             "| cloud | points kept | arm pts / tok / cov / patch cm | gripper tok | nut pts / tok / cov / patch cm | peg pts / tok / cov / patch cm |",
             "|---|---|---|---|---|---|"]
    for c in CLOUDS:
        r = res[c]
        f = lambda p: f"{r[p]['points']:.0f} / {r[p]['tokens']:.1f} / {r[p]['coverage']:.2f} / {r[p]['patch_cm']:.1f}"
        lines.append(f"| {c} | {r['points_kept']:.0f} | {f('arm')} | {r['gripper']['tokens']:.1f} | {f('nut')} | {f('peg')} |")
    arm_tok, nut_cov = res["armsparse"]["arm"]["tokens"], res["armsparse"]["nut"]["coverage"]
    gate_ok = arm_tok >= 28 and nut_cov <= 0.60
    lines += ["", f"GATE armsparse: arm tokens {arm_tok:.1f} (>= 28), nut coverage {nut_cov:.2f} (<= 0.60) -> "
              + ("PASS" if gate_ok else "FAIL: armsparse moves tokens; do not train, report to the user")]
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "mechanism.json").write_text(json.dumps({"stride": a.stride, "frames": len(frames), "clouds": res,
                                                    "gate_pass": gate_ok}, indent=1))
    (OUT / "mechanism.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    sys.exit(0 if gate_ok else 2)


if __name__ == "__main__":
    main()
