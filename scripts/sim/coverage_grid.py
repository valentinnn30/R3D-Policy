#!/usr/bin/env python
"""Coverage grid (2026-11-14): nut / peg coverage, arm and object tokens, object patch size per (points, tokens),
uncut and arm cut [5], stored Square D2 frames (stride 40), token_allocation as in training. r3d_sim, CPU."""
import sys, numpy as np, zarr, torch
sys.path.insert(0, "/home/valentin/R3D-Policy-sim/R3D")
from r3d.common.token_allocation import box_normalize, token_allocation
torch.set_num_threads(6); torch.manual_seed(0)
z = zarr.open("/home/valentin/R3D-Policy-sim/R3D/data/square_d2-agentview_p4096-200.zarr", "r")
bbox = z["meta"].attrs["pcd_crop_bbox"]
frames = np.arange(0, z["data/point_cloud"].shape[0], 40)
cfgs = [(1024, 64), (1024, 128), (2048, 64), (2048, 128), (2048, 256), (4096, 64), (4096, 128), (4096, 256)]
acc = {(n, g, c): [] for n, g in cfgs for c in ("uncut", "cut")}
for f in frames:
    pc, lab = z["data/point_cloud"][f], z["data/point_label"][f]
    part = np.full(len(lab), 3); part[lab == 5] = 0; part[lab == 8] = 1; part[lab == 9] = 2
    for n, g in cfgs:
        for c in ("uncut", "cut"):
            keep = np.arange(len(lab)) < n
            if c == "cut": keep &= lab != 5
            xyz = pc[keep, :3]; pt = part[keep]
            r = token_allocation(box_normalize(xyz, bbox)[None], pt[None], 4, g, 32, random_start=True, device="cpu", return_idx=True)
            ci, knn = r["centre_idx"][0], r["knn_idx"][0]
            rad = np.linalg.norm(xyz[knn] - xyz[ci][:, None], axis=-1).max(1) * 100
            obj = np.isin(pt[ci], [1, 2])
            acc[(n, g, c)].append([r["coverage"][0][1], r["coverage"][0][2], r["centres"][0][0], r["centres"][0][1] + r["centres"][0][2],
                                   np.median(rad[obj]) if obj.any() else np.nan])
print(f"{len(frames)} frames; nut cov / peg cov / arm tokens / nut+peg tokens / object patch cm")
for n, g in cfgs:
    for c in ("uncut", "cut"):
        a = np.nanmean(np.array(acc[(n, g, c)], float), 0)
        print(f"{n:5d}/{g:3d} {c:5s}  {a[0]:.2f}  {a[1]:.2f}  {a[2]:5.1f}  {a[3]:5.1f}  {a[4]:.1f}")
