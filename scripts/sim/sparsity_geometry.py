#!/usr/bin/env python
"""Sparsity limit (user, 2026-11-16): token coverage and patch size on the nut / peg for 4096 -> 256 points,
64 tokens, k = 32, uncut. r3d_sim, CPU.

    python scripts/sim/sparsity_geometry.py [--stride 40]

Stored Square D2 training frames; the first N points of the stored FPS-4096 cloud (= what the policy sees);
the encoder's grouping as in training (`token_allocation`: FPS with a random start, kNN). Per config and object
(nut = label 8, peg = 9), over frames in which the object has >= 1 of the N points:
  pts         object points among the N
  coverage    fraction of those points inside >= 1 patch
  tokens      token centres on the object (frames with 0 -> 'no token' %)
  r_own       median patch radius of tokens CENTRED on the object (cm)
  r_touch     median patch radius of all patches containing >= 1 object point (cm)
  share       mean fraction of the object's points inside one token centred on it
  diameter    the object's visible diameter = max pairwise distance of its points in the FULL 4096 cloud of the
              same frame (cm); patch diameter (2 r_own) is reported relative to it
Writes R3D/data/outputs/sim/mimicgen/sparsity/geometry.{json,md}.
"""
import argparse, json, pathlib, sys
import numpy as np
ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "R3D"))
OUT = ROOT / "R3D/data/outputs/sim/mimicgen/sparsity"
POINTS = (4096, 2048, 1024, 512, 256)
OBJ = {"nut": 8, "peg": 9}


def max_extent(x):
    if len(x) < 2:
        return np.nan
    d = np.linalg.norm(x[:, None] - x[None], axis=-1)
    return d.max()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stride", type=int, default=40)
    ap.add_argument("--tokens", type=int, default=64)
    ap.add_argument("--k", type=int, default=32)
    a = ap.parse_args()
    import torch, zarr
    from r3d.common.token_allocation import box_normalize, token_allocation
    torch.set_num_threads(6); torch.manual_seed(0)
    z = zarr.open(str(ROOT / "R3D/data/square_d2-agentview_p4096-200.zarr"), "r")
    bbox = z["meta"].attrs["pcd_crop_bbox"]
    frames = np.arange(0, z["data/point_cloud"].shape[0], a.stride)
    keys = ("pts", "coverage", "tokens", "no_token", "r_own", "r_touch", "share", "diam")
    acc = {N: {o: {k: [] for k in keys} for o in OBJ} for N in POINTS}
    for f in frames:
        pc, lab = z["data/point_cloud"][f], z["data/point_label"][f]
        diam = {o: max_extent(pc[lab == l, :3]) * 100 for o, l in OBJ.items()}
        for N in POINTS:
            xyz, lb = pc[:N, :3], lab[:N]
            r = token_allocation(box_normalize(xyz, bbox)[None], lb[None].astype(np.int64), 11, a.tokens, a.k,
                                 random_start=True, device="cpu", return_idx=True)
            ci, knn = r["centre_idx"][0], r["knn_idx"][0]
            mult = np.bincount(knn.reshape(-1), minlength=N)
            rad = np.linalg.norm(xyz[knn] - xyz[ci][:, None], axis=-1).max(1) * 100
            for o, l in OBJ.items():
                m = lb == l
                n = int(m.sum())
                d = acc[N][o]
                d["pts"].append(n)
                if n == 0:
                    continue
                own = lb[ci] == l
                touch = (lb[knn] == l).any(1)
                d["coverage"].append((mult[m] > 0).mean())
                d["tokens"].append(own.sum()); d["no_token"].append(float(own.sum() == 0))
                d["r_own"].append(np.median(rad[own]) if own.any() else np.nan)
                d["r_touch"].append(np.median(rad[touch]))
                d["share"].append(((lb[knn[own]] == l).sum(1) / n).mean() if own.any() else np.nan)
                d["diam"].append(diam[o])
    res = {N: {o: {k: float(np.nanmean(v)) for k, v in acc[N][o].items()} for o in OBJ} for N in POINTS}
    for N in POINTS:
        for o in OBJ:
            res[N][o]["frames_absent"] = float(np.mean(np.array(acc[N][o]["pts"]) == 0))
            res[N][o]["patch_over_diam"] = 2 * res[N][o]["r_own"] / res[N][o]["diam"]
            res[N][o]["touch_over_diam"] = 2 * res[N][o]["r_touch"] / res[N][o]["diam"]
    L = [f"Square D2, {len(frames)} stored frames (stride {a.stride}), {a.tokens} tokens, k = {a.k}, uncut. "
         "Patch diameter = 2 x median radius (centre to farthest of its k points).", "",
         "| points | object | object points | coverage | tokens on it (frames with none) | patch radius, centred on it | patch diameter / object diameter | patches touching it: radius (diam ratio) | share of object per token | object diameter (visible, 4096) |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for N in POINTS:
        for o in OBJ:
            r = res[N][o]
            L.append(f"| {N} | {o} | {r['pts']:.0f} | {r['coverage']:.2f} | {r['tokens']:.1f} ({100 * r['no_token']:.0f}%) | "
                     f"{r['r_own']:.1f} cm | {r['patch_over_diam']:.2f} | {r['r_touch']:.1f} cm ({r['touch_over_diam']:.2f}) | "
                     f"{r['share']:.2f} | {r['diam']:.1f} cm |")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "geometry.json").write_text(json.dumps(res, indent=1))
    (OUT / "geometry.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
