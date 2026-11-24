#!/usr/bin/env python
"""How the encoder's tokens sit on the nut / peg (user, 2026-11-16): gaps, overlap, size.  r3d_sim, CPU.

    python scripts/sim/object_token_geometry.py [--stride 40]

Stored Square D2 training frames, the encoder's grouping as in training (`token_allocation`: FPS with a random start,
kNN patches). Per configuration and object (nut = label 8, peg = 9):
  coverage      fraction of the object's points in at least one patch (1 - coverage = gaps)
  multiplicity  mean number of patches an object point is in (over ALL its points; 0 for uncovered) -- overlap
  >=2           fraction of the object's points in 2 or more patches
  tokens        token centres on the object
  share/token   mean fraction of the object's points inside one token centred on it (how much of the object one
                token sees); max = the largest such fraction in the frame
  radius        median patch radius (cm) of the tokens centred on the object
Writes R3D/data/outputs/sim/mimicgen/token_geometry/object_token_geometry.{json,md} and a picture of one frame.
"""
import argparse, json, pathlib, sys
import numpy as np
ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "R3D"))
OUT = ROOT / "R3D/data/outputs/sim/mimicgen/token_geometry"
TASKS = {  # zarr, configs (name, points, tokens, k, success @250 s42), objects (label 8 = object, 9 = target)
    "square": ("R3D/data/square_d2-agentview_p4096-200.zarr",
               [("4096/64 k32", 4096, 64, 32, 8.5), ("4096/256 k32", 4096, 256, 32, 14.0),
                ("4096/64 k128", 4096, 64, 128, 19.0), ("1024/64 k32", 1024, 64, 32, 23.0)],
               {"nut": 8, "peg": 9}),
    "threading": ("R3D/data/threading_d1n-agentview_p4096-200.zarr",
                  [("4096/64 k32", 4096, 64, 32, 37.5), ("1024/64 k32", 1024, 64, 32, 35.5)],
                  {"needle": 8, "tripod": 9}),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stride", type=int, default=40)
    ap.add_argument("--frame", type=int, default=2000, help="frame for the picture")
    ap.add_argument("--task", default="square", choices=sorted(TASKS))
    a = ap.parse_args()
    global CFGS, OBJ, OUT
    zpath, CFGS, OBJ = TASKS[a.task]
    if a.task != "square":
        OUT = OUT / a.task
    import torch, zarr
    from r3d.common.token_allocation import box_normalize, token_allocation
    torch.set_num_threads(6); torch.manual_seed(0)
    z = zarr.open(str(ROOT / zpath), "r")
    bbox = z["meta"].attrs["pcd_crop_bbox"]
    frames = np.arange(0, z["data/point_cloud"].shape[0], a.stride)
    keys = ("coverage", "multiplicity", "ge2", "tokens", "share", "share_max", "radius")
    acc = {c[0]: {o: {k: [] for k in keys} for o in OBJ} for c in CFGS}
    pic = {}
    for f in list(frames) + [a.frame]:
        pc, lab = z["data/point_cloud"][f], z["data/point_label"][f]
        for name, N, G, K, _ in CFGS:
            xyz, lb = pc[:N, :3], lab[:N]
            r = token_allocation(box_normalize(xyz, bbox)[None], lb[None].astype(np.int64), 11, G, K,
                                 random_start=True, device="cpu", return_idx=True)
            ci, knn = r["centre_idx"][0], r["knn_idx"][0]
            mult = np.bincount(knn.reshape(-1), minlength=N)          # patches each point is in
            rad = np.linalg.norm(xyz[knn] - xyz[ci][:, None], axis=-1).max(1) * 100
            if f == a.frame:
                pic[name] = (xyz, lb, ci, knn, mult)
                continue
            for o, l in OBJ.items():
                m = lb == l
                n = int(m.sum())
                if n == 0:
                    continue
                own = lb[ci] == l
                share = (lb[knn[own]] == l).sum(1) / n if own.any() else np.array([np.nan])
                d = acc[name][o]
                d["coverage"].append((mult[m] > 0).mean()); d["multiplicity"].append(mult[m].mean())
                d["ge2"].append((mult[m] >= 2).mean()); d["tokens"].append(own.sum())
                d["share"].append(np.nanmean(share)); d["share_max"].append(np.nanmax(share))
                d["radius"].append(np.median(rad[own]) if own.any() else np.nan)
    res = {name: {o: {k: float(np.nanmean(v)) for k, v in acc[name][o].items()} for o in OBJ} for name in acc}
    L = [f"Object token geometry, {a.task}, {len(frames)} stored (stride {a.stride}); slots = tokens x k", "",
         "| config | slots / points | success | object | coverage | multiplicity | >= 2 patches | tokens | share of object per token (max) | patch radius |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for name, N, G, K, s in CFGS:
        for o in OBJ:
            r = res[name][o]
            L.append(f"| {name} | {G * K / N:.1f}x | {s} | {o} | {r['coverage']:.2f} | {r['multiplicity']:.2f} | "
                     f"{r['ge2']:.2f} | {r['tokens']:.1f} | {r['share']:.2f} ({r['share_max']:.2f}) | {r['radius']:.1f} cm |")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "object_token_geometry.json").write_text(json.dumps(res, indent=1))
    (OUT / "object_token_geometry.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))
    # picture: top view of the nut + peg region, points coloured by multiplicity, token centres, one patch outlined
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, len(CFGS), figsize=(5.5 * len(CFGS), 5.5))
    fig.suptitle("nut + peg points coloured by #patches covering them (0 = gap); red = token centres on the objects, orange = one patch; grey = other points", fontsize=11)
    for ax, (name, *_rest) in zip(axs, CFGS):
        xyz, lb, ci, knn, mult = pic[name]
        obj = np.isin(lb, list(OBJ.values()))
        lo, hi = xyz[obj, :2].min(0) - 0.03, xyz[obj, :2].max(0) + 0.03
        box = np.all((xyz[:, :2] >= lo) & (xyz[:, :2] <= hi), 1)
        sc = ax.scatter(xyz[box, 0], xyz[box, 1], c=np.minimum(mult[box], 6), s=5, cmap="viridis", vmin=0, vmax=6)
        ax.scatter(xyz[box & ~obj, 0], xyz[box & ~obj, 1], s=1, c="lightgrey")
        oc = ci[obj[ci]]
        ax.scatter(xyz[oc, 0], xyz[oc, 1], s=60, facecolors="none", edgecolors="red", linewidths=1.2)
        if len(oc):
            j = int(np.flatnonzero(obj[ci])[0])
            ax.scatter(xyz[knn[j], 0], xyz[knn[j], 1], s=14, facecolors="none", edgecolors="orange")
        ax.set_title(name, fontsize=12)
        ax.set_aspect("equal")
    fig.colorbar(sc, ax=axs, shrink=0.7, label="patches covering the point (0 = gap)")
    fig.savefig(OUT / "object_token_geometry.png", dpi=110, bbox_inches="tight")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
