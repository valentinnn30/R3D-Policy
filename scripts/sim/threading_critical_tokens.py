#!/usr/bin/env python
"""Threading: do tokens sit on the task-critical parts? (user, 2026-11-16)  r3d_sim, CPU.

    python scripts/sim/threading_critical_tokens.py [--stride 20]

The zarr labels the needle (8) and tripod (9) only as wholes, so the critical parts are defined geometrically per frame:
  needle tip : needle points within 1.5 cm of the needle point FARTHEST from the gripper (link cells >= 8: hand +
               fingers) -- the end that goes through the ring
  ring       : tripod points in the top 3 cm of the tripod (the ring sits on top)
Grouping as in training (token_allocation: random-start FPS, 32-NN). Per configuration and part: points, coverage,
token centres ON the part, frames with >= 1 centre on it, largest share of the part inside one patch, patch radius.
Writes R3D/data/outputs/sim/mimicgen/token_geometry/threading/critical_parts.md.
"""
import argparse, pathlib, sys
import numpy as np
ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "R3D"))
OUT = ROOT / "R3D/data/outputs/sim/mimicgen/token_geometry/threading"
CFGS = [("4096/64", 4096, 64, 37.5), ("1024/64", 1024, 64, 35.5)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stride", type=int, default=20)
    a = ap.parse_args()
    import torch, zarr
    from r3d.common.token_allocation import box_normalize, token_allocation
    torch.set_num_threads(6); torch.manual_seed(0)
    z = zarr.open(str(ROOT / "R3D/data/threading_d1n-agentview_p4096-200.zarr"), "r")
    bbox = z["meta"].attrs["pcd_crop_bbox"]
    ends = z["meta/episode_ends"][:]
    starts = np.concatenate([[0], ends[:-1]])
    frames = np.arange(0, int(ends[-1]), a.stride)
    keys = ("points", "coverage", "centres", "has_centre", "share_max", "radius")
    acc = {c[0]: {p: {k: [] for k in keys} for p in ("needle tip", "ring", "needle (all)", "tripod (all)")} for c in CFGS}
    for f in frames:
        pc, lab, link = z["data/point_cloud"][f], z["data/point_label"][f], z["data/point_link"][f]
        for name, N, G, _ in CFGS:
            xyz, lb, lk = pc[:N, :3], lab[:N], link[:N]
            needle, tripod, grip = lb == 8, lb == 9, lk >= 8
            if needle.sum() < 5 or tripod.sum() < 5 or grip.sum() < 1:
                continue
            g = xyz[grip].mean(0)
            far = xyz[needle][np.argmax(np.linalg.norm(xyz[needle] - g, axis=1))]
            tip = needle & (np.linalg.norm(xyz - far, axis=1) < 0.015)
            ring = tripod & (xyz[:, 2] >= xyz[tripod, 2].max() - 0.03)
            r = token_allocation(box_normalize(xyz, bbox)[None], np.zeros((1, N), np.int64), 1, G, 32,
                                 random_start=True, device="cpu", return_idx=True)
            ci, knn = r["centre_idx"][0], r["knn_idx"][0]
            mult = np.bincount(knn.reshape(-1), minlength=N)
            rad = np.linalg.norm(xyz[knn] - xyz[ci][:, None], axis=-1).max(1) * 100
            for part, m in (("needle tip", tip), ("ring", ring), ("needle (all)", needle), ("tripod (all)", tripod)):
                n = int(m.sum())
                if n == 0:
                    continue
                on = m[ci]
                d = acc[name][part]
                d["points"].append(n); d["coverage"].append((mult[m] > 0).mean())
                d["centres"].append(on.sum()); d["has_centre"].append(on.any())
                d["share_max"].append((m[knn].sum(1) / n).max())
                d["radius"].append(np.median(rad[on]) if on.any() else np.nan)
    L = [f"Threading critical parts, {len(frames)} stored frames (stride {a.stride}), 64 tokens, k = 32", "",
         "| config (success) | part | points | coverage | token centres on it | frames with >= 1 centre | largest share in one patch | patch radius |",
         "|---|---|---|---|---|---|---|---|"]
    for name, N, G, s in CFGS:
        for part, d in acc[name].items():
            m = {k: np.nanmean(np.array(v, float)) for k, v in d.items()}
            L.append(f"| {name} ({s}) | {part} | {m['points']:.0f} | {m['coverage']:.2f} | {m['centres']:.1f} | "
                     f"{m['has_centre']:.2f} | {m['share_max']:.2f} | {m['radius']:.1f} cm |")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "critical_parts.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
