#!/usr/bin/env python3
"""Export one episode's FUSED clouds -- exactly what the policy sees -- to .npz.

Runs in the `r3d` env because it needs zarr and RealMultiCamDataset. The result
is plain numpy so `crop_preview_node.py --npz` can replay it into RViz from
`ros_env`, which has neither.

This is the observation AFTER per-camera crop + floor cut + FPS + fusion +
subsample to num_points -- i.e. the end of the pipeline, not an intermediate.

    micromamba run -n r3d python scripts/export_fused_cloud.py \
        --zarr R3D/data/real_bimanual_floorcut80.zarr --episode 0 --out /tmp/fused_ep0.npz
"""
import argparse, sys, numpy as np, zarr
sys.path.insert(0, "R3D")
from r3d.dataset.real_multicam_dataset import RealMultiCamDataset


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--zarr", required=True)
    p.add_argument("--config", default="R3D/r3d/config/preprocess/real_bimanual.yaml")
    p.add_argument("--episode", type=int, default=0)
    p.add_argument("--stride", type=int, default=1, help="keep every Nth frame")
    p.add_argument("--out", required=True)
    a = p.parse_args()

    ds = RealMultiCamDataset(zarr_path=a.zarr, preprocess_config=a.config,
                             horizon=16, pad_before=1, pad_after=7, seed=42,
                             val_ratio=0.0, use_extrinsics_randomization=False,
                             n_obs_steps=2, use_data_augmentation=False)
    z = zarr.open(a.zarr, "r")
    ends = np.array(z["meta/episode_ends"])
    starts = np.concatenate([[0], ends[:-1]])
    s, e = starts[a.episode], ends[a.episode]
    first = {}
    for j, (bs, be, ss, se) in enumerate(ds.sampler.indices):
        if bs not in first:
            first[bs] = j

    xyz, rgb = [], []
    for f in range(s, e, a.stride):
        if f not in first:
            continue
        pc = ds[first[f]]["obs"]["point_cloud"][0].numpy()
        xyz.append(pc[:, :3].astype(np.float32))
        rgb.append(np.clip(pc[:, 3:], 0, 1).astype(np.float32))
    xyz, rgb = np.stack(xyz), np.stack(rgb)
    np.savez_compressed(a.out, xyz=xyz, rgb=rgb)
    print(f"episode {a.episode}: {len(xyz)} frames x {xyz.shape[1]} points -> {a.out}")
    print(f"  world extent  x[{xyz[...,0].min():+.3f},{xyz[...,0].max():+.3f}] "
          f"y[{xyz[...,1].min():+.3f},{xyz[...,1].max():+.3f}] "
          f"z[{xyz[...,2].min():+.3f},{xyz[...,2].max():+.3f}]")


if __name__ == "__main__":
    main()
