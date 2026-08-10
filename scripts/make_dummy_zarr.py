"""Generate a synthetic zarr in the R3D ReplayBuffer format.

Purely a smoke-test fixture: it lets us exercise config -> dataset ->
normalizer -> encoder -> UNet -> checkpoint end to end before any real
recording exists, so that the first run on real data fails for real reasons.
The trajectories are smooth nonsense; nothing here should ever be trained on
for a real policy.

    python scripts/make_dummy_zarr.py --output R3D/data/dummy.zarr \\
        --episodes 4 --length 60 --d-state 14 --d-action 14
"""

import argparse
import os
import shutil

import numpy as np
import zarr
from termcolor import cprint


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", default="R3D/data/dummy.zarr")
    p.add_argument("--episodes", type=int, default=4)
    p.add_argument("--length", type=int, default=60, help="frames per episode")
    p.add_argument("--num-points", type=int, default=1024)
    p.add_argument("--cameras", type=int, default=0,
                   help="0 writes a fused `point_cloud`; N>0 writes N per-camera "
                        "clouds in camera frame plus their nominal extrinsics")
    p.add_argument("--wrist-cameras", type=int, default=0,
                   help="how many of --cameras are wrist-mounted, i.e. carry a "
                        "per-frame extrinsic rather than a fixed one")
    p.add_argument("--d-state", type=int, default=14)
    p.add_argument("--d-action", type=int, default=14)
    p.add_argument("--d-ee", type=int, default=0, help=">0 also writes target_ee")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--force", action="store_true")
    args = p.parse_args()

    if os.path.exists(args.output):
        if not args.force:
            cprint(f"{args.output} exists; pass --force to overwrite", "red")
            raise SystemExit(1)
        shutil.rmtree(args.output)

    rng = np.random.default_rng(args.seed)
    n_cams = max(args.cameras, 0)
    n_wrist = min(max(args.wrist_cameras, 0), n_cams)
    # Cameras 0..n_cams-n_wrist-1 are static; the rest are wrist-mounted and get
    # a nominal extrinsic that moves every frame, as a real wrist camera does.

    def yaw_transform(yaw, offset):
        c, s = np.cos(yaw), np.sin(yaw)
        T = np.eye(4)
        T[:3, :3] = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        T[:3, 3] = offset
        return T

    # Distinct nominal extrinsics per camera, so fusion is genuinely exercised:
    # cameras looking at the table from different sides. Camera-frame points are
    # produced by inverting these.
    extrinsics = [
        yaw_transform(2 * np.pi * i / max(n_cams, 1), [0.1 * i, -0.1 * i, 0.05 * i])
        for i in range(n_cams)
    ]
    is_wrist = [i >= n_cams - n_wrist for i in range(n_cams)]

    clouds, states, actions, ees, ends, total = [], [], [], [], [], 0
    per_cam = [[] for _ in range(n_cams)]
    per_cam_extr = [[] for _ in range(n_cams)]

    for ep in range(args.episodes):
        t = np.linspace(0, 1, args.length)[:, None]

        # a slab of points roughly where a table would be, in the base frame
        xyz = rng.uniform([-0.4, -0.6, 0.0], [0.4, 0.6, 0.5],
                          size=(args.length, args.num_points, 3))
        xyz[..., 2] += 0.05 * np.sin(6 * t)                 # something time-varying
        rgb = rng.uniform(0.0, 1.0, size=(args.length, args.num_points, 3))

        if n_cams:
            for i, T in enumerate(extrinsics):
                if is_wrist[i]:
                    # T_base<-cam(t): a slow sweep, as if carried by a moving arm
                    Ts = np.stack([
                        yaw_transform(
                            np.arctan2(T[1, 0], T[0, 0]) + 0.4 * np.sin(2 * np.pi * tt),
                            T[:3, 3] + [0.05 * np.sin(2 * np.pi * tt), 0.0, 0.0])
                        for tt in t.ravel()
                    ])
                    R_inv = np.transpose(Ts[:, :3, :3], (0, 2, 1))
                    t_inv = -np.einsum("nij,nj->ni", R_inv, Ts[:, :3, 3])
                    xyz_cam = np.einsum("nij,npj->npi", R_inv, xyz) + t_inv[:, None, :]
                    per_cam_extr[i].append(Ts)
                else:
                    R_inv, t_inv = T[:3, :3].T, -T[:3, :3].T @ T[:3, 3]
                    xyz_cam = xyz @ R_inv.T + t_inv
                per_cam[i].append(np.concatenate([xyz_cam, rgb], -1).astype(np.float32))
        else:
            clouds.append(np.concatenate([xyz, rgb], -1).astype(np.float32))

        phase = rng.uniform(0, 2 * np.pi, args.d_state)
        states.append((0.5 * np.sin(2 * np.pi * t + phase)).astype(np.float32))
        phase_a = rng.uniform(0, 2 * np.pi, args.d_action)
        actions.append((0.5 * np.sin(2 * np.pi * t + phase_a)).astype(np.float32))
        if args.d_ee:
            phase_e = rng.uniform(0, 2 * np.pi, args.d_ee)
            ees.append((0.3 * np.cos(2 * np.pi * t + phase_e)).astype(np.float32))

        total += args.length
        ends.append(total)  # cumulative

    root = zarr.group(args.output)
    data, meta = root.create_group("data"), root.create_group("meta")
    comp = zarr.Blosc(cname="zstd", clevel=3, shuffle=1)

    def put(name, arr):
        data.create_dataset(name, data=arr, dtype="float32",
                            chunks=(100,) + arr.shape[1:], compressor=comp)

    if n_cams:
        for i in range(n_cams):
            put(f"point_cloud_cam{i}", np.concatenate(per_cam[i]))
            if is_wrist[i]:
                arr = np.concatenate(per_cam_extr[i])
                data.create_dataset(f"extrinsics_cam{i}", data=arr, dtype="float64",
                                    chunks=(100, 4, 4), compressor=comp)
            else:
                meta.create_dataset(f"extrinsics_cam{i}", data=extrinsics[i],
                                    dtype="float64")
        meta.attrs["camera_names"] = [f"cam{i}" for i in range(n_cams)]
    else:
        put("point_cloud", np.concatenate(clouds))
    put("state", np.concatenate(states))
    put("action", np.concatenate(actions))
    if ees:
        put("target_ee", np.concatenate(ees))
    meta.create_dataset("episode_ends", data=np.asarray(ends, np.int64), chunks=(100,))

    cprint(f"wrote {args.output}: {args.episodes} episodes, {total} frames, "
           f"{n_cams if n_cams else 'fused'} camera(s)", "green")


if __name__ == "__main__":
    main()
