#!/usr/bin/env python3
"""Precompute the robot-cut labels for every STORED point of a zarr (R1, step 1).

Stage 2 (fuse_cameras' uniform draw) is random per sample, so the cut cannot be
stored after it. Instead every stored stage-1 point gets its label here; the
dataset carries the labels through fuse_cameras as an extra column and drops
the cut rows after the draw. Every rule of r3d.common.robot_cut decides each
point on its own (arm/hand pose of the frame + the point's camera), so this is
exactly "K's stage-2 input minus the robot" -- verified by --verify.

Output: a sidecar zarr next to the dataset (the dataset is never modified),

    <zarr>_robotcut_<tag>.zarr
        labels_cam{i}  (T, n_i) int8   KEEP/ARM/MOUNT/SHADOW/SAVED per stored point
        attrs: params (json), source zarr, geometry npz mtime, camera_names

    micromamba activate r3d && unset PYTHONPATH && cd ~/R3D-Policy
    python scripts/precompute_robot_cut.py --workers 8          # default cut params
    python scripts/precompute_robot_cut.py --verify 300         # check only
"""
import argparse
import json
import os
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")       # one thread per worker
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np  # noqa: E402
import zarr  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "R3D"))
from r3d.common.robot_cut import (RobotCut, add_cli_args, default_paths,  # noqa: E402
                                  params_from_args, params_tag)

CHUNK = 256


def sidecar_path(zarr_path, params):
    return zarr_path.replace(".zarr", f"_robotcut_{params_tag(params)}.zarr")


def _frame_clouds(root, statics, n_cam, t):
    out = []
    for i in range(n_cam):
        c = np.asarray(root[f"data/point_cloud_cam{i}"][t])
        T = statics[i] if i in statics else np.asarray(root[f"data/extrinsics_cam{i}"][t],
                                                       dtype=np.float64)
        out.append(c[:, :3].astype(np.float64) @ T[:3, :3].T + T[:3, 3])
    return out


def frame_labels(rc, root, statics, n_cam, J, state, t):
    """Per camera, labels of the stored points of frame t (world frame)."""
    xyz = _frame_clouds(root, statics, n_cam, t)
    cam = np.concatenate([np.full(len(x), i) for i, x in enumerate(xyz)])
    lab = rc.labels(np.concatenate(xyz), J[t], state[t], cam)
    return np.split(lab, np.cumsum([len(x) for x in xyz])[:-1])


def _worker(job):
    zarr_path, out_path, joints_path, params, lo, hi = job
    rc = RobotCut(**default_paths(), **params)
    root = zarr.open(zarr_path, "r")
    out = zarr.open(out_path, "r+")
    n_cam = len(root["meta"].attrs["camera_names"])
    statics = {i: np.asarray(root[f"meta/extrinsics_cam{i}"][:], dtype=np.float64)
               for i in range(n_cam) if f"meta/extrinsics_cam{i}" in root}
    J = np.load(joints_path)["joints14"]
    state = np.asarray(root["data/state"][lo:hi], dtype=np.float64)
    bufs = [np.zeros((hi - lo, root[f"data/point_cloud_cam{i}"].shape[1]), np.int8)
            for i in range(n_cam)]
    for k, t in enumerate(range(lo, hi)):
        labs = frame_labels(rc, root, statics, n_cam, J, {t: state[k]}, t)
        for i in range(n_cam):
            bufs[i][k] = labs[i]
    for i in range(n_cam):
        out[f"labels_cam{i}"][lo:hi] = bufs[i]
    return hi - lo


def verify(args, params, n):
    """The stored labels, carried through fuse_cameras and its random draw,
    must equal the labels computed directly on the drawn cloud."""
    from r3d.common.real_preprocess import PointCloudPreprocessConfig, fuse_cameras
    from r3d.common.robot_cut import CUT_LABELS
    paths = default_paths()
    rc = RobotCut(**paths, **params)
    cfg = PointCloudPreprocessConfig.from_yaml(paths["preprocess_yaml"])
    root = zarr.open(args.zarr, "r")
    side = zarr.open(sidecar_path(args.zarr, params), "r")
    n_cam = len(root["meta"].attrs["camera_names"])
    statics = {i: np.asarray(root[f"meta/extrinsics_cam{i}"][:], dtype=np.float64)
               for i in range(n_cam) if f"meta/extrinsics_cam{i}" in root}
    J = np.load(args.zarr.replace(".zarr", "_joints14.npz"))["joints14"]
    state = root["data/state"]
    rng = np.random.default_rng(1)
    frames = rng.choice(len(J), size=n, replace=False)
    bad = 0
    for t in frames:
        clouds = [np.asarray(root[f"data/point_cloud_cam{i}"][t]) for i in range(n_cam)]
        labels = [np.asarray(side[f"labels_cam{i}"][t]) for i in range(n_cam)]
        Ts = [statics[i] if i in statics else np.asarray(root[f"data/extrinsics_cam{i}"][t])
              for i in range(n_cam)]
        # pad_to_min_points draws from the GLOBAL rng -- seed it for both calls
        np.random.seed(int(t))
        f = fuse_cameras(clouds, Ts, cfg, method="random", rng=np.random.default_rng(int(t)),
                         camera_ids=list(range(n_cam)), n_cameras=n_cam, point_labels=labels)
        np.random.seed(int(t))
        ref = fuse_cameras(clouds, Ts, cfg, method="random", rng=np.random.default_rng(int(t)),
                           camera_ids=list(range(n_cam)), n_cameras=n_cam)
        if not np.array_equal(f[:, :-1], ref):
            raise RuntimeError(f"frame {t}: carrying labels changed the drawn cloud")
        # column layout: xyz rgb one-hot label
        carried = f[:, -1].astype(np.int8)
        direct = rc.labels(f[:, :3], J[t], np.asarray(state[t], np.float64),
                           f[:, 6:-1].argmax(1))
        if not np.array_equal(np.isin(carried, CUT_LABELS), np.isin(direct, CUT_LABELS)):
            bad += 1
            print(f"  frame {t}: {int(np.sum(carried != direct))} labels differ")
    print(f"verify: {n - bad}/{n} frames identical cut mask")
    return bad == 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zarr", default="R3D/data/real_bimanual_mixed150.zarr")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--verify", type=int, default=0,
                    help="only verify N random frames against an existing sidecar")
    add_cli_args(ap)
    args = ap.parse_args()
    params = params_from_args(args)
    if args.verify:
        sys.exit(0 if verify(args, params, args.verify) else 1)

    root = zarr.open(args.zarr, "r")
    names = list(root["meta"].attrs["camera_names"])
    T = root["data/state"].shape[0]
    joints = args.zarr.replace(".zarr", "_joints14.npz")
    if np.load(joints)["joints14"].shape[0] != T:
        raise RuntimeError(f"{joints} does not match {args.zarr}")
    out_path = sidecar_path(args.zarr, params)
    out = zarr.open(out_path, "w")
    for i in range(len(names)):
        n_i = root[f"data/point_cloud_cam{i}"].shape[1]
        out.zeros(f"labels_cam{i}", shape=(T, n_i), chunks=(CHUNK, n_i), dtype=np.int8)
    out.attrs.update({"params": json.dumps(params), "zarr": os.path.abspath(args.zarr),
                      "camera_names": names,
                      "geometry_mtime": os.path.getmtime(default_paths()["geometry_npz"])})
    jobs = [(args.zarr, out_path, joints, params, lo, min(lo + CHUNK, T))
            for lo in range(0, T, CHUNK)]
    print(f"{T} frames, {len(jobs)} chunks, {args.workers} workers -> {out_path}")
    t0, done = time.time(), 0
    if args.workers <= 1:
        results = map(_worker, jobs)
    else:
        import multiprocessing as mp
        pool = mp.get_context("spawn").Pool(args.workers)
        results = pool.imap_unordered(_worker, jobs)
    for n in results:
        done += n
        if done % (CHUNK * 20) < CHUNK:
            el = time.time() - t0
            print(f"  {done}/{T}  {el:.0f} s, eta {el / done * (T - done):.0f} s", flush=True)
    out.attrs["complete"] = True
    print(f"done in {time.time() - t0:.0f} s -> {out_path}")


if __name__ == "__main__":
    main()
