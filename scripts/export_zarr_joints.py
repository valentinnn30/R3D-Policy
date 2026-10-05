#!/usr/bin/env python3
"""Recover the 14 arm joint angles per zarr frame (sidecar for the robot cut).

The zarr stores the arms as EE poses (`qpos_arm`) and the wrist extrinsics, not
joint angles, and the 7-DOF arm is redundant, so links 1..7 cannot be rebuilt
from link8. The synced h5s still carry observations/joint_states/position.

Each zarr episode is matched to its h5 by CONTENT -- zarr state[14:48] equals
the h5's qpos_hand exactly (R3D CLAUDE.md) -- never by file order: mixed150 is
YELLOW 0-29, BLUE 0-29, RED 60-89, YBR 0-59, which no sort reproduces.

Verified per frame: base @ panda_fk_flange(q) @ T_link8<-cam reproduces the
zarr's extrinsics_cam{1,2} (the wrist placement the clouds were fused with).

    python scripts/export_zarr_joints.py                     # mixed150
    -> R3D/data/real_bimanual_mixed150_joints14.npz  (joints14 (T, 14) float64)
"""
import argparse
import glob
import os
import sys

import h5py
import numpy as np
import yaml
import zarr

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "R3D"))
from r3d.common.real_preprocess import panda_fk_flange  # noqa: E402
from r3d.common.robot_cut import SIDES, wrist_bases  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zarr", default="R3D/data/real_bimanual_mixed150.zarr")
    ap.add_argument("--h5-glob", default=os.path.expanduser(
        "~/ros2_ws/tmp/*_processed_20hz_lzf/*.h5"))
    ap.add_argument("--preprocess", default="R3D/r3d/config/preprocess/real_bimanual.yaml")
    ap.add_argument("--out", default=None)
    ap.add_argument("--tol-mm", type=float, default=0.01)
    args = ap.parse_args()
    out = args.out or args.zarr.replace(".zarr", "_joints14.npz")

    root = zarr.open(args.zarr, "r")
    ends = np.asarray(root["meta/episode_ends"][:])
    starts = np.r_[0, ends[:-1]]
    hands = np.asarray(root["data/state"][:, 14:48])
    names = list(root["meta"].attrs["camera_names"])

    by_len = {}
    for fn in sorted(glob.glob(args.h5_glob)):
        with h5py.File(fn, "r") as f:
            n = f["observations/qpos_hand/left"].shape[0]
        by_len.setdefault(n, []).append(fn)

    joints = np.full((len(hands), 14), np.nan)
    sources = []
    for e, (s, t) in enumerate(zip(starts, ends)):
        hits = []
        for fn in by_len.get(int(t - s), []):
            with h5py.File(fn, "r") as f:
                qh = np.concatenate([f["observations/qpos_hand/left"][:],
                                     f["observations/qpos_hand/right"][:]], 1)
                if np.array_equal(qh.astype(np.float32), hands[s:t]):
                    hits.append((fn, f["observations/joint_states/position"][:]))
        if len(hits) != 1:
            raise RuntimeError(f"episode {e}: {len(hits)} matching h5 files")
        sources.append(os.path.relpath(hits[0][0], os.path.expanduser("~")))
        joints[s:t] = hits[0][1]

    # verify against the stored wrist extrinsics
    raw = yaml.safe_load(open(args.preprocess))
    bases = wrist_bases(args.preprocess)
    worst = {}
    for j, side in enumerate(SIDES):
        ci = names.index(f"cam_wrist_{side}")
        he = np.asarray(raw["cameras"][ci]["extrinsics"], dtype=np.float64)
        ext = np.asarray(root[f"data/extrinsics_cam{ci}"][:])
        err = np.array([np.abs((bases[side] @ panda_fk_flange(q[7 * j:7 * j + 7]) @ he)[:3, 3]
                               - ext[t][:3, 3]).max() for t, q in enumerate(joints)])
        worst[side] = float(err.max() * 1000)
    print(f"FK vs stored wrist extrinsics, worst translation error: "
          f"left {worst['left']:.4f} mm, right {worst['right']:.4f} mm")
    if max(worst.values()) > args.tol_mm:
        raise RuntimeError("joint angles do not reproduce the stored extrinsics")

    np.savez(out, joints14=joints, episode_sources=np.array(sources),
             zarr=os.path.abspath(args.zarr))
    print(f"wrote {out}: {len(joints)} frames, {len(sources)} episodes")


if __name__ == "__main__":
    main()
