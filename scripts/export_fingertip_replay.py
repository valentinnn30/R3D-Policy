#!/usr/bin/env python3
"""Export one episode for the RViz fingertip replay (compute half).

Everything the policy would see, frame by frame, computed with the SAME
functions the training dataset uses (RealMultiCamDataset._sample_to_data /
_fingertip_anchors), so what RViz shows is what the model gets:

  cloud        fuse_cameras(stored per-camera clouds, NOMINAL extrinsics,
               fuse_method 'random', num_points from the preprocess yaml)
               -> xyz (world, m), rgb, source camera id
  anchors      10 FK fingertips from the MEASURED hand angles (state[14:48]),
               flange = per-frame wrist extrinsic @ inv(hand-eye)
  anchors_cmd  the same from the COMMANDED hand angles (action[14:48]), the
               "ghosts" -- same flange, only the hand angles differ
  patch_idx    per fingertip, the indices of the <= 32 points its token groups:
               nearest by METRIC distance, never beyond radius_m, empty if the
               anchor is outside the workspace box -- the encoder's rule
               (Uni3DPointcloudEncoder._fingertip_patches) at eval time
  link8        world <- {left,right}_link8, for the posed hand models
  q_meas/q_cmd the 17 hand angles per hand (rad), for the hand models

The publisher (`experiments/scripts/fingertip_replay_node.py`) runs in ros_env,
which has neither zarr nor pytorch3d -- hence this split, as for inference.

    micromamba activate r3d && unset PYTHONPATH
    cd ~/R3D-Policy
    python scripts/export_fingertip_replay.py --episode 0
"""
import argparse
import os
import sys

import numpy as np
import yaml
import zarr

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "R3D"))
from r3d.common.real_preprocess import (  # noqa: E402
    HAND_FINGERS, HAND_STATE_SLICES, HandFK, PointCloudPreprocessConfig, fuse_cameras)

K_GROUP = 32  # group_size of the encoder


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zarr", default="R3D/data/real_bimanual_mixed150.zarr")
    ap.add_argument("--episode", type=int, default=0)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--preprocess", default="R3D/r3d/config/preprocess/real_bimanual.yaml")
    ap.add_argument("--fk", default="R3D/r3d/config/preprocess/orcahand_v1b_fk.yaml")
    ap.add_argument("--radius", type=float, default=0.03)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = PointCloudPreprocessConfig.from_yaml(args.preprocess)
    raw = yaml.safe_load(open(args.preprocess))
    fk = HandFK(args.fk)
    root = zarr.open(args.zarr, mode="r")
    names = list(root["meta"].attrs["camera_names"])
    if names != [c["name"] for c in raw["cameras"]]:
        raise ValueError("zarr camera_names != preprocess yaml cameras")
    n_cam = len(names)
    statics = {i: np.asarray(root[f"meta/extrinsics_cam{i}"][:], dtype=np.float64)
               for i in range(n_cam) if f"meta/extrinsics_cam{i}" in root}
    inv_he = {s: np.linalg.inv(np.asarray(raw["cameras"][names.index(f"cam_wrist_{s}")]
                                          ["extrinsics"], dtype=np.float64))
              for s in HandFK.SIDES}
    side_cam = {s: names.index(f"cam_wrist_{s}") for s in HandFK.SIDES}
    lo, hi = cfg.bounds

    ends = np.asarray(root["meta/episode_ends"][:])
    starts = np.concatenate([[0], ends[:-1]])
    s0, e0 = int(starts[args.episode]), int(ends[args.episode])
    frames = np.arange(s0, e0, args.stride)
    print(f"episode {args.episode}: frames {s0}..{e0 - 1} ({len(frames)} exported)")

    T, N = len(frames), cfg.num_points
    xyz = np.zeros((T, N, 3), np.float32)
    rgb = np.zeros((T, N, 3), np.uint8)
    cam = np.zeros((T, N), np.uint8)
    anchors = np.zeros((T, 10, 3), np.float32)
    anchors_cmd = np.zeros((T, 10, 3), np.float32)
    patch_idx = np.full((T, 10, K_GROUP), -1, np.int32)
    n_valid = np.zeros((T, 10), np.int16)
    empty = np.zeros((T, 10), bool)
    link8 = np.zeros((T, 2, 4, 4), np.float64)
    q_meas = np.zeros((T, 2, 17), np.float32)
    q_cmd = np.zeros((T, 2, 17), np.float32)

    for k, t in enumerate(frames):
        clouds = [np.asarray(root[f"data/point_cloud_cam{i}"][t]) for i in range(n_cam)]
        Ts = [statics[i] if i in statics else
              np.asarray(root[f"data/extrinsics_cam{i}"][t], dtype=np.float64)
              for i in range(n_cam)]
        fused = fuse_cameras(clouds, Ts, cfg, method="random",
                             rng=np.random.default_rng(args.seed + int(t)),
                             camera_ids=list(range(n_cam)), n_cameras=n_cam)
        xyz[k] = fused[:, :3]
        rgb[k] = np.clip(np.round(fused[:, 3:6] * 255), 0, 255).astype(np.uint8)
        cam[k] = np.argmax(fused[:, 6:], axis=1)

        L8 = {s: Ts[side_cam[s]] @ inv_he[s] for s in HandFK.SIDES}
        state = np.asarray(root["data/state"][t], dtype=np.float64)
        action = np.asarray(root["data/action"][t], dtype=np.float64)
        cmd = state.copy()
        cmd[14:48] = action[14:48]
        anchors[k] = fk.anchors(L8, state)
        anchors_cmd[k] = fk.anchors(L8, cmd)
        for j, s in enumerate(HandFK.SIDES):
            link8[k, j] = L8[s]
            q_meas[k, j] = state[HAND_STATE_SLICES[s]]
            q_cmd[k, j] = action[HAND_STATE_SLICES[s]]

        # the encoder's grouping rule, in metres
        d = np.linalg.norm(fused[None, :, :3] - anchors[k][:, None, :], axis=-1)  # (10, N)
        inside = np.all((anchors[k] >= lo) & (anchors[k] <= hi), axis=1)
        order = np.argsort(d, axis=1)[:, :K_GROUP]
        for a in range(10):
            sel = order[a][d[a, order[a]] <= args.radius]
            n_valid[k, a] = len(sel)
            empty[k, a] = len(sel) == 0 or not inside[a]
            if not empty[k, a]:
                patch_idx[k, a, :len(sel)] = sel
        if k % 50 == 0:
            print(f"  frame {k}/{T}", flush=True)

    out = args.out or f"R3D/data/outputs/fingertip_replay/ep{args.episode:03d}.npz"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    np.savez_compressed(
        out, xyz=xyz, rgb=rgb, cam=cam, anchors=anchors, anchors_cmd=anchors_cmd,
        patch_idx=patch_idx, n_valid=n_valid, empty=empty, link8=link8,
        q_meas=q_meas, q_cmd=q_cmd, frames=frames,
        phase=(frames - s0) / max(e0 - s0 - 1, 1),
        episode=args.episode, control_hz=float(root["meta"].attrs.get("control_hz", 20.0)),
        radius=args.radius, camera_names=np.array(names),
        sides=np.array(HandFK.SIDES), fingers=np.array(HAND_FINGERS),
        fk_config=os.path.abspath(args.fk), zarr=os.path.abspath(args.zarr))
    print(f"wrote {out}  ({os.path.getsize(out) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
