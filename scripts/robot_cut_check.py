#!/usr/bin/env python3
"""Safety + coverage checks for the v2 robot cut (r3d.common.robot_cut).

  1 FINGERTIPS   on real frames: points within 3 cm of any FK fingertip that
                 the cut removes -- must be 0.
  2 GRASP SWEEP  a grid of the grasp volume in front of each palm (hand_root:
                 wrist plane .. +16 cm along the fingers, +-7 cm across, +1 ..
                 +10 cm along the palm normal), rotated about the wrist axis
                 over the wrist range seen in the data, against rules 1 + 2 of
                 its OWN arm on real arm poses -- reports the fraction cut (the
                 keep zone is NOT applied here, so this is the margin the
                 object has on its own).
  3 SELF-MODEL   points accumulated in each hand_root frame over the dataset:
                 1 cm voxels occupied in >= --stable of frames are rigidly
                 attached to the flange (mount, tower, cables; the fingers move,
                 objects come and go). Reports how many of those behind the
                 wrist plane the cut does NOT remove -- the parts of the printed
                 mount that rule 2 misses, with their extent.

    python scripts/robot_cut_check.py [--stride 20] [cut params, e.g. --forearm-radius 0.18]
"""
import argparse
import os
import sys

import numpy as np
import zarr

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "R3D"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from r3d.common.real_preprocess import (HAND_STATE_SLICES, HandFK,  # noqa: E402
                                        PointCloudPreprocessConfig, _axis_angle_matrix)
from r3d.common.robot_cut import (CUT_LABELS, SIDES, RobotCut, add_cli_args,  # noqa: E402
                                  default_paths, params_from_args)
from robot_cut_tokens import fused_frame  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zarr", default="R3D/data/real_bimanual_mixed150.zarr")
    ap.add_argument("--stride", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--stable", type=float, default=0.3,
                    help="self-model: occupancy fraction that counts as rigidly attached")
    add_cli_args(ap)
    args = ap.parse_args()
    paths = default_paths()
    params = params_from_args(args)
    rc = RobotCut(**paths, **params)
    fk = HandFK(paths["fk_yaml"])
    cfg = PointCloudPreprocessConfig.from_yaml(paths["preprocess_yaml"])
    root = zarr.open(args.zarr, "r")
    J = np.load(args.zarr.replace(".zarr", "_joints14.npz"))["joints14"]
    state = np.asarray(root["data/state"][:], dtype=np.float64)
    n_cam = len(root["meta"].attrs["camera_names"])
    statics = {i: np.asarray(root[f"meta/extrinsics_cam{i}"][:], dtype=np.float64)
               for i in range(n_cam) if f"meta/extrinsics_cam{i}" in root}
    frames = np.arange(0, len(state), args.stride)
    print(f"cut {params}\n{len(frames)} frames (stride {args.stride})")

    # ------------------------------------------------ 1 + 3 on real frames
    near_tip = near_tip_cut = 0
    vox = 0.01
    occ = {s: {} for s in SIDES}          # voxel -> [count, cut count]
    for t in frames:
        f = fused_frame(root, statics, n_cam, cfg, t, args.seed)
        cam = f[:, 6:].argmax(1)
        lab = rc.labels(f[:, :3], J[t], state[t], cam)
        cut = np.isin(lab, CUT_LABELS)
        L = rc.link_poses(J[t])
        H = rc.hand_root(None, L)
        tips = fk.anchors({s: L[s][8] for s in SIDES}, state[t])
        d = np.linalg.norm(f[None, :, :3] - tips[:, None], axis=-1).min(0)
        near_tip += int(np.sum(d < 0.03))
        near_tip_cut += int(np.sum(cut & (d < 0.03)))
        for s in SIDES:
            p = (f[:, :3] - H[s][:3, 3]) @ H[s][:3, :3]
            sel = np.linalg.norm(p - [rc.axis_xy[s][0], rc.axis_xy[s][1], rc.wrist_z[s]],
                                 axis=1) < 0.25
            for key, c in zip(map(tuple, np.floor(p[sel] / vox).astype(int)), cut[sel]):
                v = occ[s].setdefault(key, [0, 0])
                v[0] += 1
                v[1] += int(c)
    print(f"\n[1] fingertips: {near_tip_cut} of {near_tip} points within 3 cm of a "
          f"fingertip are cut  -> {'PASS' if near_tip_cut == 0 else 'FAIL'}")

    # ------------------------------------------------ 2 grasp sweep
    print("\n[2] grasp volume vs own arm (rules 1+2, keep zone OFF):")
    g = np.stack(np.meshgrid(np.arange(-0.07, 0.0701, 0.01), np.arange(0.01, 0.1001, 0.01),
                             np.arange(0.0, 0.1601, 0.01), indexing="ij"), -1).reshape(-1, 3)
    pose_frames = frames[:: max(1, len(frames) // 50)]
    for j, s in enumerate(SIDES):
        wr = state[:, HAND_STATE_SLICES[s].start]
        lo, hi = np.percentile(wr, [0.5, 99.5])
        w = np.array([rc.axis_xy[s][0], 0.0, rc.wrist_z[s]])
        worst = 0.0
        for ang in np.linspace(lo, hi, 7):
            R = _axis_angle_matrix([1.0, 0.0, 0.0], ang)
            ph = (g + [rc.axis_xy[s][0], rc.axis_xy[s][1], rc.wrist_z[s]] - w) @ R.T + w
            ph = ph[ph[:, 2] > rc.wrist_z[s] - 0.005]          # in front of the wrist only
            for t in pose_frames:
                L = rc.link_poses(J[t])
                H = rc.hand_root(None, L)[s]
                pw = ph @ H[:3, :3].T + H[:3, 3]
                own = rc._in_mount(ph, s)
                for c in range(len(rc.cap_r)):
                    T = L[s][rc.cap_link[c]]
                    a = T[:3, :3] @ rc.cap_p0[c] + T[:3, 3]
                    b = T[:3, :3] @ rc.cap_p1[c] + T[:3, 3]
                    from r3d.common.robot_cut import _seg_dist
                    own |= _seg_dist(pw, a, b) <= rc.cap_r[c]
                worst = max(worst, own.mean())
        print(f"    {s}: wrist {np.degrees(lo):.0f}..{np.degrees(hi):.0f} deg, worst "
              f"{100 * worst:.2f}% of the grasp volume cut")

    # ------------------------------------------------ 3 self-model
    print(f"\n[3] self-model: 1 cm voxels within 25 cm of the wrist occupied in "
          f">= {100 * args.stable:.0f}% of {len(frames)} frames (rigid with the flange):")
    for s in SIDES:
        keys = np.array([k for k, v in occ[s].items() if v[0] >= args.stable * len(frames)])
        if not len(keys):
            print(f"    {s}: none")
            continue
        rate = np.array([occ[s][tuple(k)][1] / occ[s][tuple(k)][0] for k in keys])
        ctr = (keys + 0.5) * vox
        behind = ctr[:, 2] < rc.wrist_z[s]
        miss = behind & (rate < 0.5)
        print(f"    {s}: {len(keys)} stable voxels, {int(behind.sum())} behind the wrist plane, "
              f"{int(miss.sum())} of those NOT cut")
        if miss.any():
            m = ctr[miss]
            r = np.linalg.norm(m[:, :2] - rc.axis_xy[s], axis=1)
            y = m[:, 1] - rc.axis_xy[s][1]
            print(f"      missed: hand z {m[:, 2].min() * 100:.1f}..{m[:, 2].max() * 100:.1f} cm "
                  f"(wrist {rc.wrist_z[s] * 100:.1f}), radius from forearm axis "
                  f"{r.min() * 100:.1f}..{r.max() * 100:.1f} cm")
            zcut = rc.wrist_z[s] - rc.p["wrist_offset"]
            for name, sel in [("palm side, y > palm_side (thumb base)", y > rc.p["palm_side"]),
                              ("within 1 cm of the wrist cut plane",
                               (y <= rc.p["palm_side"]) & (m[:, 2] > zcut - 0.01)),
                              ("elsewhere = mount NOT covered",
                               (y <= rc.p["palm_side"]) & (m[:, 2] <= zcut - 0.01))]:
                print(f"        {name}: {int(sel.sum())}")
        front = ~behind
        print(f"      in front of the wrist (hand / held object): {int(front.sum())} stable "
              f"voxels, {int(np.sum(front & (rate >= 0.5)))} of them cut")


if __name__ == "__main__":
    main()
