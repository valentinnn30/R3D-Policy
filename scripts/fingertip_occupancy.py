#!/usr/bin/env python3
"""Fingertip-patch occupancy on a fused zarr -- the pre-training gate.

For every sampled frame the cloud is fused EXACTLY as training fuses it
(`fuse_cameras`, method random, 8192 points, nominal extrinsics, now with the
per-point camera one-hot), the 10 fingertips are placed by `HandFK`, and for
each fingertip we record what a 3 cm metric patch around it would contain:

  * number of points (0 = the patch would be the learned "empty" token)
  * which cameras those points come from
  * how many are red-ish -- a crude CUBE PROXY by colour, only meaningful if
    the cube in the data is red. It is a heuristic, not a segmentation.
  * whether the anchor is inside the workspace box at all

and plots all of it against normalised episode phase, where the grasp and the
handover are visible.

It also writes .ply frames (fused cloud + FK fingertips) for the VISUAL check of
the analytic hand mount: the coloured spheres must sit on the real fingertips.
Colours: left hand tips green, right hand tips magenta; a sparse grey shell
marks the 3 cm patch radius.

The flange pose is recovered from the per-frame wrist extrinsics,
    T_world<-link8(t) = extrinsics_camK(t) @ inv(T_link8<-camK),
i.e. the SAME joint-FK route that placed the wrist clouds, so anchors and
cloud share one registration.

Reads the zarr lazily, frame by frame: a full ReplayBuffer is ~12.6 GiB.

    micromamba activate r3d && unset PYTHONPATH
    cd ~/R3D-Policy
    python scripts/fingertip_occupancy.py            # defaults: mixed150, stride 10
"""

import argparse
import os
import sys

import numpy as np
import yaml
import zarr

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "R3D"))
from r3d.common.real_preprocess import (  # noqa: E402
    HAND_FINGERS, HandFK, PointCloudPreprocessConfig, fuse_cameras)

TIP_NAMES = [f"{s[0].upper()}_{f}" for s in HandFK.SIDES for f in HAND_FINGERS]


def red_mask(rgb):
    r, g, b = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    return (r > 0.35) & (r > 1.6 * g) & (r > 1.6 * b)


def sphere(center, radius, n):
    v = np.random.default_rng(0).normal(size=(n, 3))
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    return center + radius * v


def write_ply(path, xyz, rgb01):
    rgb = np.clip(np.round(rgb01 * 255), 0, 255).astype(np.uint8)
    with open(path, "w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(xyz)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for p, c in zip(xyz, rgb):
            f.write(f"{p[0]:.5f} {p[1]:.5f} {p[2]:.5f} {c[0]} {c[1]} {c[2]}\n")


FINGER_COLORS = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00"]


def mount_check(root, raw_cfg, fk, names, episodes, phases, max_dist, out):
    """Per-FRAME mount check: one PNG per (episode, hand), one column per phase.

    Each panel is ONE recorded frame: that hand's OWN wrist camera cloud (the
    stored per-camera cloud, i.e. what training fuses), moved into the flange
    frame link8 by the calibrated hand-eye transform, and the 5 FK fingertips
    of the SAME frame (analytic mount + that frame's recorded hand angles).

    Background removal uses no kinematics: only points closer than `max_dist`
    to the camera are kept (the hand is the nearest thing a wrist camera sees;
    fingertips measured at 0.17-0.31 m). The view is then zoomed to the
    fingertips. If the mount is right, each coloured marker sits on the end of
    a finger in the cloud.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ends = np.asarray(root["meta/episode_ends"][:])
    starts = np.concatenate([[0], ends[:-1]])
    written = []
    for side in HandFK.SIDES:
        ci = names.index(f"cam_wrist_{side}")
        he = np.asarray(raw_cfg["cameras"][ci]["extrinsics"], dtype=np.float64)
        sl = slice(14, 31) if side == "left" else slice(31, 48)
        for ep in episodes:
            s, e = int(starts[ep]), int(ends[ep])
            fig, axes = plt.subplots(2, len(phases), figsize=(4.2 * len(phases), 8.4),
                                     squeeze=False)
            for col, ph in enumerate(phases):
                t = s + int(round(ph * (e - s - 1)))
                pc = np.asarray(root[f"data/point_cloud_cam{ci}"][t])
                pc = pc[np.linalg.norm(pc[:, :3], axis=1) < max_dist]
                p8 = pc[:, :3] @ he[:3, :3].T + he[:3, 3]
                tips = fk.tips_link8(side, np.asarray(root["data/state"][t])[sl])
                lo_, hi_ = tips.min(0) - 0.06, tips.max(0) + 0.06
                keep = np.all((p8 > lo_) & (p8 < hi_), axis=1)
                for row, (a, b, lab) in enumerate(((0, 2, "side view  x-z"),
                                                   (0, 1, "top view  x-y"))):
                    ax = axes[row, col]
                    ax.scatter(p8[keep, a], p8[keep, b], s=2,
                               c=np.clip(pc[keep, 3:6], 0, 1))
                    for f, name in enumerate(HAND_FINGERS):
                        ax.scatter(tips[f, a], tips[f, b], s=90, marker="o",
                                   facecolors="none", edgecolors=FINGER_COLORS[f],
                                   linewidths=2.2, label=name if (row, col) == (0, 0) else None)
                    ax.set_xlim(lo_[a], hi_[a])
                    ax.set_ylim(lo_[b], hi_[b])
                    ax.set_aspect("equal")
                    ax.grid(alpha=0.3)
                    ax.set_title(f"phase {ph:.2f} (frame {t})\n{lab} [m, link8]", fontsize=9)
            axes[0, 0].legend(loc="upper left", fontsize=8, title="FK fingertip")
            fig.suptitle(f"{side} hand, episode {ep}: own wrist camera, points < "
                         f"{max_dist:.2f} m from the camera; circles = FK fingertips "
                         "of the same frame", fontsize=11)
            fig.tight_layout()
            path = os.path.join(out, f"mount_check_{side}_ep{ep:03d}.png")
            fig.savefig(path, dpi=100)
            plt.close(fig)
            written.append(path)
    return written


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zarr", default="R3D/data/real_bimanual_mixed150.zarr")
    ap.add_argument("--preprocess", default="R3D/r3d/config/preprocess/real_bimanual.yaml")
    ap.add_argument("--fk", default="R3D/r3d/config/preprocess/orcahand_v1b_fk.yaml")
    ap.add_argument("--radius", type=float, default=0.03)
    ap.add_argument("--stride", type=int, default=10, help="every Nth frame")
    ap.add_argument("--max-episodes", type=int, default=None)
    ap.add_argument("--ply-episodes", type=int, default=3)
    ap.add_argument("--ply-phases", default="0.15,0.35,0.5,0.65")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="R3D/data/outputs/fingertip_occupancy")
    ap.add_argument("--mount-only", action="store_true",
                    help="only write the per-frame mount check figures")
    ap.add_argument("--mount-episodes", default="0,74,149")
    ap.add_argument("--mount-max-dist", type=float, default=0.36,
                    help="background cut: keep points closer than this to the camera [m]")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    cfg = PointCloudPreprocessConfig.from_yaml(args.preprocess)
    raw_cfg = yaml.safe_load(open(args.preprocess))
    fk = HandFK(args.fk)
    root = zarr.open(args.zarr, mode="r")

    names = list(root["meta"].attrs["camera_names"])
    yaml_names = [c["name"] for c in raw_cfg["cameras"]]
    if names != yaml_names:
        raise ValueError(f"zarr cameras {names} != preprocess yaml {yaml_names}")
    side_cam = {"left": names.index("cam_wrist_left"),
                "right": names.index("cam_wrist_right")}
    hand_eye = {s: np.asarray(raw_cfg["cameras"][i]["extrinsics"], dtype=np.float64)
                for s, i in side_cam.items()}
    n_cam = len(names)

    ply_phases_ = [float(p) for p in args.ply_phases.split(",")]
    for path in mount_check(root, raw_cfg, fk, names,
                            [int(x) for x in args.mount_episodes.split(",")],
                            ply_phases_, args.mount_max_dist, args.out):
        print(f"wrote {path}")
    if args.mount_only:
        return

    ends = np.asarray(root["meta/episode_ends"][:])
    starts = np.concatenate([[0], ends[:-1]])
    n_ep = len(ends) if args.max_episodes is None else min(args.max_episodes, len(ends))
    E_static = {}
    for i in range(n_cam):
        if f"meta/extrinsics_cam{i}" in root:
            E_static[i] = np.asarray(root[f"meta/extrinsics_cam{i}"][:], dtype=np.float64)
    pcs = [root[f"data/point_cloud_cam{i}"] for i in range(n_cam)]
    ext = {i: root[f"data/extrinsics_cam{i}"] for i in range(n_cam) if i not in E_static}
    state = root["data/state"]
    lo, hi = cfg.bounds

    ply_phases = [float(p) for p in args.ply_phases.split(",")]
    ply_eps = set(np.linspace(0, n_ep - 1, args.ply_episodes).round().astype(int).tolist())

    rng = np.random.default_rng(args.seed)
    rec = {k: [] for k in ("episode", "phase", "count", "red", "cam_counts", "inside", "anchor")}
    r2 = args.radius ** 2
    for ep in range(n_ep):
        s, e = int(starts[ep]), int(ends[ep])
        frames = list(range(s, e, args.stride))
        ply_frames = {s + int(round(p * (e - s - 1))): p for p in ply_phases} \
            if ep in ply_eps else {}
        frames = sorted(set(frames) | set(ply_frames))
        for t in frames:
            clouds = [np.asarray(pcs[i][t]) for i in range(n_cam)]
            Ts = [E_static[i] if i in E_static else np.asarray(ext[i][t], dtype=np.float64)
                  for i in range(n_cam)]
            fused = fuse_cameras(clouds, Ts, cfg, method="random", rng=rng,
                                 camera_ids=list(range(n_cam)), n_cameras=n_cam)
            link8 = {sd: Ts[side_cam[sd]] @ np.linalg.inv(hand_eye[sd]) for sd in HandFK.SIDES}
            anchors = fk.anchors(link8, np.asarray(state[t]))

            d2 = ((fused[None, :, :3] - anchors[:, None, :]) ** 2).sum(-1)  # (10, N)
            within = d2 <= r2
            red = red_mask(fused[:, 3:6])
            rec["episode"].append(ep)
            rec["phase"].append((t - s) / max(e - s - 1, 1))
            rec["count"].append(within.sum(1))
            rec["red"].append((within & red[None]).sum(1))
            rec["cam_counts"].append(within.astype(np.int32) @ fused[:, 6:6 + n_cam].astype(np.int32))
            rec["inside"].append(np.all((anchors > lo) & (anchors < hi), axis=1))
            rec["anchor"].append(anchors)

            if t in ply_frames:
                pts, cols = [fused[:, :3]], [fused[:, 3:6]]
                for k, a in enumerate(anchors):
                    c = np.array([0.1, 0.9, 0.1]) if k < 5 else np.array([0.9, 0.1, 0.9])
                    ball = sphere(a, 0.004, 120)
                    shell = sphere(a, args.radius, 150)
                    pts += [ball, shell]
                    cols += [np.tile(c, (len(ball), 1)), np.tile([0.6, 0.6, 0.6], (len(shell), 1))]
                path = os.path.join(args.out, f"ep{ep:03d}_phase{ply_frames[t]:.2f}_t{t}.ply")
                write_ply(path, np.concatenate(pts), np.concatenate(cols))
        print(f"episode {ep + 1}/{n_ep} done ({len(frames)} frames)", flush=True)

    R = {k: np.asarray(v) for k, v in rec.items()}
    np.savez_compressed(os.path.join(args.out, "occupancy.npz"), tip_names=TIP_NAMES,
                        camera_names=names, radius=args.radius, **R)

    # ---------------------------------------------------------------- summary
    lines = [f"zarr {args.zarr}  episodes {n_ep}  frames {len(R['phase'])}  "
             f"radius {args.radius * 100:.1f} cm  stride {args.stride}", ""]
    lines.append(f"{'tip':10s} {'median':>6s} {'p10':>5s} {'p90':>5s} {'empty%':>7s} "
                 f"{'outbox%':>8s} {'red>0%':>7s}  camera share of in-radius points "
                 f"({', '.join(names)})")
    for k, nm in enumerate(TIP_NAMES):
        c = R["count"][:, k]
        cc = R["cam_counts"][:, k].sum(0)
        share = cc / max(cc.sum(), 1)
        lines.append(
            f"{nm:10s} {np.median(c):6.0f} {np.percentile(c, 10):5.0f} {np.percentile(c, 90):5.0f} "
            f"{100 * np.mean(c == 0):6.1f}% {100 * np.mean(~R['inside'][:, k]):7.1f}% "
            f"{100 * np.mean(R['red'][:, k] > 0):6.1f}%  "
            + " ".join(f"{x:.2f}" for x in share))
    bins = np.linspace(0, 1, 11)
    lines += ["", "median points in radius per phase decile (all 10 tips pooled):"]
    b = np.clip(np.digitize(R["phase"], bins) - 1, 0, 9)
    for i in range(10):
        m = b == i
        lines.append(f"  phase {bins[i]:.1f}-{bins[i + 1]:.1f}: count {np.median(R['count'][m]):5.0f}  "
                     f"empty {100 * np.mean(R['count'][m] == 0):5.1f}%  "
                     f"red>0 {100 * np.mean(R['red'][m] > 0):5.1f}%")
    txt = "\n".join(lines)
    print(txt)
    open(os.path.join(args.out, "summary.txt"), "w").write(txt + "\n")

    # ------------------------------------------------------------------ plots
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    centers = (bins[:-1] + bins[1:]) / 2
    fig, axes = plt.subplots(2, 3, figsize=(16, 8), sharex=True)
    for row, side in enumerate(("left", "right")):
        for col, (key, title) in enumerate((("count", "points within radius"),
                                            ("red", "fraction of patches with red-ish points (cube proxy)"),
                                            ("empty", "empty patch fraction"))):
            ax = axes[row, col]
            for f in range(5):
                k = row * 5 + f
                if key == "empty":
                    y = [np.mean(R["count"][b == i, k] == 0) for i in range(10)]
                elif key == "red":
                    y = [np.mean(R["red"][b == i, k] > 0) for i in range(10)]
                else:
                    y = [np.median(R[key][b == i, k]) for i in range(10)]
                ax.plot(centers, y, marker="o", label=HAND_FINGERS[f])
            ax.set_title(f"{side} hand: {title}")
            ax.grid(alpha=0.3)
            if row == 1:
                ax.set_xlabel("episode phase")
    axes[0, 0].legend()
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "occupancy_vs_phase.png"), dpi=120)

    print(f"wrote {args.out}/summary.txt, occupancy_vs_phase.png, occupancy.npz, *.ply")


if __name__ == "__main__":
    main()
