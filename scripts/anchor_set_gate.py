#!/usr/bin/env python3
"""Pre-training gate for the fingertip ANCHOR-SET variants (2026-09-24).

Per anchor type (real_preprocess.ANCHOR_TYPES), on clouds fused EXACTLY as
training fuses them, and with each type's own metric radius:

  * occupancy -- share of patches with >= 1 point (0 = the learned empty token)
  * median in-radius points of the non-empty patches
  * cube proxy -- share of patches holding >= 3 red points (same red heuristic
    as fingertip_occupancy.py; only meaningful because the cube is red)

in three phase windows: approach [0, 0.4), grasp/handover [0.45, 0.65],
after [0.75, 1]. Plus, per pinch pair thumb-X (index, middle, ring), the
thumb-X pad distance: the share of handover-window frames where it closes
below --close (default 5 cm). That decides whether thumb-ring earns a token.

    micromamba activate r3d && unset PYTHONPATH
    cd ~/R3D-Policy
    python scripts/anchor_set_gate.py              # mixed150, stride 10
"""

import argparse
import os
import sys

import numpy as np
import yaml
import zarr

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "R3D"))
from r3d.common.real_preprocess import (  # noqa: E402
    ANCHOR_TYPES, HAND_FINGERS, HandFK, PointCloudPreprocessConfig, anchor_layout,
    fuse_cameras, normalize_anchor_spec)

# run F's list + the thumb-ring pinch under test + the reference tip
SPEC = [{"type": "tip", "radius_m": 0.03},
        {"type": "pad_back", "radius_m": 0.015},
        {"type": "pad_mid", "radius_m": 0.015},
        {"type": "pad_tip", "radius_m": 0.015},
        {"type": "axial", "radius_m": 0.015},
        {"type": "pad_front", "radius_m": 0.015},
        {"type": "pinch", "fingers": ["index", "middle", "ring"], "radius_m": 0.03}]
WINDOWS = {"approach": (0.0, 0.4), "handover": (0.45, 0.65), "after": (0.75, 1.0001)}


def red_mask(rgb):
    r, g, b = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    return (r > 0.35) & (r > 1.6 * g) & (r > 1.6 * b)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zarr", default="R3D/data/real_bimanual_mixed150.zarr")
    ap.add_argument("--preprocess", default="R3D/r3d/config/preprocess/real_bimanual.yaml")
    ap.add_argument("--fk", default="R3D/r3d/config/preprocess/orcahand_v1b_fk.yaml")
    ap.add_argument("--pad-offset", type=float, default=0.009)
    ap.add_argument("--stride", type=int, default=10)
    ap.add_argument("--max-episodes", type=int, default=None)
    ap.add_argument("--close", type=float, default=0.05, help="pinch 'closed' threshold [m]")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="R3D/data/outputs/anchor_set_gate")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    cfg = PointCloudPreprocessConfig.from_yaml(args.preprocess)
    raw_cfg = yaml.safe_load(open(args.preprocess))
    fk = HandFK(args.fk)
    root = zarr.open(args.zarr, mode="r")
    names = list(root["meta"].attrs["camera_names"])
    if names != [c["name"] for c in raw_cfg["cameras"]]:
        raise ValueError("zarr cameras != preprocess yaml cameras")
    n_cam = len(names)
    side_cam = {s: names.index(f"cam_wrist_{s}") for s in HandFK.SIDES}
    inv_he = {s: np.linalg.inv(np.asarray(raw_cfg["cameras"][i]["extrinsics"], dtype=np.float64))
              for s, i in side_cam.items()}
    E_static = {i: np.asarray(root[f"meta/extrinsics_cam{i}"][:], dtype=np.float64)
                for i in range(n_cam) if f"meta/extrinsics_cam{i}" in root}
    pcs = [root[f"data/point_cloud_cam{i}"] for i in range(n_cam)]
    ext = {i: root[f"data/extrinsics_cam{i}"] for i in range(n_cam) if i not in E_static}
    state = root["data/state"]

    entries = normalize_anchor_spec(SPEC)
    lay = anchor_layout(entries)
    types = np.array([t for _, _, t, _ in lay])
    radius = np.array([r for *_, r in lay])
    hands = np.array([h for h, *_ in lay])
    fingers = np.array([f for _, f, _, _ in lay])
    pinch = ANCHOR_TYPES.index("pinch")

    ends = np.asarray(root["meta/episode_ends"][:])
    starts = np.concatenate([[0], ends[:-1]])
    n_ep = len(ends) if args.max_episodes is None else min(args.max_episodes, len(ends))
    rng = np.random.default_rng(args.seed)
    phase, count, red, gap = [], [], [], []
    for ep in range(n_ep):
        s, e = int(starts[ep]), int(ends[ep])
        for t in range(s, e, args.stride):
            clouds = [np.asarray(pcs[i][t]) for i in range(n_cam)]
            Ts = [E_static[i] if i in E_static else np.asarray(ext[i][t], dtype=np.float64)
                  for i in range(n_cam)]
            fused = fuse_cameras(clouds, Ts, cfg, method="random", rng=rng)
            link8 = {sd: Ts[side_cam[sd]] @ inv_he[sd] for sd in HandFK.SIDES}
            q = np.asarray(state[t])
            anchors = fk.anchors(link8, q, spec=SPEC, pad_offset_m=args.pad_offset)
            d = np.sqrt(((fused[None, :, :3] - anchors[:, None, :]) ** 2).sum(-1))
            within = d <= radius[:, None]
            phase.append((t - s) / max(e - s - 1, 1))
            count.append(within.sum(1))
            red.append((within & red_mask(fused[:, 3:6])[None]).sum(1))
            # thumb-X pad distance per hand, X = index, middle, ring
            g = []
            for sd in HandFK.SIDES:
                from r3d.common.real_preprocess import HAND_STATE_SLICES
                _, pad, _, _ = fk.distal_link8(sd, q[HAND_STATE_SLICES[sd]], args.pad_offset)
                g += [np.linalg.norm(pad[0] - pad[i]) for i in (1, 2, 3)]
            gap.append(g)
        print(f"episode {ep + 1}/{n_ep}", flush=True)

    phase, count, red, gap = map(np.asarray, (phase, count, red, gap))
    np.savez_compressed(os.path.join(args.out, "gate.npz"), phase=phase, count=count, red=red,
                        gap=gap, types=types, radius=radius, hands=hands, fingers=fingers)

    lines = [f"anchor-set gate  {args.zarr}  {len(phase)} frames, {n_ep} episodes, "
             f"stride {args.stride}, pad offset {args.pad_offset * 1000:.0f} mm", ""]
    hdr = f"{'type':22s} {'r cm':>5s}" + "".join(
        f" | {w:>8s} occ%  med  red%" for w in WINDOWS)
    lines += [hdr, "-" * len(hdr)]
    groups = []
    for ti, name in enumerate(ANCHOR_TYPES):
        sel = types == ti
        if not sel.any():
            continue
        if ti == pinch:
            for f in sorted(set(fingers[sel])):
                groups.append((f"pinch thumb-{HAND_FINGERS[f]}", sel & (fingers == f)))
        else:
            groups.append((name, sel))
    for label, sel in groups:
        row = f"{label:22s} {radius[sel][0] * 100:5.1f}"
        for lo, hi in WINDOWS.values():
            m = (phase >= lo) & (phase < hi)
            c, r = count[m][:, sel], red[m][:, sel]
            occ = 100 * (c > 0).mean()
            med = np.median(c[c > 0]) if (c > 0).any() else 0
            row += f" | {occ:13.0f} {med:4.0f} {100 * (r >= 3).mean():5.0f}"
        lines.append(row)
    lines += ["", f"thumb-X pad distance, handover window: share of frames below "
              f"{args.close * 100:.0f} cm (and median cm)"]
    m = (phase >= WINDOWS["handover"][0]) & (phase < WINDOWS["handover"][1])
    for hi_, sd in enumerate(HandFK.SIDES):
        for k, f in enumerate(("index", "middle", "ring")):
            gg = gap[m][:, 3 * hi_ + k]
            lines.append(f"  {sd:5s} thumb-{f:6s}  {100 * (gg < args.close).mean():5.1f}%   "
                         f"median {100 * np.median(gg):4.1f} cm   min {100 * gg.min():4.1f} cm")
    report = "\n".join(lines)
    print("\n" + report)
    open(os.path.join(args.out, "report.txt"), "w").write(report + "\n")


if __name__ == "__main__":
    main()
