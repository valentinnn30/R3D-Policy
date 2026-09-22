#!/usr/bin/env python3
"""Which thumb joint convention puts the FK thumb tip on the real thumb?

Visual check on FULL-RESOLUTION recorded images, not a fit. For a few frames
of one episode, every candidate convention is drawn as a numbered marker on
crops of all three cameras, around one hand's thumb. The user picks the
candidate that sits on the thumb.

Candidates (applied to the measured hand angles before FK):
  0  as-is (current model)
  1  thumb_mcp sign flipped
  2  thumb_abd sign flipped
  3  thumb_pip sign flipped
  4  thumb_dip sign flipped
  5  thumb_mcp <-> thumb_abd values swapped
A correct convention must hold in EVERY frame, including the control frames
where the current model already looks right.

Sources: the synced 3D h5 of the episode (grid timestamps, measured
qpos_hand, joint_states, native intrinsics) and the RAW h5 it came from
(JPEGs at native resolution, picked per grid time by nearest header stamp --
the same rule syncronize_data_2d.py uses). Camera poses and hand FK exactly
as in fingertip_image_check.py.

    python scripts/thumb_convention_check.py \\
        --synced ~/ros2_ws/tmp/31-60_YELLOW_processed_20hz_lzf/0000.h5 \\
        --raw-dir ~/ros2_ws/tmp/31-60_YELLOW --side right --frames 199,235,289
"""
import argparse
import os
import sys

import cv2
import h5py
import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "R3D"))
from fingertip_image_check import H5_CAMS, camera_poses  # noqa: E402
from r3d.common.real_preprocess import HAND_JOINT_ORDER, HandFK  # noqa: E402

CANDIDATES = [
    ("as-is", lambda q: q),
    ("-thumb_mcp", lambda q: _flip(q, "thumb_mcp")),
    ("-thumb_abd", lambda q: _flip(q, "thumb_abd")),
    ("-thumb_pip", lambda q: _flip(q, "thumb_pip")),
    ("-thumb_dip", lambda q: _flip(q, "thumb_dip")),
    ("mcp<->abd", lambda q: _swap(q, "thumb_mcp", "thumb_abd")),
]
COLORS_BGR = [(40, 40, 230), (230, 160, 40), (60, 200, 60), (200, 60, 200),
              (0, 150, 255), (255, 255, 0)]


def _flip(q, name):
    q = q.copy()
    q[HAND_JOINT_ORDER.index(name)] *= -1
    return q


def _swap(q, a, b):
    q = q.copy()
    i, j = HAND_JOINT_ORDER.index(a), HAND_JOINT_ORDER.index(b)
    q[i], q[j] = q[j], q[i]
    return q


# verbatim from ros2_ws/.../logger/syncronize_data.py (_entry_times, _nearest_index)
def _entry_times(group):
    keys = [k for k in group.keys() if k.lstrip("-").isdigit()]
    times = np.array([int(group[k].attrs.get("stamp", k)) for k in keys], dtype=np.int64)
    order = np.argsort(times, kind="stable")
    return times[order], [keys[i] for i in order]


def _nearest_index(src_times, dst_times):
    hi = np.clip(np.searchsorted(src_times, dst_times), 1, len(src_times) - 1)
    take_lo = (dst_times - src_times[hi - 1]) <= (src_times[hi] - dst_times)
    idx = np.where(take_lo, hi - 1, hi)
    return idx, np.abs(src_times[idx] - dst_times)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--synced", required=True)
    ap.add_argument("--raw-dir", required=True)
    ap.add_argument("--side", default="right", choices=["left", "right"])
    ap.add_argument("--frames", default="199,235,289")
    ap.add_argument("--candidates", default=None,
                    help="comma list of candidate ids to draw (default: all)")
    ap.add_argument("--auto-frames", type=int, default=0,
                    help="instead of --frames: the N frames with the largest |thumb_dip| "
                         "(spread >= 20 frames apart) plus the frame with the smallest")
    ap.add_argument("--preprocess", default="R3D/r3d/config/preprocess/real_bimanual.yaml")
    ap.add_argument("--fk", default="R3D/r3d/config/preprocess/orcahand_v1b_fk.yaml")
    ap.add_argument("--crop", type=int, default=260, help="half size of the crop [px]")
    ap.add_argument("--out", default="R3D/data/outputs/fingertip_occupancy")
    args = ap.parse_args()

    raw_cfg = yaml.safe_load(open(args.preprocess))
    cams = raw_cfg["cameras"]
    names = [c["name"] for c in cams]
    fk = HandFK(args.fk)
    td = h5py.File(args.synced, "r")
    raw = h5py.File(os.path.join(args.raw_dir, td.attrs["source_file"]), "r")
    grid = np.asarray(td["timestamps_ns"])
    sl = slice(0, 17)
    K = {n: np.asarray(td[f"observations/images/{H5_CAMS[n]}/intrinsics/K"]) for n in names}
    D = {n: np.asarray(td[f"observations/images/{H5_CAMS[n]}/intrinsics/disto"]) for n in names}
    img_idx = {}
    for n in names:
        g = raw[f"{H5_CAMS[n]}/color/compressed"]
        times, keys = _entry_times(g)
        idx, gaps = _nearest_index(times, grid)
        img_idx[n] = (g, keys, idx, gaps)

    keep_c = (set(int(c) for c in args.candidates.split(",")) if args.candidates
              else set(range(len(CANDIDATES))))
    if args.auto_frames:
        dip = np.abs(np.asarray(td[f"observations/qpos_hand/{args.side}"])[
            :, HAND_JOINT_ORDER.index("thumb_dip")])
        frames = []
        for t in np.argsort(-dip):
            if all(abs(int(t) - f) >= 20 for f in frames):
                frames.append(int(t))
            if len(frames) == args.auto_frames:
                break
        frames = sorted(frames) + [int(np.argmin(dip))]
        print("frames (|thumb_dip| deg):",
              {f: round(float(np.degrees(dip[f])), 1) for f in frames})
    else:
        frames = [int(f) for f in args.frames.split(",")]
    rows = []
    thumb = HandFK.SIDES.index(args.side) * 5  # thumb index in the anchor order
    for t in frames:
        joints14 = np.asarray(td["observations/joint_states/position"][t], dtype=np.float64)
        Ts = camera_poses(cams, joints14)
        link8 = {s: Ts[names.index(f"cam_wrist_{s}")]
                 @ np.linalg.inv(np.asarray(cams[names.index(f"cam_wrist_{s}")]["extrinsics"]))
                 for s in HandFK.SIDES}
        q_meas = {s: np.asarray(td[f"observations/qpos_hand/{s}"][t], dtype=np.float64)[sl]
                  for s in HandFK.SIDES}
        panels = []
        for n in names:
            g, keys, idx, gaps = img_idx[n]
            img = cv2.imdecode(np.asarray(g[keys[idx[t]]]), cv2.IMREAD_COLOR)
            T = np.linalg.inv(Ts[names.index(n)])
            pts = []
            for ci, (_, fn) in enumerate(CANDIDATES):
                if ci not in keep_c:
                    continue
                ap_ = np.zeros(48)
                ap_[14:31] = q_meas["left"] if args.side != "left" else fn(q_meas["left"])
                ap_[31:48] = q_meas["right"] if args.side != "right" else fn(q_meas["right"])
                tip = fk.anchors(link8, ap_)[thumb]
                p = tip @ T[:3, :3].T + T[:3, 3]
                if p[2] <= 0.01:
                    continue
                uv, _ = cv2.projectPoints(p.reshape(1, 1, 3), np.zeros(3), np.zeros(3), K[n], D[n])
                pts.append((ci, uv.ravel()))
            h, w = img.shape[:2]
            inside = [(ci, uv) for ci, uv in pts if 0 <= uv[0] < w and 0 <= uv[1] < h]
            if inside:
                c = np.mean([uv for _, uv in inside], axis=0)
            else:
                c = np.array([w / 2, h / 2])
            for ci, uv in pts:
                u, v = int(round(uv[0])), int(round(uv[1]))
                cv2.circle(img, (u, v), 11, COLORS_BGR[ci], 3)
                cv2.putText(img, str(ci), (u + 12, v - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                            (0, 0, 0), 5, cv2.LINE_AA)
                cv2.putText(img, str(ci), (u + 12, v - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                            COLORS_BGR[ci], 2, cv2.LINE_AA)
            r = args.crop
            x0 = int(np.clip(c[0] - r, 0, max(w - 2 * r, 0)))
            y0 = int(np.clip(c[1] - r, 0, max(h - 2 * r, 0)))
            crop = img[y0:y0 + 2 * r, x0:x0 + 2 * r]
            crop = cv2.copyMakeBorder(crop, 0, 2 * r - crop.shape[0], 0, 2 * r - crop.shape[1],
                                      cv2.BORDER_CONSTANT, value=(0, 0, 0))
            label = (f"{n}  frame {t}  (image {abs(gaps[t]) / 1e6:.0f} ms from grid)"
                     + ("" if inside else "  [thumb not in view]"))
            cv2.rectangle(crop, (0, 0), (2 * r, 34), (0, 0, 0), -1)
            cv2.putText(crop, label, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (255, 255, 255), 1, cv2.LINE_AA)
            panels.append(crop)
        rows.append(np.hstack(panels))
    legend = np.zeros((44 * len(CANDIDATES) // 2 + 20, rows[0].shape[1], 3), np.uint8)
    for ci, (name, _) in enumerate(CANDIDATES):
        if ci not in keep_c:
            continue
        x = 20 + (ci % 2) * rows[0].shape[1] // 2
        y = 34 + (ci // 2) * 44
        cv2.circle(legend, (x + 10, y - 8), 11, COLORS_BGR[ci], 3)
        cv2.putText(legend, f"{ci}: {name}", (x + 34, y), cv2.FONT_HERSHEY_SIMPLEX, 0.85,
                    COLORS_BGR[ci], 2, cv2.LINE_AA)
    out = np.vstack([legend] + rows)
    ep = os.path.splitext(os.path.basename(args.synced))[0]
    tag = os.path.basename(os.path.dirname(os.path.abspath(args.synced))).split("_processed")[0]
    path = os.path.join(args.out, f"thumb_conventions_{args.side}_{tag}_{ep}"
                        + (f"_c{''.join(sorted(map(str, keep_c)))}" if args.candidates else "")
                        + ".png")
    cv2.imwrite(path, out)
    print(f"wrote {path}  ({out.shape[1]}x{out.shape[0]})")


if __name__ == "__main__":
    main()
