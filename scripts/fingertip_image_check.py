#!/usr/bin/env python3
"""Draw the FK fingertips into the REAL camera images -- the mount check.

One figure per episode: rows = episode phases (one recorded frame each),
columns = the three cameras (top, left wrist, right wrist). Every panel is the
recorded colour image of that camera at that frame, with all 10 FK fingertips
projected into it:

  * colour = finger (thumb red, index blue, middle green, ring purple, pinky
    orange); circle = LEFT hand, square = RIGHT hand;
  * a fingertip that is behind the camera or outside the image is not drawn.

Where everything comes from -- the same sources and routes the policy uses:

  images      R3D/data/real_bimanual_2d_mixed150.zarr, image_cam{0,1,2}
              (322x182, frame-identical to the 3D zarr)
  intrinsics  K + full OpenCV distortion from a synced 2D h5 (identical in all
              150 files), already rescaled to 322x182
  arm joints  the 2D zarr's `joint_states` (14 recorded joint angles)
  cameras     top: the static T_world<-cam from real_bimanual.yaml;
              wrists: base @ panda_fk_flange(joints) @ T_link8<-cam -- the
              route that places the wrist clouds and that inference uses
  fingertips  HandFK (analytic mount + the frame's recorded hand angles),
              flange from the same joint FK

    micromamba activate r3d && unset PYTHONPATH
    cd ~/R3D-Policy
    python scripts/fingertip_image_check.py --episode 0
"""
import argparse
import os
import sys

import cv2
import h5py
import numpy as np
import yaml
import zarr

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "R3D"))
from r3d.common.real_preprocess import HAND_FINGERS, HandFK, panda_fk_flange  # noqa: E402

FINGER_COLORS = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00"]
H5_CAMS = {"cam_top": "oakd_TOP_CAMERA",
           "cam_wrist_left": "orbbec_left_wrist_view",
           "cam_wrist_right": "orbbec_right_wrist_view"}


def camera_poses(cams, joints14):
    """T_world<-cam per camera -- identical to ObservationBuilder.nominal_extrinsics."""
    out = []
    for cam in cams:
        he = np.asarray(cam["extrinsics"], dtype=np.float64)
        if cam.get("mount", "static") != "wrist":
            out.append(he)
            continue
        fp = cam["flange_pose"]
        a, b = fp["joint_slice"]
        out.append(np.asarray(fp["base"], dtype=np.float64)
                   @ panda_fk_flange(joints14[a:b]) @ he)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episode", type=int, default=0)
    ap.add_argument("--phases", default="0.10,0.30,0.45,0.55,0.65,0.80")
    ap.add_argument("--zarr2d", default="R3D/data/real_bimanual_2d_mixed150.zarr")
    ap.add_argument("--intrinsics-h5",
                    default=os.path.expanduser("~/ros2_ws/tmp/1-90_RED_2d_322x182/0000.h5"))
    ap.add_argument("--preprocess", default="R3D/r3d/config/preprocess/real_bimanual.yaml")
    ap.add_argument("--fk", default="R3D/r3d/config/preprocess/orcahand_v1b_fk.yaml")
    ap.add_argument("--out", default="R3D/data/outputs/fingertip_occupancy")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    raw = yaml.safe_load(open(args.preprocess))
    cams = raw["cameras"]
    names = [c["name"] for c in cams]
    root = zarr.open(args.zarr2d, mode="r")
    if list(root["meta"].attrs["camera_names"]) != names:
        raise ValueError("2D zarr camera order != preprocess yaml camera order")
    h5 = h5py.File(args.intrinsics_h5, "r")
    K = {n: h5[f"observations/images/{H5_CAMS[n]}/intrinsics/K"][:] for n in names}
    D = {n: h5[f"observations/images/{H5_CAMS[n]}/intrinsics/disto"][:] for n in names}
    fk = HandFK(args.fk)

    ends = np.asarray(root["meta/episode_ends"][:])
    starts = np.concatenate([[0], ends[:-1]])
    s, e = int(starts[args.episode]), int(ends[args.episode])
    phases = [float(p) for p in args.phases.split(",")]

    fig, axes = plt.subplots(len(phases), len(names),
                             figsize=(6.4 * len(names), 3.8 * len(phases)), squeeze=False)
    for row, ph in enumerate(phases):
        t = s + int(round(ph * (e - s - 1)))
        joints14 = np.asarray(root["data/joint_states"][t], dtype=np.float64)
        state = np.asarray(root["data/state"][t], dtype=np.float64)
        Ts = camera_poses(cams, joints14)
        link8 = {side: Ts[names.index(f"cam_wrist_{side}")]
                 @ np.linalg.inv(np.asarray(cams[names.index(f"cam_wrist_{side}")]["extrinsics"]))
                 for side in HandFK.SIDES}
        tips = fk.anchors(link8, state)                       # (10, 3) world
        for col, n in enumerate(names):
            ax = axes[row, col]
            img = np.asarray(root[f"data/image_cam{col}"][t])
            h, w = img.shape[:2]
            ax.imshow(img)
            T_cam_world = np.linalg.inv(Ts[col])
            pc = tips @ T_cam_world[:3, :3].T + T_cam_world[:3, 3]
            front = pc[:, 2] > 0.01
            uv, _ = cv2.projectPoints(pc.reshape(-1, 1, 3), np.zeros(3), np.zeros(3),
                                      K[n], D[n])
            uv = uv.reshape(-1, 2)
            shown = 0
            for k in range(10):
                u, v = uv[k]
                if not front[k] or not (0 <= u < w and 0 <= v < h):
                    continue
                f = k % 5
                ax.scatter(u, v, s=110, marker="o" if k < 5 else "s",
                           facecolors="none", edgecolors=FINGER_COLORS[f], linewidths=2.2)
                shown += 1
            ax.set_xlim(0, w)
            ax.set_ylim(h, 0)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(f"{n} | phase {ph:.2f} (frame {t}) | {shown}/10 tips in view",
                         fontsize=10)
    handles = [plt.Line2D([], [], ls="", marker="o", mfc="none", mec=c, mew=2, ms=10, label=f)
               for f, c in zip(HAND_FINGERS, FINGER_COLORS)]
    handles += [plt.Line2D([], [], ls="", marker="o", mfc="none", mec="k", ms=10, label="left hand"),
                plt.Line2D([], [], ls="", marker="s", mfc="none", mec="k", ms=10, label="right hand")]
    fig.legend(handles=handles, loc="upper center", ncol=7, fontsize=11,
               bbox_to_anchor=(0.5, 1.0))
    fig.suptitle(f"episode {args.episode}: FK fingertips projected into the recorded images",
                 y=1.02, fontsize=13)
    fig.tight_layout()
    path = os.path.join(args.out, f"fingertips_in_images_ep{args.episode:03d}.png")
    fig.savefig(path, dpi=110, bbox_inches="tight")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
