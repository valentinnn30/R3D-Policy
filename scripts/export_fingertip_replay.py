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
    ANCHOR_TYPES, HAND_FINGERS, HAND_STATE_SLICES, HandFK, PointCloudPreprocessConfig,
    anchor_layout, fuse_cameras, normalize_anchor_spec)

# The ANCHOR SET of the 2026-09-24 variants: every type, run F's list plus
# the reference tip. Published on ~/anchor_set, one RViz namespace per type.
ANCHOR_SET = [{"type": "tip", "radius_m": 0.03},
              {"type": "pad_back", "radius_m": 0.03},
              {"type": "pad_mid", "radius_m": 0.03},
              {"type": "pad_tip", "radius_m": 0.03},
              {"type": "axial", "radius_m": 0.03},
              {"type": "pad_front", "radius_m": 0.03},
              {"type": "pinch", "fingers": ["index", "middle", "ring"], "radius_m": 0.03}]

K_GROUP = 32  # group_size of the encoder
FPS_CANDIDATES = 512  # fingertip_tokens.fps_candidates default


def pick_patch(dist_row, radius, sampling, xyz, n_cand=FPS_CANDIDATES):
    """The encoder's draw for one anchor: indices of up to K_GROUP in-radius
    points. nearest = the K nearest; fps = farthest-point sampling over the
    n_cand nearest in-radius points, starting at the nearest one (the
    encoder's deterministic rule; float ties may pick differently)."""
    order = np.argsort(dist_row)[:n_cand if sampling == "fps" else K_GROUP]
    cand = order[dist_row[order] <= radius]
    if sampling == "nearest" or len(cand) <= K_GROUP:
        return cand[:K_GROUP]
    p = xyz[cand].astype(np.float64)
    chosen = [0]
    d = ((p - p[0]) ** 2).sum(1)
    for _ in range(K_GROUP - 1):
        j = int(d.argmax())
        chosen.append(j)
        d = np.minimum(d, ((p - p[j]) ** 2).sum(1))
    return cand[chosen]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zarr", default="R3D/data/real_bimanual_mixed150.zarr")
    ap.add_argument("--episode", type=int, default=0)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--preprocess", default="R3D/r3d/config/preprocess/real_bimanual.yaml")
    ap.add_argument("--fk", default="R3D/r3d/config/preprocess/orcahand_v1b_fk.yaml")
    ap.add_argument("--radius", type=float, default=0.03)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--pad-offset", type=float, default=0.009,
                    help="anchor set: FK tip -> pad surface along the pad normal [m]")
    ap.add_argument("--task", default=None,
                    help="anchor set of ONE run: a task yaml name, e.g. "
                         "real_bimanual_fingertip_F (its anchors, radii and pad offset). "
                         "Default: every anchor type (ANCHOR_SET above)")
    ap.add_argument("--sampling", choices=("nearest", "fps"), default=None,
                    help="anchor-set patch draw; default: the task's "
                         "fingertip_tokens.sampling, else nearest")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    set_spec, pad_offset, set_name = ANCHOR_SET, args.pad_offset, "all types"
    sampling = "nearest"
    if args.task:
        tdoc = yaml.safe_load(open(f"R3D/r3d/config/task/{args.task}.yaml"))
        ftc = tdoc.get("fingertip_tokens") or {}
        # no anchors key = the legacy 10 tips (reference, A); no block at all = K
        set_spec = ftc.get("anchors") or (
            [{"type": "tip", "radius_m": float(ftc.get("radius_m", 0.03))}] if ftc else [])
        pad_offset = float(ftc.get("pad_offset_m", args.pad_offset))
        sampling = str(ftc.get("sampling", "nearest"))
        set_name = args.task
    if args.sampling:
        sampling = args.sampling

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
    set_lay = anchor_layout(normalize_anchor_spec(set_spec)) if set_spec else []
    n_set = len(set_lay)
    set_r = np.array([r for *_, r in set_lay])
    anchor_set = np.zeros((T, n_set, 3), np.float32)
    # the encoder's grouping for every set anchor: nearest K_GROUP within ITS radius
    set_patch = np.full((T, n_set, K_GROUP), -1, np.int32)
    set_nvalid = np.zeros((T, n_set), np.int16)
    set_empty = np.zeros((T, n_set), bool)

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
        if n_set:
            anchor_set[k] = fk.anchors(L8, state, spec=set_spec, pad_offset_m=pad_offset)
            ds = np.linalg.norm(fused[None, :, :3] - anchor_set[k][:, None, :], axis=-1)
            ins = np.all((anchor_set[k] >= lo) & (anchor_set[k] <= hi), axis=1)
            for a in range(n_set):
                sel = pick_patch(ds[a], set_r[a], sampling, fused[:, :3])
                set_nvalid[k, a] = len(sel)
                set_empty[k, a] = len(sel) == 0 or not ins[a]
                if not set_empty[k, a]:
                    set_patch[k, a, :len(sel)] = sel
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
        fk_config=os.path.abspath(args.fk), zarr=os.path.abspath(args.zarr),
        anchor_set=anchor_set, anchor_set_types=np.array(ANCHOR_TYPES),
        anchor_set_type=np.array([t for _, _, t, _ in set_lay]),
        anchor_set_hand=np.array([h for h, *_ in set_lay]),
        anchor_set_finger=np.array([f for _, f, _, _ in set_lay]),
        anchor_set_radius=set_r, anchor_set_patch=set_patch,
        anchor_set_nvalid=set_nvalid, anchor_set_empty=set_empty,
        anchor_set_name=f"{set_name}, sampling {sampling}", anchor_set_sampling=sampling,
        pad_offset=pad_offset)
    print(f"wrote {out}  ({os.path.getsize(out) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
