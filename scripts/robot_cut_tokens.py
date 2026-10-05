#!/usr/bin/env python3
"""How many scene tokens land on the robot, and what the robot cut buys.

Step 1 of ~/ros2_ws/robot_cut_r1_plan.txt. Per frame, K's exact stage-2 cloud
(fuse_cameras, method random, seeded like export_fingertip_replay.py so the
replay shows the same points), labelled keep / arm / tower by RobotCut, then
the encoder's own token centres -- deterministic FPS (the inference setting:
pytorch3d sample_farthest_points from index 0, fps_random_config off):

  (a) 512 centres on the UNCUT cloud   -- the fused baseline's tokens
  (b) 568 centres on the UNCUT cloud   -- K's tokens
  (c) 512 centres on the CUT cloud     -- R1's tokens, all relevant by construction

A centre counts as RELEVANT when its own point is not robot. Also reported:
how many of each token's 32 neighbours are robot (patch contamination).

Prediction: if (b)'s relevant count exceeds 512, K still has more tokens on
the relevant part than R1 does, and should stay ahead; below 512, R1 should win.

    micromamba activate r3d && unset PYTHONPATH && cd ~/R3D-Policy
    python scripts/robot_cut_tokens.py                       # stride 5, default cut
    python scripts/robot_cut_tokens.py --arm-padding 0.06 --forearm-radius 0.2
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import zarr
from pytorch3d.ops import knn_points, sample_farthest_points

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "R3D"))
from r3d.common.real_preprocess import PointCloudPreprocessConfig, fuse_cameras  # noqa: E402
from r3d.common.robot_cut import (CUT_LABELS, LABEL_NAMES, RobotCut, add_cli_args,  # noqa: E402
                                  default_paths, params_from_args, params_tag)

K_GROUP = 32
PHASES = [(0.0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.0001)]


def fused_frame(root, statics, n_cam, cfg, t, seed):
    """K's stage-2 cloud for zarr frame t (identical to export_fingertip_replay)."""
    clouds = [np.asarray(root[f"data/point_cloud_cam{i}"][t]) for i in range(n_cam)]
    Ts = [statics[i] if i in statics else
          np.asarray(root[f"data/extrinsics_cam{i}"][t], dtype=np.float64)
          for i in range(n_cam)]
    return fuse_cameras(clouds, Ts, cfg, method="random",
                        rng=np.random.default_rng(seed + int(t)),
                        camera_ids=list(range(n_cam)), n_cameras=n_cam)


def token_sets(xyz, keep, device, groups=(512, 568), cut_groups=512):
    """Deterministic FPS centres + kNN patches for a batch of frames.

    xyz (B, N, 3) float32, keep (B, N) bool. Returns dict name -> (centre_idx
    (B, G) into the UNCUT cloud, patch_idx (B, G, 32) into the uncut cloud)."""
    x = torch.from_numpy(xyz).to(device)
    out = {}
    for g in groups:
        _, ci = sample_farthest_points(x, K=g)
        _, pi, _ = knn_points(torch.gather(x, 1, ci[..., None].expand(-1, -1, 3)), x, K=K_GROUP)
        out[f"uncut{g}"] = (ci.cpu().numpy(), pi.cpu().numpy())
    # cut cloud: kept points packed to the front, lengths mask the rest
    B, N = keep.shape
    order = np.argsort(~keep, axis=1, kind="stable")       # kept first, original order
    lengths = keep.sum(1)
    xc = torch.from_numpy(np.take_along_axis(xyz, order[..., None], 1)).to(device)
    ln = torch.from_numpy(lengths).to(device)
    _, ci = sample_farthest_points(xc, lengths=ln, K=cut_groups)
    cen = torch.gather(xc, 1, ci[..., None].expand(-1, -1, 3))
    _, pi, _ = knn_points(cen, xc, lengths2=ln, K=K_GROUP)
    ci, pi = ci.cpu().numpy(), pi.cpu().numpy()
    # back to indices into the uncut cloud
    out[f"cut{cut_groups}"] = (np.take_along_axis(order, ci, 1),
                               np.stack([order[b][pi[b]] for b in range(B)]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zarr", default="R3D/data/real_bimanual_mixed150.zarr")
    ap.add_argument("--joints", default=None, help="default: <zarr>_joints14.npz")
    ap.add_argument("--stride", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--out-dir", default="R3D/data/outputs/robot_cut")
    add_cli_args(ap)
    args = ap.parse_args()

    paths = default_paths()
    params = params_from_args(args)
    rc = RobotCut(**paths, **params)
    cfg = PointCloudPreprocessConfig.from_yaml(paths["preprocess_yaml"])
    root = zarr.open(args.zarr, "r")
    J = np.load(args.joints or args.zarr.replace(".zarr", "_joints14.npz"))["joints14"]
    state = np.asarray(root["data/state"][:], dtype=np.float64)
    names = list(root["meta"].attrs["camera_names"])
    n_cam = len(names)
    statics = {i: np.asarray(root[f"meta/extrinsics_cam{i}"][:], dtype=np.float64)
               for i in range(n_cam) if f"meta/extrinsics_cam{i}" in root}
    ends = np.asarray(root["meta/episode_ends"][:])
    starts = np.r_[0, ends[:-1]]
    frames = np.arange(0, ends[-1], args.stride)
    ep = np.searchsorted(ends, frames, side="right")
    phase = (frames - starts[ep]) / np.maximum(ends[ep] - starts[ep] - 1, 1)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"{len(frames)} frames (stride {args.stride}), cut {params}, FPS on {device}")

    F = len(frames)
    rec = {k: np.zeros((F,) + s, np.int32) for k, s in [
        ("removed", (n_cam, len(LABEL_NAMES))),   # per camera: points per label
        ("rel512", (n_cam,)), ("rel568", (n_cam,)), ("cut512", (n_cam,)),
        ("robot_tok512", (len(LABEL_NAMES),)), ("robot_tok568", (len(LABEL_NAMES),)),
        ("patch_robot512", ()), ("patch_robot568", ()),   # robot points inside patches
    ]}
    t0 = time.time()
    for s in range(0, F, args.batch):
        idx = range(s, min(s + args.batch, F))
        fused = [fused_frame(root, statics, n_cam, cfg, frames[k], args.seed) for k in idx]
        xyz = np.stack([f[:, :3] for f in fused]).astype(np.float32)
        cam = np.stack([f[:, 6:].argmax(1) for f in fused])
        lab = np.stack([rc.labels(f[:, :3], J[frames[k]], state[frames[k]], cam[b])
                        for b, (f, k) in enumerate(zip(fused, idx))])
        cut = np.isin(lab, CUT_LABELS)
        sets = token_sets(xyz, ~cut, device)
        for b, k in enumerate(idx):
            for c in range(n_cam):
                rec["removed"][k, c] = np.bincount(lab[b][cam[b] == c], minlength=len(LABEL_NAMES))
            for name, key in (("uncut512", "512"), ("uncut568", "568")):
                ci, pi = sets[name]
                cl = lab[b][ci[b]]
                for c in range(n_cam):
                    rec[f"rel{key}"][k, c] = np.sum(~cut[b][ci[b]] & (cam[b][ci[b]] == c))
                rec[f"robot_tok{key}"][k] = np.bincount(cl, minlength=len(LABEL_NAMES))
                rec[f"patch_robot{key}"][k] = np.sum(cut[b][pi[b]])
            ci, _ = sets["cut512"]
            for c in range(n_cam):
                rec["cut512"][k, c] = np.sum(cam[b][ci[b]] == c)
        if (s // args.batch) % 20 == 0:
            print(f"  {min(s + args.batch, F)}/{F}  {time.time() - t0:.0f} s", flush=True)

    os.makedirs(args.out_dir, exist_ok=True)
    tag = f"{params_tag(params)}_s{args.stride}"
    np.savez(os.path.join(args.out_dir, f"tokens_{tag}.npz"), frames=frames, episode=ep,
             phase=phase, camera_names=np.array(names), params=json.dumps(params), **rec)
    summary = report(rec, phase, names, params)
    with open(os.path.join(args.out_dir, f"tokens_{tag}.json"), "w") as f:
        json.dump(summary, f, indent=1)
    print(f"wrote {args.out_dir}/tokens_{tag}.npz / .json")


def _q(v):
    v = np.asarray(v, float)
    return {"mean": float(v.mean()), "p10": float(np.percentile(v, 10)),
            "median": float(np.median(v)), "p90": float(np.percentile(v, 90))}


def report(rec, phase, names, params):
    rem = rec["removed"]
    n_pts = rem.sum((1, 2))
    cut = rem[:, :, list(CUT_LABELS)].sum(2)          # (F, cam)
    rel512, rel568 = rec["rel512"].sum(1), rec["rel568"].sum(1)
    s = {"params": params, "frames": int(len(phase)),
         "points_removed": _q(cut.sum(1)), "points_removed_frac": _q(cut.sum(1) / n_pts),
         **{f"points_{n}": _q(rem[:, :, i].sum(1)) for i, n in enumerate(LABEL_NAMES) if i},
         "relevant_tokens_512": _q(rel512), "relevant_tokens_568": _q(rel568),
         "frac_frames_568_above_512": float(np.mean(rel568 > 512)),
         "patch_robot_points_512": _q(rec["patch_robot512"]),
         "per_camera": {}, "per_phase": {}}
    for c, n in enumerate(names):
        s["per_camera"][n] = {"points": _q(rem[:, c].sum(1)), "removed": _q(cut[:, c]),
                              "tokens_512": _q(rec["rel512"][:, c] + 0),
                              "relevant_512": _q(rec["rel512"][:, c]),
                              "relevant_568": _q(rec["rel568"][:, c]),
                              "cut_512": _q(rec["cut512"][:, c])}
    for lo, hi in PHASES:
        m = (phase >= lo) & (phase < hi)
        s["per_phase"][f"{lo:.2f}-{min(hi, 1):.2f}"] = {
            "points_removed": _q(cut.sum(1)[m]), "relevant_512": _q(rel512[m]),
            "relevant_568": _q(rel568[m]), "frac_568_above_512": float(np.mean(rel568[m] > 512))}

    def row(label, d):
        return f"  {label:34s} mean {d['mean']:7.1f}  p10 {d['p10']:7.1f}  " \
               f"median {d['median']:7.1f}  p90 {d['p90']:7.1f}"
    print(f"\n=== robot cut {params}, {len(phase)} frames ===")
    print(row("points removed (of 8192)", s["points_removed"]))
    for i, n in enumerate(LABEL_NAMES):
        if i:
            print(row(f"  {n}" + (" (kept: keep zone)" if n == "saved" else ""), s[f"points_{n}"]))
    print(row("(a) relevant tokens, 512 uncut", s["relevant_tokens_512"]))
    print(row("(b) relevant tokens, 568 uncut (K)", s["relevant_tokens_568"]))
    print("  (c) relevant tokens, 512 cut (R1)   512 by construction")
    print(f"  frames where K has MORE than 512 relevant tokens: "
          f"{100 * s['frac_frames_568_above_512']:.1f}%")
    print(row("robot points inside 512 patches", s["patch_robot_points_512"]))
    print("\n  per camera (median): points / removed / relevant 512 / relevant 568 / R1 512")
    for n, d in s["per_camera"].items():
        print(f"    {n:16s} {d['points']['median']:6.0f} {d['removed']['median']:8.0f} "
              f"{d['relevant_512']['median']:12.0f} {d['relevant_568']['median']:13.0f} "
              f"{d['cut_512']['median']:8.0f}")
    print("\n  per phase (mean): removed / relevant 512 / relevant 568 / % frames K > 512")
    for p, d in s["per_phase"].items():
        print(f"    {p}  {d['points_removed']['mean']:7.0f} {d['relevant_512']['mean']:9.1f} "
              f"{d['relevant_568']['mean']:9.1f} {100 * d['frac_568_above_512']:8.1f}%")
    return s


if __name__ == "__main__":
    main()
