#!/usr/bin/env python3
"""Verify RealMultiCamUnfusedDataset against the fused pipeline it replaces.

The dangerous failure here is silent: a crop applied in the wrong frame, a pose
tag that does not belong to the cloud it is attached to, or a sentinel point
reaching the encoder all train perfectly happily and just produce a worse
policy. So these are assertions, not eyeballing.

    python scripts/check_unfused_dataset.py \
        --task R3D/r3d/config/task/real_bimanual_unfused.yaml
"""
import argparse
import copy
import sys

import numpy as np
import yaml

sys.path.insert(0, "R3D")
from r3d.common.real_preprocess import PointCloudPreprocessConfig  # noqa: E402
from r3d.dataset.real_multicam_dataset import RealMultiCamUnfusedDataset  # noqa: E402


def rot_from_r6d(v):
    """Inverse of _r6d_from_matrix: [col0(3), col1(3)] -> 3x3 via Gram-Schmidt."""
    a, b = np.asarray(v[:3], float), np.asarray(v[3:6], float)
    c0 = a / np.linalg.norm(a)
    c1 = b - (c0 @ b) * c0
    c1 /= np.linalg.norm(c1)
    return np.stack([c0, c1, np.cross(c0, c1)], axis=1)


def build(task_yaml, **over):
    task = yaml.safe_load(open(task_yaml))
    spec = dict(task["dataset"])
    # Hydra resolves ${task.num_groups_per_camera}; this harness does not run
    # Hydra, so pull it from the same top-level key Hydra would.
    spec["num_groups_per_camera"] = task["num_groups_per_camera"]
    kw = {k: v for k, v in spec.items() if not (
        isinstance(v, str) and v.startswith("${"))}
    kw.pop("_target_")
    kw.update(dict(horizon=16, pad_before=1, pad_after=7, n_obs_steps=2,
                   use_target_ee=False, use_data_augmentation=False,
                   use_color_jitter=False))
    kw.update(over)
    return RealMultiCamUnfusedDataset(**kw), kw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="R3D/r3d/config/task/real_bimanual_unfused.yaml")
    ap.add_argument("--samples", type=int, default=40)
    args = ap.parse_args()

    fails = []

    def check(name, ok, detail=""):
        print(f"  {'PASS' if ok else 'FAIL'}  {name}{'  ' + detail if detail else ''}")
        if not ok:
            fails.append(name)

    # ---------------------------------------------------------------- run B --
    print("\n[1] Run B (pose_source=extrinsic), randomization OFF")
    ds, kw = build(args.task, use_extrinsics_randomization=False)
    cfg = PointCloudPreprocessConfig.from_yaml(kw["preprocess_config"])
    lo, hi = cfg.bounds
    n_cams = len(ds.cam_keys)
    rng = np.random.default_rng(0)

    shapes_ok = valid_ok = inside_ok = True
    worst_inside = 1.0
    n_invalid = 0
    for idx in rng.integers(0, len(ds), size=args.samples):
        s = ds[int(idx)]
        obs = s["obs"]
        for i, key in enumerate(ds.cam_keys):
            pc = obs[key].numpy()
            if pc.shape != (2, ds.points_per_camera_encoder[i], 6):
                shapes_ok = False
            if obs[ds.campose_keys[i]].numpy().shape != (2, 9):
                shapes_ok = False
            for t in range(pc.shape[0]):
                v = obs["cam_valid"].numpy()[t, i]
                if v not in (0.0, 1.0):
                    valid_ok = False
                if v == 0.0:
                    n_invalid += 1
                    continue
                # The pose tag must be the transform that actually placed this
                # cloud: send the points through it and they have to land in
                # the workspace box the crop used.
                pose = obs[ds.campose_keys[i]].numpy()[t]
                T = np.eye(4)
                T[:3, 3] = pose[:3]        # translation
                T[:3, :3] = rot_from_r6d(pose[3:])   # 6D rotation
                base = pc[t, :, :3] @ T[:3, :3].T + T[:3, 3]
                frac = np.all((base > lo) & (base < hi), axis=-1).mean()
                worst_inside = min(worst_inside, frac)
                if frac < 0.99:
                    inside_ok = False

    check("shapes match the configured budgets", shapes_ok)
    check("cam_valid is strictly 0/1", valid_ok)
    check("pose tag round-trips: >=99% of each valid cloud lands in the "
          "workspace", inside_ok, f"(worst frame {worst_inside:.4f})")
    check("no sentinel leaked into a valid cloud", worst_inside > 0.9,
          f"(a sentinel would read ~0.0)")
    print(f"  info  {n_invalid} of {args.samples * 2 * n_cams} "
          f"camera-frames were masked out")

    # ------------------------------------------------------- token budgets --
    print("\n[2] Budgets against the fused run")
    check("total points == fused num_points",
          sum(ds.points_per_camera_encoder) == cfg.num_points,
          f"({sum(ds.points_per_camera_encoder)} vs {cfg.num_points})")
    check("total tokens == 512", sum(ds.num_groups_per_camera) == 512,
          f"({ds.num_groups_per_camera})")
    check("every camera has at least as many points as groups",
          all(p >= g for p, g in zip(ds.points_per_camera_encoder,
                                     ds.num_groups_per_camera)))

    # -------------------------------------------------------- normalization --
    print("\n[3] one shared isotropic length unit for clouds AND pose")
    from r3d.dataset.real_multicam_dataset import METRIC_SCALE_M
    R = METRIC_SCALE_M
    nrm = ds.get_normalizer()
    rng = np.random.default_rng(1)

    for i, key in enumerate(ds.cam_keys):
        raw = ds[0]["obs"][key]
        n = nrm[key].normalize(raw).detach().numpy()
        r = raw.numpy()
        check(f"{key} xyz scaled by exactly 1/{R} (isotropic)",
              np.allclose(n[..., :3], r[..., :3] / R, atol=1e-6))
        check(f"{key} rgb mapped [0,1] -> [-1,1]",
              np.allclose(n[..., 3:], r[..., 3:] * 2.0 - 1.0, atol=1e-6))

    # Every camera must get the SAME factor, or shared encoder weights see the
    # same physical shape at different scales -- the failure the per-camera
    # boxes had.
    factors = [nrm[k].params_dict["scale"][:3].detach().numpy()
               for k in ds.cam_keys]
    check("all cameras share one scale vector",
          all(np.allclose(f, factors[0]) for f in factors),
          f"({np.round(factors[0], 5)})")
    check("that scale is isotropic (a cube stays a cube)",
          np.allclose(factors[0], factors[0][0]))
    check("campose translation carries the same factor",
          np.allclose(nrm[ds.campose_keys[0]].params_dict["scale"][:3]
                      .detach().numpy(), factors[0]))

    # The encoder's PositionEmbeddingRandom refuses group centres outside
    # [-1, 1], so this is a hard requirement, not a preference.
    worst_abs, worst_inside = 0.0, 1.0
    for idx in rng.integers(0, len(ds), size=10):
        obs = ds[int(idx)]["obs"]
        for i, key in enumerate(ds.cam_keys):
            n = nrm[key].normalize(obs[key]).detach().numpy()
            npose = nrm[ds.campose_keys[i]].normalize(
                obs[ds.campose_keys[i]]).detach().numpy()
            for t in range(n.shape[0]):
                if obs["cam_valid"].numpy()[t, i] == 0.0:
                    continue
                worst_abs = max(worst_abs, np.abs(n[t, :, :3]).max())
                T = np.eye(4)
                T[:3, 3] = npose[t][:3]
                T[:3, :3] = rot_from_r6d(npose[t][3:])
                base = n[t, :, :3] @ T[:3, :3].T + T[:3, 3]
                worst_inside = min(worst_inside, np.all(
                    (base > lo / R) & (base < hi / R), axis=-1).mean())
    check("normalized xyz stays inside [-1, 1] (PositionEmbeddingRandom "
          "rejects anything else)", worst_abs <= 1.0,
          f"(max |xyz| {worst_abs:.4f})")
    check("normalized cloud composes with normalized pose into the scaled "
          "workspace -- the rigid transform survives, no scale term left",
          worst_inside == 1.0, f"(worst frame {worst_inside:.4f})")

    print("\n[4] Ablation A' (pose_source=proprio)")
    # Share the loaded ReplayBuffer instead of building a second dataset: it is
    # ~12.6 GiB in RAM for this zarr, and two copies get the script OOM-killed
    # (exit 137, no output). Same trick get_validation_dataset uses -- only
    # `pose_source` changes, and `_campose` reads it off self at call time.
    ds2 = copy.copy(ds)
    ds2.pose_source = "proprio"
    obs = ds2[0]["obs"]
    p0 = obs[ds2.campose_keys[0]].numpy()
    check("static camera gets a zero pose (identity embedding only)",
          np.allclose(p0, 0.0))
    for i in (1, 2):
        pi = obs[ds2.campose_keys[i]].numpy()
        a, b = ds2.camera_arm_slice[i]
        ee = obs["agent_pos"].numpy()[:, a:b]
        check(f"cam{i} pose translation == agent_pos[{a}:{b}] xyz",
              np.allclose(pi[:, :3], ee[:, :3], atol=1e-6))
    check("A' pose differs from run B's extrinsic tag",
          not np.allclose(obs[ds2.campose_keys[1]].numpy(),
                          ds[0]["obs"][ds.campose_keys[1]].numpy()))

    # -------------------------------------------------------- sentinel path --
    # `preprocess_camera_frame` writes tiled points ~100 m outside the box for a
    # camera that saw nothing, relying on the FUSED crop to delete them. There
    # is no fused crop here, so this is the one path that could put garbage
    # geometry straight into the encoder. Checked directly rather than by
    # sampling, because these frames are rare enough that 40 random windows
    # miss them (measured: 2 of 30 episodes).
    print("\n[5] Sentinel frames are masked, not encoded")
    from r3d.common.real_preprocess import select_camera_frame
    lo_ws, hi_ws = cfg.bounds
    T = np.eye(4)
    far = np.tile((hi_ws + 100.0), (3840, 1))
    sentinel = np.concatenate([far, np.zeros((3840, 3))], axis=-1)
    out, ok = select_camera_frame(sentinel, T, cfg, 3200, 200,
                                  rng=np.random.default_rng(0))
    check("a sentinel cloud is reported invalid", ok is False)
    check("a sentinel cloud is emitted as zeros, not far-away geometry",
          np.abs(out[:, :3]).max() == 0.0, f"(max |xyz| {np.abs(out[:, :3]).max()})")
    check("emitted shape is still the configured budget", out.shape == (3200, 6))

    print("\n" + ("ALL CHECKS PASSED" if not fails
                  else f"{len(fails)} FAILED: {fails}"))
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
