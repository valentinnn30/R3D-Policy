#!/usr/bin/env python3
"""Verify RealHybridDataset block by block against the pipelines it recycles.

Every block is recomputed INDEPENDENTLY from the raw zarr with the shared
functions, and must come out bit-identical (augmentation off, same seed):

* unfused 3D block == select_camera_frame(stored cloud, zarr extrinsic, cfg,
  points, groups, rng)  -- the Run B / A' path
* fused 3D block   == fuse_cameras(stored wrist clouds, zarr extrinsics, cfg
  with num_points overridden, rng)  -- the run A path
* image block      == zarr pixels / 255, channels-first  -- the 2D path
* campose          == the per-camera rule for EVERY (pose_source_3d,
  pose_source_2d) pair, including the 2D extrinsic tag agreeing with the 3D
  zarr's stored per-frame extrinsic (FK composed at load vs at conversion)
* cam_valid in block order, normalizer keys == shape_meta, and the degenerate
  configurations raise.

Builds ONE hybrid dataset at a time and frees it before the next: two resident
3D buffers on this box get the process OOM-killed silently.

    python scripts/check_hybrid_dataset.py                # H1, H2-U, H2-F
    python scripts/check_hybrid_dataset.py --only h2f --samples 3
"""
import argparse
import gc
import itertools
import sys

import numpy as np
import torch
import yaml
import zarr

sys.path.insert(0, "R3D")
from r3d.common.real_preprocess import (  # noqa: E402
    PointCloudPreprocessConfig,
    fuse_cameras,
    pose9_from_matrix,
    pose9_from_pos_quat,
    select_camera_frame,
)
from r3d.dataset.real_hybrid_dataset import RealHybridDataset  # noqa: E402
from r3d.dataset.real_image_dataset import campose_for_camera, load_preprocess_geometry  # noqa: E402

TASKS = {
    "h1": "R3D/r3d/config/task/real_bimanual_hybrid_top3d.yaml",
    "h2u": "R3D/r3d/config/task/real_bimanual_hybrid_wrist3d.yaml",
    "h2f": "R3D/r3d/config/task/real_bimanual_hybrid_wrist3d_fused.yaml",
}
SOURCES = ("extrinsic", "proprio", "none")

fails = []
n_checks = [0]


def check(name, ok, detail=""):
    n_checks[0] += 1
    if not ok:
        print(f"  FAIL  {name}  {detail}")
        fails.append(name)
    return ok


def kwargs_from_task(path, s3, s2):
    task = yaml.safe_load(open(path))
    spec = dict(task["dataset"])
    spec.pop("_target_")
    if "num_groups_per_camera" in spec:
        spec["num_groups_per_camera"] = task["num_groups_per_camera"]
    kw = {k: v for k, v in spec.items() if not (isinstance(v, str) and v.startswith("${"))}
    kw.update(horizon=16, pad_before=1, pad_after=7, n_obs_steps=2, use_target_ee=False,
              use_data_augmentation=False, use_color_jitter=False,
              rgb_pixel_noise_std=0.0, image_shift_px=0,
              pose_source_3d=s3, pose_source_2d=s2)
    return task, kw


def expected_campose(kind_2d, cam, s, T_zarr, agent_pos_t, joints_t, arm_slice, geo):
    if s == "none":
        return np.zeros(9)
    if s == "proprio":
        if arm_slice is None:
            return np.zeros(9)
        a, b = arm_slice
        return pose9_from_pos_quat(agent_pos_t[a:b][:3], agent_pos_t[a:b][3:7])
    if kind_2d:
        return campose_for_camera(geo[cam], "extrinsic", None, joints_t, agent_pos_t)
    return pose9_from_matrix(T_zarr)


def verify_pair(name, path, s3, s2, n_samples, z3, z2, geo):
    task, kw = kwargs_from_task(path, s3, s2)
    try:
        ds = RealHybridDataset(**kw)
    except ValueError as e:
        return "raised", str(e)

    cfg = PointCloudPreprocessConfig.from_yaml(kw["preprocess_config_3d"])
    slices = kw["camera_arm_slice"]
    obs_meta = task["shape_meta"]["obs"]

    # shape_meta <-> blocks <-> normalizer
    check(f"{name} normalizer keys", set(ds.get_normalizer().params_dict.keys())
          == set(obs_meta) | {"action"})
    check(f"{name} cam_valid width", obs_meta["cam_valid"]["shape"] == [len(ds.blocks)])

    rs = np.random.default_rng(7)
    for idx in rs.integers(0, len(ds), size=n_samples):
        idx = int(idx)
        np.random.seed(idx % 100000)
        out = ds[idx]
        obs = out["obs"]

        # independent reconstruction ------------------------------------------
        np.random.seed(idx % 100000)
        rng3 = np.random.default_rng(np.random.randint(0, 2 ** 31 - 1))
        s3d = ds.ds3d.sampler.sample_sequence(idx)
        s2d = ds.ds2d.sampler.sample_sequence(idx)
        agent = np.asarray(s3d["state"][:2], dtype=np.float64)
        joints = np.asarray(s2d["joint_states"][:2], dtype=np.float64)

        b_start, _, smp_start, _ = (int(v) for v in ds.ds3d.sampler.indices[idx])

        def frame_of(t):
            # window position -> replay-buffer frame, including start padding
            return b_start + max(0, t - smp_start)

        def T_of(cam, t):
            key = f"extrinsics_cam{cam}"
            if key in s3d:                       # loaded by the 3D sub-dataset
                return np.asarray(s3d[key][t], dtype=np.float64)
            if key in z3["data"]:                # a wrist camera that is 2D here
                return np.asarray(z3["data"][key][frame_of(t)], dtype=np.float64)
            return np.asarray(z3["meta"][key][...], dtype=np.float64)

        # replay the dataset's rng consumption: unfused -> per camera, per frame
        exp_pc = {}
        if ds.fuse_3d:
            cfg_f = PointCloudPreprocessConfig(**{**cfg.__dict__,
                                                  "num_points": kw["fused_num_points"]})
            exp_pc["point_cloud"] = np.stack([
                fuse_cameras([s3d[f"point_cloud_cam{c}"][t] for c in kw["cameras_3d"]],
                             [T_of(c, t) for c in kw["cameras_3d"]], cfg_f,
                             method="random", rng=rng3)
                for t in range(2)])
        else:
            valid = {}
            for j, c in enumerate(kw["cameras_3d"]):
                clouds, oks = [], []
                for t in range(2):
                    pts, ok = select_camera_frame(
                        s3d[f"point_cloud_cam{c}"][t], T_of(c, t), cfg,
                        kw["points_per_camera_encoder"][j], kw["num_groups_per_camera"][j],
                        rng=rng3)
                    clouds.append(pts)
                    oks.append(float(ok))
                exp_pc[f"point_cloud_cam{c}"] = np.stack(clouds)
                valid[c] = oks

        for b_i, b in enumerate(ds.blocks):
            got = obs[b["key"]].numpy()
            if b["kind"] == "image":
                (c,) = b["cameras"]
                # via the sampler, so episode-start padding is reproduced
                raw = np.asarray(s2d[f"image_cam{c}"][:2])
                exp = (raw.astype(np.float32) / 255.0).transpose(0, 3, 1, 2)
                check(f"{name} {b['key']} bit-identical", np.array_equal(got, exp))
                check(f"{name} {b['key']} valid", bool((obs["cam_valid"][:, b_i] == 1).all()))
            else:
                exp = exp_pc[b["key"]].astype(np.float32)
                check(f"{name} {b['key']} bit-identical", np.array_equal(got, exp),
                      f"max|d|={np.abs(got - exp).max() if got.shape == exp.shape else 'shape'}")
                if b["kind"] == "pc":
                    (c,) = b["cameras"]
                    check(f"{name} {b['key']} valid",
                          obs["cam_valid"][:, b_i].tolist() == valid[c])
                else:
                    check(f"{name} fused valid", bool((obs["cam_valid"][:, b_i] == 1).all()))
                    check(f"{name} fused has no campose",
                          not any(k.startswith("campose_cam") and int(k[11:]) in b["cameras"]
                                  for k in obs))

            if b["campose_key"] is not None:
                (c,) = b["cameras"]
                s = s2 if b["kind"] == "image" else s3
                sl = None if slices[c] is None else tuple(slices[c])
                for t in range(2):
                    exp_p = expected_campose(b["kind"] == "image", c, s, T_of(c, t),
                                             agent[t], joints[t], sl, geo)
                    got_p = obs[b["campose_key"]][t].numpy().astype(np.float64)
                    check(f"{name} {b['campose_key']} ({s}) t={t}",
                          np.allclose(got_p, exp_p, atol=1e-6),
                          f"got {got_p.round(4)} exp {np.asarray(exp_p).round(4)}")
                    if b["kind"] == "image" and s == "extrinsic":
                        # the 2D tag (FK at load) == the 3D zarr's stored extrinsic
                        check(f"{name} 2D FK == stored extrinsic cam{c}",
                              np.allclose(got_p, pose9_from_matrix(T_of(c, t)), atol=1e-4))

        check(f"{name} action", torch.equal(
            out["action"], torch.from_numpy(np.asarray(s3d["action"], dtype=np.float32))))
        check(f"{name} agent_pos", np.array_equal(obs["agent_pos"].numpy(),
                                                  agent.astype(np.float32)))

    val = ds.get_validation_dataset()
    check(f"{name} val windows aligned",
          np.array_equal(val.ds3d.sampler.indices, val.ds2d.sampler.indices) and len(val) > 0)
    val[0]
    del ds, val
    gc.collect()
    return "ok", None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=list(TASKS), default=None)
    ap.add_argument("--samples", type=int, default=4)
    args = ap.parse_args()

    z3 = zarr.open("R3D/data/real_bimanual_mixed150.zarr", "r")
    z2 = zarr.open("R3D/data/real_bimanual_2d_mixed150.zarr", "r")
    geo, _ = load_preprocess_geometry("R3D/r3d/config/preprocess/real_bimanual_2d.yaml")

    for name, path in TASKS.items():
        if args.only and name != args.only:
            continue
        print(f"\n[{name}] {path}")
        for s3, s2 in itertools.product(SOURCES, SOURCES):
            status, msg = verify_pair(f"{name}({s3},{s2})", path, s3, s2,
                                      args.samples, z3, z2, geo)
            # Written out per task rather than derived, so this is an independent
            # statement of the rule, not a second copy of hybrid_layout:
            #   fused block + pose_source_3d != none          -> raise
            #   'proprio' somewhere but no block has a pose   -> raise
            #     H1  : static top is 3D, wrists 2D   -> (proprio, none)
            #     H2-U: static top is 2D, wrists 3D   -> (none, proprio)
            #     H2-F: static top is 2D, fused wrists-> (none, proprio)
            should_raise = {
                "h1": (s3, s2) == ("proprio", "none"),
                "h2u": (s3, s2) == ("none", "proprio"),
                "h2f": s3 != "none" or s2 == "proprio",
            }[name]
            if status == "raised":
                check(f"{name}({s3},{s2}) raised only when degenerate", should_raise, msg)
                print(f"  ({s3:9s},{s2:9s}) raises as designed" if should_raise
                      else f"  ({s3:9s},{s2:9s}) RAISED UNEXPECTEDLY: {msg}")
            else:
                check(f"{name}({s3},{s2}) should have raised", not should_raise)
                print(f"  ({s3:9s},{s2:9s}) ok")

    print(f"\n{n_checks[0]} checks run")
    print(f"{'ALL CHECKS PASSED' if not fails else f'{len(fails)} FAILED'}")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
