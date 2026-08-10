"""Validate a converted zarr against what R3D actually expects, and emit the
matching task config.

Run this before every training launch. Most of the ways a real-data zarr goes
wrong are silent: RGB at 0-255 survives the dataloader and is only destroyed
later by the colour-jitter augmentation, a per-episode `episode_ends` just
produces nonsense sequences, and extrinsics that do not quite land the clouds
in the workspace box leave the crop starved.

    python scripts/inspect_zarr.py R3D/data/real_bimanual.zarr
    python scripts/inspect_zarr.py R3D/data/real_bimanual.zarr --write-task real_bimanual
"""

import argparse
import os
import sys

import numpy as np
import zarr
from termcolor import cprint

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "R3D"))

from r3d.common.real_preprocess import PointCloudPreprocessConfig  # noqa: E402

MULTICAM_TASK_TEMPLATE = """name: ${{task_name}}

shape_meta: &shape_meta
  obs:
    point_cloud:
      shape: [{n_points}, 6]   # the FUSED cloud; per-camera clouds are {points_per_cam}
      type: point_cloud
    agent_pos:
      shape: [{d_state}]
      type: low_dim
  action:
    shape: [{d_action}]  # {action_note}

# No simulator in this checkout -- train.py sets this to None and every use
# site guards on it, so we get train/val loss only and no rollout metrics.
env_runner: null

dataset:
  _target_: r3d.dataset.real_multicam_dataset.RealMultiCamDataset
  # train.py chdirs to the repo root, so these are relative to ~/R3D-Policy
  zarr_path: {zarr_path}
  preprocess_config: {preprocess_config}
  horizon: ${{horizon}}
  pad_before: ${{eval:'${{n_obs_steps}}-1'}}
  pad_after: ${{eval:'${{n_action_steps}}-1'}}
  seed: 42
  val_ratio: {val_ratio}
  max_train_episodes: null

  # Extrinsics randomization: per camera, redrawn per sampled window, constant
  # across the window. Validation always fuses at the nominal calibration.
  use_extrinsics_randomization: {rand_enabled}
  extrinsics_rot_sigma_deg: {rot_sigma}
  extrinsics_trans_sigma_m: {trans_sigma}
  anchor_first_camera: {anchor}
  fuse_method: {fuse_method}
  fuse_device: cpu   # runs in dataloader workers; 'random' does not need a GPU

  # Fuse (and load) only the frames the policy actually encodes: dp3.py:335
  # slices observations to n_obs_steps, so the other `horizon - n_obs_steps`
  # are built and thrown away. This also feeds SequenceSampler's `key_first_k`,
  # which stops them being copied out of the replay buffer at all.
  # `dataset_obs_steps` is declared in every policy config and was the intended
  # plumbing for exactly this; it had no consumer until now.
  # Requires policy.obs_as_global_cond: true, which is the default.
  n_obs_steps: ${{dataset_obs_steps}}

  use_data_augmentation: ${{data_augmentation.use_augmentation}}
  pc_xyz_noise_std: ${{data_augmentation.pc_xyz_noise_std}}
  pc_rgb_noise_std: ${{data_augmentation.pc_rgb_noise_std}}
  agent_pos_noise_std: ${{data_augmentation.agent_pos_noise_std}}
  use_color_jitter: ${{data_augmentation.use_color_jitter}}
  brightness_range: ${{data_augmentation.brightness_range}}
  contrast_range: ${{data_augmentation.contrast_range}}
  saturation_range: ${{data_augmentation.saturation_range}}
  use_target_ee: ${{policy.use_target_ee}}
"""

FUSED_TASK_TEMPLATE = """name: ${{task_name}}

shape_meta: &shape_meta
  obs:
    point_cloud:
      shape: [{n_points}, 6]
      type: point_cloud
    agent_pos:
      shape: [{d_state}]
      type: low_dim
  action:
    shape: [{d_action}]  # {action_note}

env_runner: null

dataset:
  _target_: r3d.dataset.robotwin_dataset.RobotwinDataset
  zarr_path: {zarr_path}
  horizon: ${{horizon}}
  pad_before: ${{eval:'${{n_obs_steps}}-1'}}
  pad_after: ${{eval:'${{n_action_steps}}-1'}}
  seed: 42
  val_ratio: {val_ratio}
  max_train_episodes: null
  use_data_augmentation: ${{data_augmentation.use_augmentation}}
  pc_xyz_noise_std: ${{data_augmentation.pc_xyz_noise_std}}
  pc_rgb_noise_std: ${{data_augmentation.pc_rgb_noise_std}}
  agent_pos_noise_std: ${{data_augmentation.agent_pos_noise_std}}
  use_color_jitter: ${{data_augmentation.use_color_jitter}}
  brightness_range: ${{data_augmentation.brightness_range}}
  contrast_range: ${{data_augmentation.contrast_range}}
  saturation_range: ${{data_augmentation.saturation_range}}
  use_target_ee: ${{policy.use_target_ee}}
"""


class Report:
    def __init__(self):
        self.errors, self.warnings = [], []

    def error(self, msg):
        self.errors.append(msg)
        cprint(f"  FAIL  {msg}", "red")

    def warn(self, msg):
        self.warnings.append(msg)
        cprint(f"  WARN  {msg}", "yellow")

    def ok(self, msg):
        cprint(f"  ok    {msg}", "green")

    def note(self, msg):
        cprint(f"  --    {msg}", "cyan")


def check_cloud_values(rep, sample, label):
    """RGB range and finiteness -- the two failures that survive silently."""
    if not np.isfinite(sample).all():
        rep.error(f"{label} contains NaN/Inf")
    rgb = sample[..., 3:]
    if rgb.max() > 1.0 + 1e-4:
        rep.error(
            f"{label} rgb max is {rgb.max():.3f} -- must be in [0, 1]. At 0-255 "
            "the colour-jitter augmentation clips every colour to white."
        )
    elif rgb.min() < -1e-4:
        rep.error(f"{label} rgb min is {rgb.min():.3f} -- must be in [0, 1]")


def report_registration(rep, clouds_by_cam, cam_keys):
    """Measure how well the cameras agree once transformed to the base frame.

    This is the direct test of whether the extrinsics are right. Where two
    cameras see the same surface, their points should coincide; a systematic
    nearest-neighbour distance is exactly the misregistration the training-time
    perturbation is meant to model. So this number does double duty: it
    validates the calibration, and it tells you what to set
    `extrinsics_randomization` to instead of guessing from the ChArUco residual.
    """
    from scipy.spatial import cKDTree

    if len(cam_keys) < 2:
        return
    cprint("  registration (nearest-neighbour distance between cameras):", "cyan")
    for i in range(len(cam_keys)):
        for j in range(i + 1, len(cam_keys)):
            a, b = clouds_by_cam[i], clouds_by_cam[j]
            if len(a) == 0 or len(b) == 0:
                continue
            # Only compare where the views actually overlap: a point of A with no
            # B point within 5 cm is surface B simply cannot see, and including
            # it would measure coverage rather than alignment.
            d, _ = cKDTree(b).query(a, k=1, distance_upper_bound=0.05)
            overlap = np.isfinite(d)
            frac = overlap.mean()
            if frac < 0.02:
                rep.warn(f"{cam_keys[i]} vs {cam_keys[j]}: only {frac * 100:.1f}% of "
                         "points have a counterpart within 5 cm -- either the views "
                         "barely overlap or an extrinsic is badly wrong")
                continue
            dd = d[overlap]
            med, p90 = np.median(dd) * 1000, np.percentile(dd, 90) * 1000
            line = (f"    {cam_keys[i]} vs {cam_keys[j]}: median {med:5.1f} mm, "
                    f"p90 {p90:5.1f} mm  ({frac * 100:.0f}% overlap)")
            if med > 20:
                rep.warn(line.strip() + "  <- >2 cm suggests a wrong extrinsic, "
                         "not calibration noise")
            else:
                cprint(line, "green")


def _compact(idx):
    """[14,15,16,...,47] -> '14-47', so a 34-dim block reads as one range."""
    if len(idx) == 0:
        return "[]"
    runs, start, prev = [], idx[0], idx[0]
    for i in idx[1:]:
        if i != prev + 1:
            runs.append((start, prev))
            start = i
        prev = i
    runs.append((start, prev))
    return "[" + ", ".join(f"{a}-{b}" if b > a else str(a) for a, b in runs) + "]"


def describe_extent(xyz):
    return "  ".join(
        f"{ax}[{lo:+.3f}, {hi:+.3f}]"
        for ax, lo, hi in zip("xyz", xyz.min(0), xyz.max(0))
    )


def check_extrinsics(rep, T, label):
    R = T[:3, :3]
    if not np.allclose(R @ R.T, np.eye(3), atol=1e-4):
        rep.error(f"{label} rotation block is not orthonormal")
    elif not np.isclose(np.linalg.det(R), 1.0, atol=1e-4):
        rep.error(f"{label} rotation has det {np.linalg.det(R):.4f}, expected +1 "
                  "(a reflection usually means a transposed matrix)")
    if np.allclose(T, np.eye(4)):
        rep.warn(f"{label} is identity -- is the ChArUco solve filled in?")


def check(zarr_path, pc_cfg, rep):
    root = zarr.open(zarr_path, mode="r")
    if "data" not in root or "meta/episode_ends" not in root:
        rep.error("not an R3D replay buffer (needs data/ and meta/episode_ends)")
        return None

    arrays = list(root["data"].array_keys())
    cam_keys = sorted([k for k in arrays if k.startswith("point_cloud_cam")],
                      key=lambda k: int(k.replace("point_cloud_cam", "")))
    multicam = len(cam_keys) > 0
    if not multicam and "point_cloud" not in arrays:
        rep.error("no point_cloud or point_cloud_cam* arrays")
        return None
    for key in ("state", "action"):
        if key not in arrays:
            rep.error(f"data/{key} is missing")
    if rep.errors:
        return None

    ref = root[f"data/{cam_keys[0]}"] if multicam else root["data/point_cloud"]
    n_frames = ref.shape[0]
    state, action = root["data/state"], root["data/action"]
    ends = np.asarray(root["meta/episode_ends"][...])

    for name, arr in [("state", state), ("action", action)]:
        if arr.shape[0] != n_frames:
            rep.error(f"{name} has {arr.shape[0]} frames, clouds have {n_frames}")

    # --- episode_ends is CUMULATIVE -----------------------------------------
    if ends.ndim != 1 or len(ends) == 0:
        rep.error(f"episode_ends must be 1-D and non-empty, got {ends.shape}")
    elif not np.all(np.diff(ends) > 0):
        rep.error("episode_ends is not strictly increasing -- it holds CUMULATIVE "
                  "frame counts, not per-episode lengths")
    elif ends[-1] != n_frames:
        rep.error(f"episode_ends[-1] = {ends[-1]} but there are {n_frames} frames")
    else:
        lengths = np.diff(np.concatenate([[0], ends]))
        rep.ok(f"{len(ends)} episodes, {n_frames} frames (len min/mean/max = "
               f"{lengths.min()}/{lengths.mean():.0f}/{lengths.max()})")
        if lengths.min() < 2:
            rep.warn(f"shortest episode is {lengths.min()} frame(s)")
        if len(ends) < 2:
            # get_val_mask caps n_val at n_episodes-1, so a single episode gives
            # an EMPTY validation set and training reports no val loss at all.
            rep.warn("only one episode -- the validation split will be empty "
                     "(get_val_mask caps n_val at n_episodes-1) and training "
                     "will report no val loss")

    n_sample = min(n_frames, 200)

    if multicam:
        rep.ok(f"{len(cam_keys)} cameras: {', '.join(cam_keys)}")
        transformed, single_frame = [], []
        for i, key in enumerate(cam_keys):
            arr = root[f"data/{key}"]
            if arr.shape[0] != n_frames:
                rep.error(f"{key} has {arr.shape[0]} frames, cam0 has {n_frames}")
            if arr.ndim != 3 or arr.shape[2] != 6:
                rep.error(f"{key} must be (N, P, 6), got {arr.shape}")
                continue
            sample = np.asarray(arr[:n_sample])
            check_cloud_values(rep, sample, key)

            # A wrist camera carries one nominal per frame (its base-frame
            # extrinsic moves with the arm); a static one carries a single matrix.
            frame_key, static_key = f"data/extrinsics_cam{i}", f"meta/extrinsics_cam{i}"
            if frame_key in root:
                full = root[frame_key]
                if full.shape != (n_frames, 4, 4):
                    rep.error(f"{frame_key} must be ({n_frames}, 4, 4), got {full.shape}")
                    continue
                T = np.asarray(full[:n_sample], dtype=np.float64)
                for j in range(min(5, len(T))):
                    check_extrinsics(rep, T[j], f"{frame_key}[{j}]")
                if np.allclose(T, T[0]):
                    rep.warn(f"{frame_key} does not change over the sampled frames "
                             "-- is this really a wrist camera?")
                rep.ok(f"{key} {arr.shape} (camera frame, per-frame extrinsics)")
                per_frame_xyz = (np.einsum("nij,npj->npi", T[:, :3, :3], sample[..., :3])
                                 + T[:, None, :3, 3])
                xyz = per_frame_xyz.reshape(-1, 3)
            elif static_key in root:
                T = np.asarray(root[static_key][...], dtype=np.float64)
                check_extrinsics(rep, T, static_key)
                rep.ok(f"{key} {arr.shape} (camera frame, static extrinsics)")
                per_frame_xyz = sample[..., :3] @ T[:3, :3].T + T[:3, 3]
                xyz = per_frame_xyz.reshape(-1, 3)
            else:
                rep.error(f"neither {frame_key} nor {static_key} is present -- "
                          "fusion cannot be reproduced without the nominal extrinsics")
                continue
            transformed.append(xyz)
            # One frame, not the pooled set: a wrist camera moves, so pooling
            # across frames unions many viewpoints and would blur the comparison.
            single_frame.append(per_frame_xyz[len(per_frame_xyz) // 2])

        if len(single_frame) > 1 and not rep.errors:
            report_registration(rep, single_frame, cam_keys)

        if transformed and not rep.errors:
            fused = np.concatenate(transformed, axis=0)
            rep.ok("fused extent (nominal)  " + describe_extent(fused))

            # The crop has to leave FPS/subsample something to work with.
            lo, hi = pc_cfg.bounds
            inside = np.all((fused > lo) & (fused < hi), axis=-1)
            per_frame = inside.sum() / n_sample
            frac = inside.mean()
            rep.ok(f"{frac * 100:.1f}% of fused points land inside the workspace "
                   f"({per_frame:.0f} per frame, need {pc_cfg.num_points})")
            if per_frame < pc_cfg.num_points:
                rep.error(
                    f"only ~{per_frame:.0f} points per frame survive the workspace "
                    f"crop but num_points is {pc_cfg.num_points}; fusion would pad "
                    "with duplicates. Widen `workspace`, raise `points_per_camera`, "
                    "or check the extrinsics."
                )
            elif frac < 0.25:
                rep.warn("under a quarter of the stored points are usable -- "
                         "`crop_margin` may be far larger than it needs to be")
        points_per_cam = root[f"data/{cam_keys[0]}"].shape[1]
        n_points = pc_cfg.num_points
    else:
        pc = root["data/point_cloud"]
        if pc.ndim != 3 or pc.shape[2] != 6:
            rep.error(f"point_cloud must be (N, P, 6) xyz+rgb, got {pc.shape}")
        else:
            rep.ok(f"point_cloud {pc.shape} {pc.dtype}")
        sample = np.asarray(pc[:n_sample])
        check_cloud_values(rep, sample, "point_cloud")
        rep.ok("xyz extent  " + describe_extent(sample[..., :3].reshape(-1, 3)))
        if np.abs(sample[..., :3]).max() > 5.0:
            rep.warn("xyz spans more than 5 m -- are the points in metres, in the base frame?")
        points_per_cam = None
        n_points = pc.shape[1]
        if n_points != pc_cfg.num_points:
            rep.warn(f"point_cloud has {n_points} points but the config's "
                     f"num_points is {pc_cfg.num_points}; shape_meta must match "
                     "the data exactly")

    # --- proprioception / actions -------------------------------------------
    st, ac = np.asarray(state[...]), np.asarray(action[...])
    if not (np.isfinite(st).all() and np.isfinite(ac).all()):
        rep.error("state/action contains NaN/Inf")
    for name, arr in [("state", st), ("action", ac)]:
        const = np.flatnonzero(arr.std(axis=0) < 1e-8)
        if len(const):
            rep.warn(f"{name} dims {const.tolist()} never change")

    # Dimensions where the "command" is bit-for-bit the measured state carry no
    # commanded/measured distinction: they are implicitly a next_state action
    # while the rest are true controller targets. Known for the hand blocks in
    # the first recording (a collection-time issue); this reports whether a new
    # dataset still has it.
    if st.shape == ac.shape:
        same = np.flatnonzero((st == ac).all(axis=0))
        if len(same):
            rep.warn(f"action == state exactly in {len(same)}/{st.shape[1]} dims "
                     f"{_compact(same)} -- no commanded signal for those; they "
                     "behave as next_state actions")
        else:
            rep.ok("action differs from state in every dim (true commanded targets)")
    rep.ok(f"state  {st.shape}  range [{st.min():.3f}, {st.max():.3f}]")
    rep.ok(f"action {ac.shape}  range [{ac.min():.3f}, {ac.max():.3f}]")

    has_ee = "target_ee" in arrays
    if has_ee:
        rep.ok(f"target_ee {root['data/target_ee'].shape} (use_target_ee is usable)")
    else:
        rep.note("no target_ee; keep policy.use_target_ee=false")

    return {
        "multicam": multicam,
        "n_points": n_points,
        "points_per_cam": points_per_cam,
        "d_state": st.shape[1],
        "d_action": ac.shape[1],
        "has_ee": has_ee,
        "d_ee": root["data/target_ee"].shape[1] if has_ee else 0,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("zarr_path")
    p.add_argument("--config", default="R3D/r3d/config/preprocess/real_bimanual.yaml",
                   help="preprocess config; its workspace box defines the crop check")
    p.add_argument("--write-task", metavar="NAME",
                   help="write R3D/r3d/config/task/NAME.yaml with these dims")
    p.add_argument("--val-ratio", type=float, default=0.02)
    args = p.parse_args()

    import yaml
    with open(args.config) as f:
        raw_cfg = yaml.safe_load(f)
    pc_cfg = PointCloudPreprocessConfig.from_yaml(args.config)

    cprint(f"\n{args.zarr_path}", "cyan")
    rep = Report()
    dims = check(args.zarr_path, pc_cfg, rep)

    if dims is None or rep.errors:
        cprint(f"\n{len(rep.errors)} error(s) -- do not train on this zarr.\n", "red")
        sys.exit(1)

    action_note = (
        f"joint only; {dims['d_action'] + dims['d_ee']} if policy.use_target_ee"
        if dims["has_ee"] else "joint only"
    )
    cprint("\nshape_meta for the task config:", "cyan")
    print(f"    point_cloud: [{dims['n_points']}, 6]")
    print(f"    agent_pos:   [{dims['d_state']}]")
    print(f"    action:      [{dims['d_action']}]")

    if args.write_task:
        rand = raw_cfg.get("extrinsics_randomization", {})
        out = os.path.join("R3D/r3d/config/task", f"{args.write_task}.yaml")
        common = dict(
            n_points=dims["n_points"], d_state=dims["d_state"],
            d_action=dims["d_action"], action_note=action_note,
            zarr_path=args.zarr_path, val_ratio=args.val_ratio)
        if dims["multicam"]:
            text = MULTICAM_TASK_TEMPLATE.format(
                points_per_cam=dims["points_per_cam"],
                preprocess_config=args.config,
                rand_enabled=str(rand.get("enabled", True)).lower(),
                rot_sigma=rand.get("rot_sigma_deg", 0.3),
                trans_sigma=rand.get("trans_sigma_m", 0.003),
                anchor=str(rand.get("anchor_first_camera", False)).lower(),
                fuse_method=raw_cfg.get("fusion", {}).get("method", "random"),
                **common)
        else:
            text = FUSED_TASK_TEMPLATE.format(**common)
        with open(out, "w") as f:
            f.write(text)
        cprint(f"\nwrote {out}", "green")

    cprint(f"\npassed with {len(rep.warnings)} warning(s).\n", "green")


if __name__ == "__main__":
    main()
