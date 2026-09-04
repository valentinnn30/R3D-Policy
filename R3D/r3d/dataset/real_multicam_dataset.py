"""Dataset for real bimanual data recorded with several fused cameras.

Unlike `RobotwinDataset`, the zarr holds one cloud *per camera* in that
camera's own frame (`point_cloud_cam0`, `point_cloud_cam1`, ...) together with
the nominal extrinsics from the ChArUco hand-eye solve. Fusion happens here, at
sample time, under a random per-camera perturbation of those extrinsics.

That is the point of the whole arrangement: a miscalibrated camera in a fused
rig produces *misregistration* -- the clouds disagree with each other -- and no
transform applied to an already-fused cloud can reproduce that. Perturbing each
camera independently can.

The perturbation is drawn once per sampled window and held constant across it.
A real calibration error is static; redrawing it per frame would present the
policy with temporal flicker that never occurs on the robot, and it would learn
to average it out. Across windows it is redrawn freely -- windows are
independent training examples, so a delta differing between them is not
observable as temporal noise.

Only the first `n_obs_steps` frames of each window are fused: dp3.py:335 slices
observations to that many, so the rest would be built and discarded.

Actions are untouched by the perturbation. The clouds live in the world frame
and the perturbation models camera-calibration error, so the scene moves
relative to the cameras while the robot does not move at all.
"""

import copy
from typing import Dict

import numpy as np
import torch
import zarr
from termcolor import cprint

from r3d.common.pytorch_util import dict_apply
from r3d.common.real_preprocess import (
    PointCloudPreprocessConfig,
    fuse_cameras,
    pose9_from_matrix,
    pose9_from_pos_quat,
    random_se3,
    select_camera_frame,
    workspace_limits,
)
from r3d.common.replay_buffer import ReplayBuffer
from r3d.common.sampler import SequenceSampler, downsample_mask, get_val_mask
from r3d.dataset.base_dataset import BaseDataset
from r3d.dataset.robotwin_dataset import add_noise, apply_color_jitter
from r3d.model.common.normalizer import LinearNormalizer, SingleFieldLinearNormalizer


def _camera_keys(zarr_path):
    """Discover point_cloud_cam* in the zarr, ordered by camera index."""
    root = zarr.open(zarr_path, mode="r")
    keys = [k for k in root["data"].array_keys() if k.startswith("point_cloud_cam")]
    if not keys:
        raise KeyError(
            f"{zarr_path} has no point_cloud_cam* arrays. It looks like a fused "
            "zarr -- use r3d.dataset.robotwin_dataset.RobotwinDataset for that."
        )
    return sorted(keys, key=lambda k: int(k.replace("point_cloud_cam", "")))


def _nominal_extrinsics(zarr_path, cam_keys):
    """Nominal camera->base per camera: a fixed 4x4, or one 4x4 per frame.

    A wrist camera's base-frame extrinsic is not constant --
    ``T_base<-cam(t) = T_base<-flange(t) @ T_flange<-cam`` -- so the converter
    writes it as a (N_frames, 4, 4) array under ``data/`` for those cameras and
    a single matrix under ``meta/`` for statically mounted ones.

    Returns (statics, per_frame_keys): ``statics[i]`` is the fixed matrix or
    None, and ``per_frame_keys[i]`` is the zarr key or None. Exactly one of the
    two is set for each camera.
    """
    root = zarr.open(zarr_path, mode="r")
    statics, per_frame_keys = [], []
    for k in cam_keys:
        name = "extrinsics_" + k.replace("point_cloud_", "")
        if f"data/{name}" in root:
            statics.append(None)
            per_frame_keys.append(name)
        elif f"meta/{name}" in root:
            statics.append(np.asarray(root[f"meta/{name}"][...], dtype=np.float64))
            per_frame_keys.append(None)
        else:
            raise KeyError(
                f"{zarr_path} has neither data/{name} nor meta/{name}. The "
                "converter writes the nominal extrinsics next to the clouds; "
                "fusion cannot be reproduced without them."
            )
    return statics, per_frame_keys


class RealMultiCamDataset(BaseDataset):
    # The unfused subclass reuses this __init__ wholesale but fuses nothing, so
    # the fusion-specific banner lines below would misdescribe what is running
    # -- and a training log claiming "Fusion downsample" is exactly the kind of
    # thing that makes someone conclude the wrong pipeline is active.
    FUSES = True

    def __init__(self,
            zarr_path,
            preprocess_config,
            horizon=1,
            pad_before=0,
            pad_after=0,
            seed=42,
            val_ratio=0.0,
            max_train_episodes=None,
            # extrinsics randomization
            use_extrinsics_randomization=True,
            extrinsics_rot_sigma_deg=0.7,
            extrinsics_trans_sigma_m=0.004,
            anchor_first_camera=False,
            fuse_method="random",
            fuse_device="cuda",
            # Fuse only the first `n_obs_steps` frames of each window instead of
            # all `horizon` of them. dp3.py:335 slices observations to
            # n_obs_steps, so the rest are built and thrown away -- with horizon
            # 16 and n_obs_steps 2 that is 8x the fusion work for nothing.
            # ONLY VALID WITH policy.obs_as_global_cond: true (the default).
            n_obs_steps=None,
            # per-point augmentation, as in RobotwinDataset
            use_data_augmentation=False,
            pc_xyz_noise_std=0.002,
            pc_rgb_noise_std=0.01,
            agent_pos_noise_std=0.0002,
            use_color_jitter=False,
            brightness_range=(-0.125, 0.125),
            contrast_range=(0.5, 1.5),
            saturation_range=(0.5, 1.5),
            use_target_ee=False,
            ):
        super().__init__()

        self.pc_cfg = PointCloudPreprocessConfig.from_yaml(preprocess_config)
        self.cam_keys = _camera_keys(zarr_path)
        self.static_extrinsics, self.per_frame_extrinsics = _nominal_extrinsics(
            zarr_path, self.cam_keys)
        self.n_obs_steps = n_obs_steps

        self.use_extrinsics_randomization = use_extrinsics_randomization
        self.extrinsics_rot_sigma_deg = extrinsics_rot_sigma_deg
        self.extrinsics_trans_sigma_m = extrinsics_trans_sigma_m
        self.anchor_first_camera = anchor_first_camera
        self.fuse_method = fuse_method
        self.fuse_device = fuse_device

        self.use_target_ee = use_target_ee
        self.use_data_augmentation = use_data_augmentation
        self.pc_xyz_noise_std = pc_xyz_noise_std
        self.pc_rgb_noise_std = pc_rgb_noise_std
        self.agent_pos_noise_std = agent_pos_noise_std
        self.use_color_jitter = use_color_jitter
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range
        self.saturation_range = saturation_range

        cprint("--------------------------", "cyan")
        cprint(f"Multi-camera: {len(self.cam_keys)} cameras {self.cam_keys}"
               + (" (FUSED at sample time)" if self.FUSES else ""), "cyan")
        if use_extrinsics_randomization:
            cprint(
                f"Extrinsics randomization: rot_sigma={extrinsics_rot_sigma_deg} deg, "
                f"trans_sigma={extrinsics_trans_sigma_m * 1000:.1f} mm, "
                f"anchor_first={anchor_first_camera}",
                "cyan")
        else:
            cprint("Extrinsics randomization DISABLED (nominal fusion)", "yellow")
        if anchor_first_camera and use_extrinsics_randomization \
                and self.per_frame_extrinsics[0] is not None:
            cprint("anchor_first_camera pins camera 0, but camera 0 has per-frame "
                   "extrinsics, i.e. it is WRIST mounted. Anchoring is meant to "
                   "pin the STATIC camera so the fused cloud's frame stays tied "
                   "to the robot base -- check the camera order in the config.",
                   "red")
        mounts = ", ".join(
            f"{k}={'per-frame' if pf else 'static'}"
            for k, pf in zip(self.cam_keys, self.per_frame_extrinsics))
        cprint(f"Nominal extrinsics: {mounts}", "cyan")
        if self.FUSES:
            cprint(f"Fusion downsample: {fuse_method} -> "
                   f"{self.pc_cfg.num_points} points", "cyan")
        if n_obs_steps is not None:
            cprint(f"Building only the first {n_obs_steps} frames per window "
                   "(requires policy.obs_as_global_cond: true)", "cyan")
        cprint("--------------------------", "cyan")

        keys = ["state", "action"] + self.cam_keys
        keys += [k for k in self.per_frame_extrinsics if k]
        if self.use_target_ee:
            keys.append("target_ee")
        self.replay_buffer = ReplayBuffer.copy_from_path(zarr_path, keys=keys)

        # Stop the sampler copying observation frames the policy will discard.
        # At 3 cameras x 4096 points that is ~4.7 MB per sample sliced out of the
        # replay buffer, of which we keep ~0.6 MB. See `key_first_k` in
        # SequenceSampler, and the gotcha in CLAUDE.md before touching this.
        self._key_first_k = {}
        if n_obs_steps is not None:
            obs_keys = self.cam_keys + [k for k in self.per_frame_extrinsics if k]
            # n_obs_steps alone is provably sufficient (see CLAUDE.md); the
            # pad_before margin is slack against future changes to the padding.
            self._key_first_k = {k: n_obs_steps + pad_before for k in obs_keys}
        # key_first_k NaN-fills the region it did not load, as a tripwire. That
        # NaN cannot reach the output -- crop_to_workspace drops it, since every
        # comparison against NaN is False -- so it surfaces as an empty crop,
        # which fuse_cameras diagnoses explicitly.

        val_mask = get_val_mask(
            n_episodes=self.replay_buffer.n_episodes, val_ratio=val_ratio, seed=seed)
        train_mask = downsample_mask(
            mask=~val_mask, max_n=max_train_episodes, seed=seed)
        self.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=horizon,
            pad_before=pad_before,
            pad_after=pad_after,
            key_first_k=self._key_first_k,
            episode_mask=train_mask)
        self.train_mask = train_mask
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=self.horizon,
            pad_before=self.pad_before,
            pad_after=self.pad_after,
            key_first_k=self._key_first_k,
            episode_mask=~self.train_mask)
        val_set.train_mask = ~self.train_mask
        # Validate against the real calibration: otherwise val loss moves with
        # the perturbation draw and stops being comparable between epochs.
        val_set.use_extrinsics_randomization = False
        return val_set

    def get_normalizer(self, mode="limits", **kwargs):
        """point_cloud limits come from the workspace box, not from the data.

        Every cloud `fuse_cameras` emits has been cropped to that box and has
        rgb in [0, 1], so the limits are known analytically. Fitting them to the
        recorded data instead would make normalization depend on which scenes
        happened to be collected, and inference -- which never sees this zarr --
        could not reproduce it.
        """
        if self.use_target_ee:
            action = np.concatenate(
                [self.replay_buffer["action"], self.replay_buffer["target_ee"]], axis=-1)
        else:
            action = self.replay_buffer["action"]

        normalizer = LinearNormalizer()
        normalizer.fit(
            data={"action": action, "agent_pos": self.replay_buffer["state"]},
            last_n_dims=1, mode=mode, **kwargs)

        lo, hi = workspace_limits(self.pc_cfg)
        scale = 2.0 / np.maximum(hi - lo, 1e-8)
        offset = -1.0 - scale * lo
        normalizer["point_cloud"] = SingleFieldLinearNormalizer.create_manual(
            scale=torch.from_numpy(scale),
            offset=torch.from_numpy(offset.astype(np.float32)),
            input_stats_dict={
                "min": torch.from_numpy(lo),
                "max": torch.from_numpy(hi),
                "mean": torch.from_numpy(((lo + hi) / 2).astype(np.float32)),
                "std": torch.from_numpy(((hi - lo) / 4).astype(np.float32)),
            })
        return normalizer

    def __len__(self) -> int:
        return len(self.sampler)

    def _sample_deltas(self, rng):
        """One perturbation per camera, drawn once for the whole window.

        These are applied on the RIGHT of the nominal transform, i.e. in the
        camera's own frame. For a wrist camera that is exactly the hand-eye
        error being modelled:

            T_base<-flange(t) @ (T_flange<-cam @ delta) = T_nominal(t) @ delta

        so a constant delta stays constant even though the arm -- and therefore
        the nominal -- moves during the window.
        """
        deltas = []
        for i in range(len(self.cam_keys)):
            anchored = (i == 0 and self.anchor_first_camera)
            if self.use_extrinsics_randomization and not anchored:
                deltas.append(random_se3(
                    self.extrinsics_rot_sigma_deg,
                    self.extrinsics_trans_sigma_m,
                    rng=rng))
            else:
                deltas.append(np.eye(4))
        return deltas

    def _nominal_at(self, sample, i, t):
        key = self.per_frame_extrinsics[i]
        if key is None:
            return self.static_extrinsics[i]
        return np.asarray(sample[key][t], dtype=np.float64)

    def _sample_to_data(self, sample, rng):
        joint_action = sample["action"].astype(np.float32)
        if self.use_target_ee:
            action = np.concatenate(
                [joint_action, sample["target_ee"].astype(np.float32)], axis=-1)
        else:
            action = joint_action

        # Observations beyond n_obs_steps are discarded by the policy, so do not
        # pay to fuse them. Actions keep the full horizon -- they are what the
        # diffusion head predicts.
        n_steps = sample[self.cam_keys[0]].shape[0]
        if self.n_obs_steps is not None:
            n_steps = min(n_steps, self.n_obs_steps)
        agent_pos = sample["state"][:n_steps].astype(np.float32)

        deltas = self._sample_deltas(rng)
        point_cloud = np.stack([
            fuse_cameras(
                [sample[k][t] for k in self.cam_keys],
                [self._nominal_at(sample, i, t) @ deltas[i]
                 for i in range(len(self.cam_keys))],
                self.pc_cfg,
                method=self.fuse_method,
                rng=rng,
                device=self.fuse_device)
            for t in range(n_steps)
        ]).astype(np.float32)

        return {
            "obs": {"point_cloud": point_cloud, "agent_pos": agent_pos},
            "action": action,
        }

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # Seeded off the global numpy state, which the DataLoader already gives
        # a distinct value per worker per epoch.
        rng = np.random.default_rng(np.random.randint(0, 2 ** 31 - 1))

        sample = self.sampler.sample_sequence(idx)
        data = self._sample_to_data(sample, rng)

        if self.use_data_augmentation:
            pc = data["obs"]["point_cloud"]
            xyz = add_noise(pc[..., :3], self.pc_xyz_noise_std, 2 * self.pc_xyz_noise_std)
            rgb = add_noise(pc[..., 3:], self.pc_rgb_noise_std, 2 * self.pc_rgb_noise_std)
            data["obs"]["point_cloud"] = np.concatenate([xyz, rgb], axis=-1)
            # ONE draw per window, broadcast across the observation frames --
            # NOT independent per frame, which is what add_noise() does.
            #
            # agent_pos is (n_obs_steps, 48) and the pair is the ONLY velocity
            # signal the policy has. Independent noise lands on the difference
            # too: at std 2 mm that is sqrt(2)*2 = 2.83 mm of noise per axis
            # against a measured median inter-frame EE motion of 1.31 mm, i.e. a
            # velocity signal-to-noise of 0.46 -- the cue was more noise than
            # signal. Correlated noise cancels exactly in the difference, so
            # velocity survives untouched while absolute pose keeps the same
            # uncertainty that stops the policy copying state into action.
            #
            # Clipping matches add_noise(): +-2*std.
            ap = data["obs"]["agent_pos"]
            jitter = np.clip(
                np.random.normal(0.0, self.agent_pos_noise_std, ap.shape[-1]),
                -2 * self.agent_pos_noise_std, 2 * self.agent_pos_noise_std)
            data["obs"]["agent_pos"] = ap + jitter

        if self.use_color_jitter:
            pc = data["obs"]["point_cloud"]
            rgb = apply_color_jitter(
                pc[..., 3:], self.brightness_range,
                self.contrast_range, self.saturation_range)
            data["obs"]["point_cloud"] = np.concatenate([pc[..., :3], rgb], axis=-1)

        data["obs"]["point_cloud"] = data["obs"]["point_cloud"].astype(np.float32)
        data["obs"]["agent_pos"] = data["obs"]["agent_pos"].astype(np.float32)
        data["action"] = data["action"].astype(np.float32)
        return dict_apply(data, torch.from_numpy)


# ---------------------------------------------------------------------------
# Unfused variant -- clouds stay in camera frame, merged in FEATURE space
# ---------------------------------------------------------------------------

# One shared length unit, in metres, for BOTH the clouds and the camera pose
# tags. Not a per-camera box and not per-axis: a single isotropic scalar.
#
# Why any scaling at all, when metres would be the honest unit: the encoder's
# `PositionEmbeddingRandom` (pointnet_extractor.py) hard-rejects group centres
# outside [-1, 1] -- its random-Fourier frequency scale assumes that range --
# and the top camera's depth reaches 1.215 m. 1.3 puts the widest observed
# coordinate (1.2664 m) at 0.974.
#
# Why this does NOT reintroduce the problem a per-camera box had: clouds and
# `campose_*` translation are divided by the SAME constant, so
#
#     p_world / R = R(campose) @ (p_cam / R) + t(campose) / R
#
# holds exactly. The composition stays a pure rigid transform with no scale
# term for the model to learn, geometry stays isotropic (a cube stays a cube),
# and all three cameras share one map, which is what lets them share encoder
# weights. The unit is simply 1.3 m rather than 1 m.
METRIC_SCALE_M = 1.3


class RealMultiCamUnfusedDataset(RealMultiCamDataset):
    """Per-camera clouds in CAMERA frame; no fusion anywhere.

    The fused parent hands the policy one world-frame cloud, which means an
    extrinsic error physically deforms the encoder's input. Here the extrinsic
    is used ONLY to decide which points fall inside the workspace; the
    coordinates that reach the encoder are the camera's own and do not depend
    on the calibration at all. Merging happens later, in feature space, by
    concatenating each camera's tokens into the attention conditioning.

    `pose_source` decides what each camera's tokens are tagged with, and it
    lives here rather than in the encoder because both candidate sources are
    already numpy at this point:

    - ``extrinsic`` (Run B) -- the calibrated ``T_world<-cam(t)``, perturbed
      exactly as the fused run perturbs it.
    - ``proprio`` (Ablation A') -- the recorded EE pose of the arm that camera
      rides on, straight out of ``agent_pos``. No calibration is read at all.
      A statically-mounted camera has no arm and gets a zero pose vector plus
      its identity embedding, which is sufficient: its transform is a constant,
      and a constant carries no information a network has to be told.
    - ``none`` -- zero pose for every camera, identity embeddings only.

    Note what is NOT here: any attempt to correct the extrinsic error. The
    flex residual is not a function of any observable, so no function of the
    input can recover it; `use_extrinsics_randomization` still carries that
    load, exactly as in the fused run.
    """

    FUSES = False
    POSE_SOURCES = ("extrinsic", "proprio", "none")

    def __init__(self, *args,
                 points_per_camera_encoder=None,
                 num_groups_per_camera=None,
                 pose_source="extrinsic",
                 camera_arm_slice=None,
                 **kwargs):
        super().__init__(*args, **kwargs)

        n_cams = len(self.cam_keys)
        if pose_source not in self.POSE_SOURCES:
            raise ValueError(
                f"pose_source must be one of {self.POSE_SOURCES}, got {pose_source!r}")
        self.pose_source = pose_source

        # Budgets default to an even split of the fused run's totals, so a
        # config that sets neither still matches run A on both axes.
        if points_per_camera_encoder is None:
            points_per_camera_encoder = [self.pc_cfg.num_points // n_cams] * n_cams
        if num_groups_per_camera is None:
            num_groups_per_camera = [512 // n_cams] * n_cams
        for name, v in (("points_per_camera_encoder", points_per_camera_encoder),
                        ("num_groups_per_camera", num_groups_per_camera)):
            if len(v) != n_cams:
                raise ValueError(
                    f"{name} has {len(v)} entries for {n_cams} cameras; it needs "
                    "one per camera, in point_cloud_cam* order")
        self.points_per_camera_encoder = [int(v) for v in points_per_camera_encoder]
        self.num_groups_per_camera = [int(v) for v in num_groups_per_camera]

        for i, (p, g) in enumerate(zip(self.points_per_camera_encoder,
                                       self.num_groups_per_camera)):
            if g > p:
                raise ValueError(
                    f"camera {i}: num_groups {g} exceeds points {p}. The encoder "
                    "picks that many FPS centres out of the cloud, so it cannot "
                    "have more groups than points.")

        # Which slice of `agent_pos` holds the EE pose of the arm each camera
        # rides on: [x, y, z, qx, qy, qz, qw]. None for a static camera.
        if camera_arm_slice is None:
            camera_arm_slice = [None] * n_cams
        if len(camera_arm_slice) != n_cams:
            raise ValueError(
                f"camera_arm_slice has {len(camera_arm_slice)} entries for "
                f"{n_cams} cameras")
        self.camera_arm_slice = [
            None if s is None else (int(s[0]), int(s[1])) for s in camera_arm_slice]
        if pose_source == "proprio" and all(s is None for s in self.camera_arm_slice):
            raise ValueError(
                "pose_source='proprio' but no camera has a camera_arm_slice, so "
                "every camera would get a zero pose. Set the slice for the "
                "wrist-mounted cameras (left [0, 7], right [7, 14]).")

        self.extrinsics_keys = [
            "extrinsics_" + k.replace("point_cloud_", "") for k in self.cam_keys]
        self.campose_keys = [
            "campose_" + k.replace("point_cloud_", "") for k in self.cam_keys]

        cprint("--------------------------", "cyan")
        cprint("UNFUSED multi-camera: clouds stay in camera frame, "
               "merged in feature space", "cyan")
        cprint(f"  pose_source      : {self.pose_source}", "cyan")
        cprint(f"  points per camera: {self.points_per_camera_encoder} "
               f"(total {sum(self.points_per_camera_encoder)})", "cyan")
        cprint(f"  groups per camera: {self.num_groups_per_camera} "
               f"(total {sum(self.num_groups_per_camera)} tokens)", "cyan")
        cprint(f"  arm slices       : {self.camera_arm_slice}", "cyan")
        cprint("--------------------------", "cyan")

    # -- pose tag ----------------------------------------------------------

    def _campose(self, i, T, agent_pos_t):
        if self.pose_source == "none":
            return np.zeros(9, dtype=np.float64)
        if self.pose_source == "extrinsic":
            return pose9_from_matrix(T)
        sl = self.camera_arm_slice[i]
        if sl is None:
            # Static camera under A': no arm to read. Its world pose is a
            # constant, which the identity embedding already stands in for.
            return np.zeros(9, dtype=np.float64)
        a, b = sl
        arm = np.asarray(agent_pos_t[a:b], dtype=np.float64)
        return pose9_from_pos_quat(arm[:3], arm[3:7])

    # -- sample ------------------------------------------------------------

    def _sample_to_data(self, sample, rng):
        joint_action = sample["action"].astype(np.float32)
        if self.use_target_ee:
            action = np.concatenate(
                [joint_action, sample["target_ee"].astype(np.float32)], axis=-1)
        else:
            action = joint_action

        n_steps = sample[self.cam_keys[0]].shape[0]
        if self.n_obs_steps is not None:
            n_steps = min(n_steps, self.n_obs_steps)
        agent_pos = sample["state"][:n_steps].astype(np.float32)

        deltas = self._sample_deltas(rng)
        obs = {"agent_pos": agent_pos}
        cam_valid = np.zeros((n_steps, len(self.cam_keys)), dtype=np.float32)

        for i, key in enumerate(self.cam_keys):
            clouds, poses = [], []
            for t in range(n_steps):
                T = self._nominal_at(sample, i, t) @ deltas[i]
                pts, ok = select_camera_frame(
                    sample[key][t], T, self.pc_cfg,
                    self.points_per_camera_encoder[i],
                    self.num_groups_per_camera[i],
                    rng=rng)
                clouds.append(pts)
                poses.append(self._campose(i, T, agent_pos[t]))
                cam_valid[t, i] = float(ok)
            obs[key] = np.stack(clouds).astype(np.float32)
            obs[self.campose_keys[i]] = np.stack(poses).astype(np.float32)

        obs["cam_valid"] = cam_valid
        return {"obs": obs, "action": action}

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        rng = np.random.default_rng(np.random.randint(0, 2 ** 31 - 1))
        sample = self.sampler.sample_sequence(idx)
        data = self._sample_to_data(sample, rng)

        if self.use_data_augmentation:
            # Per camera, matching the fused path's per-point treatment. The
            # clouds are separate arrays now but the augmentation is the same.
            for key in self.cam_keys:
                pc = data["obs"][key]
                xyz = add_noise(pc[..., :3], self.pc_xyz_noise_std,
                                2 * self.pc_xyz_noise_std)
                rgb = add_noise(pc[..., 3:], self.pc_rgb_noise_std,
                                2 * self.pc_rgb_noise_std)
                data["obs"][key] = np.concatenate([xyz, rgb], axis=-1)

            # ONE draw per window, broadcast across observation frames -- see
            # the parent for why independent per-frame noise destroys the
            # velocity cue.
            ap = data["obs"]["agent_pos"]
            jitter = np.clip(
                np.random.normal(0.0, self.agent_pos_noise_std, ap.shape[-1]),
                -2 * self.agent_pos_noise_std, 2 * self.agent_pos_noise_std)
            data["obs"]["agent_pos"] = ap + jitter

        if self.use_color_jitter:
            for key in self.cam_keys:
                pc = data["obs"][key]
                rgb = apply_color_jitter(
                    pc[..., 3:], self.brightness_range,
                    self.contrast_range, self.saturation_range)
                data["obs"][key] = np.concatenate([pc[..., :3], rgb], axis=-1)

        for key in list(data["obs"]):
            data["obs"][key] = data["obs"][key].astype(np.float32)
        data["action"] = data["action"].astype(np.float32)
        return dict_apply(data, torch.from_numpy)

    # -- normalizer --------------------------------------------------------

    def get_normalizer(self, mode="limits", **kwargs):
        """xyz gets ONE shared isotropic length unit; only rgb is remapped.

        The fused parent maps `point_cloud` xyz onto [-1, 1] using the
        base-frame workspace box. That box is meaningless here (a camera-frame
        cloud does not live in it), and every substitute is worse than doing
        nothing:

        * A per-camera box is anisotropic AND differs between cameras -- 1.7x
          to 3.1x, disagreeing by 1.37x on y. The same physical cube would then
          produce a differently-shaped point pattern depending on which camera
          saw it, through ONE set of shared Uni3D weights. `KNNGrouper` only
          centres each patch (`radius` is never set), so that distortion
          reaches patch shape unopposed.
        * Any scale factor at all puts the cloud in different units from
          `campose_*`, which is metric. Composing a camera-frame point with its
          pose tag would then need a learned scale correction on top of the
          rigid transform.

        Literal metres would be the honest unit and costs nothing
        numerically, but `PositionEmbeddingRandom` hard-rejects group centres
        outside [-1, 1] and the top camera reaches 1.215 m. So both the clouds
        and the campose translation are divided by the same
        `METRIC_SCALE_M`, which keeps
        ``p/R = R(campose) @ (p_cam/R) + t(campose)/R`` exact -- a pure rigid
        composition with no scale term left for the model to learn.

        rgb is still mapped [0, 1] -> [-1, 1] exactly as the fused run does, so
        colour is comparable between the two experiments.

        `agent_pos` and `action` normalization is left EXACTLY as the fused run
        has it. It is anisotropic and per-arm ([6.64, 4.91, 7.43] left,
        [7.48, 4.59, 6.65] right), but it is fixed and global -- one constant
        map the network learns once, as it already does in run A. Changing it
        would confound the comparison for no geometric gain.
        """
        if self.use_target_ee:
            action = np.concatenate(
                [self.replay_buffer["action"], self.replay_buffer["target_ee"]], axis=-1)
        else:
            action = self.replay_buffer["action"]

        normalizer = LinearNormalizer()
        normalizer.fit(
            data={"action": action, "agent_pos": self.replay_buffer["state"]},
            last_n_dims=1, mode=mode, **kwargs)

        # xyz: one shared isotropic length unit. rgb: [0,1] -> [-1,1].
        k = 1.0 / METRIC_SCALE_M
        scale = np.array([k, k, k, 2., 2., 2.], dtype=np.float32)
        offset = np.array([0., 0., 0., -1., -1., -1.], dtype=np.float32)
        stats_lo = np.array([-1., -1., -1., 0., 0., 0.], dtype=np.float32)
        stats_hi = np.array([1., 1., 1., 1., 1., 1.], dtype=np.float32)
        for key in self.cam_keys:
            normalizer[key] = SingleFieldLinearNormalizer.create_manual(
                scale=torch.from_numpy(scale),
                offset=torch.from_numpy(offset),
                input_stats_dict={
                    "min": torch.from_numpy(stats_lo),
                    "max": torch.from_numpy(stats_hi),
                    "mean": torch.from_numpy((stats_lo + stats_hi) / 2),
                    "std": torch.from_numpy((stats_hi - stats_lo) / 4),
                })

        # campose: translation shares the clouds' length unit -- that shared
        # factor is what keeps `p/R = R(pose) @ (p_cam/R) + t(pose)/R` exact.
        # The 6D rotation is already unit-scale and is left alone.
        pose_scale = np.array([k, k, k, 1., 1., 1., 1., 1., 1.], dtype=np.float32)
        pose_zero = np.zeros(9, dtype=np.float32)
        for key in self.campose_keys:
            normalizer[key] = SingleFieldLinearNormalizer.create_manual(
                scale=torch.from_numpy(pose_scale),
                offset=torch.from_numpy(pose_zero),
                input_stats_dict={
                    "min": torch.from_numpy(-np.ones(9, dtype=np.float32)),
                    "max": torch.from_numpy(np.ones(9, dtype=np.float32)),
                    "mean": torch.from_numpy(pose_zero),
                    "std": torch.from_numpy(np.ones(9, dtype=np.float32)),
                })

        # create_identity() makes shape-(1,) scale/offset that broadcast over
        # any width, so this needs no per-key dimension.
        normalizer["cam_valid"] = SingleFieldLinearNormalizer.create_identity()
        return normalizer
