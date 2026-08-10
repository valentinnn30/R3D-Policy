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
    random_se3,
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
        cprint(f"Multi-camera fusion: {len(self.cam_keys)} cameras {self.cam_keys}", "cyan")
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
        cprint(f"Fusion downsample: {fuse_method} -> {self.pc_cfg.num_points} points", "cyan")
        if n_obs_steps is not None:
            cprint(f"Fusing only the first {n_obs_steps} frames per window "
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
            data["obs"]["agent_pos"] = add_noise(
                data["obs"]["agent_pos"], self.agent_pos_noise_std,
                2 * self.agent_pos_noise_std)

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
