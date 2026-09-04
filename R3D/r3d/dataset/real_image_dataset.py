"""RealMultiCamImageDataset -- the 2D counterpart of RealMultiCamUnfusedDataset.

Design brief: `~/ros2_ws/tmp/dinov2_2d_baseline_design.md`.

Reads the image zarr written by `scripts/convert_2d_h5_to_zarr.py` and emits the
same observation dict the unfused 3D dataset does, with `point_cloud_cam*`
replaced by `image_cam*`:

    image_cam{i}   (T, 3, H, W) float32 in [0, 1]
    campose_cam{i} (T, 9)       translation(3) + 6D rotation(6)
    cam_valid      (T, n_cams)  all ones -- a camera always produces pixels
    agent_pos      (T, 48)

Three things are deliberately NOT reimplemented, because a second
implementation is a second chance to differ from the run this is compared to:

* `state`, `action` and the quaternion canonicalisation come out of the zarr
  exactly as the 3D converter assembled them;
* `apply_color_jitter` and `add_noise` are the SAME functions the 3D dataset
  calls, with the same ranges, so the photometric and state augmentation match
  by construction rather than by two configs agreeing;
* `pose_source: proprio` reads the same `agent_pos` slice through the same
  `pose9_from_pos_quat`.

**Extrinsics are not stored per frame.** The 3D zarr carries
`data/extrinsics_cam*`, because there the transform also decides the workspace
crop and has to be reproducible frame by frame. Here the only consumer is the
pose tag, so the camera pose is composed at load time from the joint angles in
the zarr and the measured constants in the preprocess config:

    T_world<-cam(t) = T_world<-link0 @ panda_fk_flange(joints(t)) @ T_link8<-cam

which is the identical expression `convert_h5_to_zarr.py` evaluates, from the
identical config block. Keeping the mount in config rather than baking it into
the data is also what makes the kinematic ablation a one-line override.
"""

import copy
from typing import Dict

import numpy as np
import torch
import yaml
import zarr
from termcolor import cprint

from r3d.common.real_preprocess import (
    panda_fk_flange,
    pose9_from_matrix,
    pose9_from_pos_quat,
)
from r3d.common.replay_buffer import ReplayBuffer
from r3d.common.sampler import SequenceSampler, downsample_mask, get_val_mask
from r3d.dataset.base_dataset import BaseDataset
from r3d.dataset.real_multicam_dataset import METRIC_SCALE_M
from r3d.dataset.robotwin_dataset import add_noise, apply_color_jitter
from r3d.model.common.normalizer import LinearNormalizer, SingleFieldLinearNormalizer

POSE_SOURCES = ("extrinsic", "proprio", "none")


def _image_keys(zarr_path):
    root = zarr.open(zarr_path, mode="r")
    keys = [k for k in root["data"].array_keys() if k.startswith("image_cam")]
    if not keys:
        raise KeyError(
            f"{zarr_path} has no image_cam* arrays. A point-cloud zarr wants "
            "RealMultiCamUnfusedDataset instead.")
    return sorted(keys, key=lambda k: int(k.replace("image_cam", "")))


def load_preprocess_geometry(preprocess_config):
    """Per-camera mount geometry, in camera order, from the preprocess yaml.

    Follows `inherit:` so the 2D config can point at the 3D one for the measured
    extrinsics rather than restating them -- two copies of a calibration is two
    things to update and one to forget.
    """
    with open(preprocess_config) as f:
        cfg = yaml.safe_load(f)
    inherit = cfg.get("inherit")
    if inherit:
        with open(inherit) as f:
            base = yaml.safe_load(f)
        base.update({k: v for k, v in cfg.items() if k != "inherit"})
        cfg = base

    cams = []
    for cam in cfg["cameras"]:
        entry = {
            "name": cam["name"],
            "mount": cam.get("mount", "static"),
            "extrinsics": np.asarray(cam["extrinsics"], dtype=np.float64),
        }
        if entry["mount"] == "wrist":
            fp = cam["flange_pose"]
            if fp.get("source") != "fk":
                raise NotImplementedError(
                    f"camera '{cam['name']}': only flange_pose.source 'fk' is "
                    "supported here. The recorded-EE route folds in the "
                    "end-effector placement, which is exactly what the 3D "
                    "config avoids.")
            entry["base"] = np.asarray(fp["base"], dtype=np.float64)
            entry["joint_slice"] = tuple(int(v) for v in fp["joint_slice"])
        cams.append(entry)
    return cams, cfg


class RealMultiCamImageDataset(BaseDataset):

    def __init__(self,
            zarr_path,
            preprocess_config,
            horizon=1,
            pad_before=0,
            pad_after=0,
            seed=42,
            val_ratio=0.0,
            max_train_episodes=None,
            n_obs_steps=None,
            # 'extrinsic' -> the headline run, mirroring approach 2's Run B.
            # 'proprio'   -> kinematic tagging ablation.
            # 'none'      -> identity embeddings only (tag control).
            pose_source="extrinsic",
            camera_arm_slice=None,
            # Augmentation. Defaults mirror the 3D unfused task config so the
            # two runs face the same photometric and state noise.
            use_data_augmentation=False,
            agent_pos_noise_std=0.0002,
            use_color_jitter=False,
            brightness_range=(-0.125, 0.125),
            contrast_range=(0.5, 1.5),
            saturation_range=(0.5, 1.5),
            # Per-pixel Gaussian noise, the image counterpart of the 3D path's
            # `pc_rgb_noise_std`. Same constant, same units: the 3D comment
            # reads "1% of [0,1] rgb ~ 2.5/255", and images are float32 in
            # [0, 1] by the time this is applied, so 0.01 means the same thing
            # on both sides. Without it the 2D run would get photometrically
            # cleaner colour than the 3D run, which biases exactly the
            # comparison these two datasets exist to make.
            #
            # Not standard for image Diffusion Policy -- parity with the 3D
            # augmentation budget is the reason, not a tuning result.
            rgb_pixel_noise_std=0.0,
            # Random translation, in pixels, via reflect-pad then crop back.
            # NOT the crop the original Diffusion Policy uses: the stored
            # resolution is exactly patch-aligned, so cropping smaller would
            # break divisibility by the ViT patch size and change the token
            # count. Padding first keeps the image size, and the token count,
            # exact.
            image_shift_px=0,
            use_target_ee=False,
            ):
        super().__init__()

        if pose_source not in POSE_SOURCES:
            raise ValueError(
                f"unknown pose_source {pose_source!r} (expected one of {POSE_SOURCES})")

        self.cam_keys = _image_keys(zarr_path)
        self.campose_keys = [k.replace("image_", "campose_") for k in self.cam_keys]
        self.n_cams = len(self.cam_keys)
        self.cams, _ = load_preprocess_geometry(preprocess_config)
        if len(self.cams) != self.n_cams:
            raise ValueError(
                f"{preprocess_config} describes {len(self.cams)} cameras but the "
                f"zarr holds {self.n_cams}. Camera order is positional -- "
                "image_cam0 is cameras[0].")

        self.pose_source = pose_source
        self.camera_arm_slice = camera_arm_slice
        if pose_source == "proprio":
            if camera_arm_slice is None or len(camera_arm_slice) != self.n_cams:
                raise ValueError(
                    "pose_source='proprio' needs one camera_arm_slice entry per "
                    "camera (null for a camera that rides on no arm).")

        self.n_obs_steps = n_obs_steps
        self.use_target_ee = use_target_ee
        self.use_data_augmentation = use_data_augmentation
        self.agent_pos_noise_std = agent_pos_noise_std
        self.use_color_jitter = use_color_jitter
        self.rgb_pixel_noise_std = float(rgb_pixel_noise_std)
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range
        self.saturation_range = saturation_range
        self.image_shift_px = int(image_shift_px)

        # create_from_path, NOT copy_from_path. The 3D run loads its whole zarr
        # resident (~10.8 GB of clouds); the same policy here would be ~57 GB of
        # uint8 pixels on a 62 GB box. The converter writes one chunk per frame
        # so a 2-frame window touches 2 chunks per camera, not a 100-frame slab.
        self.replay_buffer = ReplayBuffer.create_from_path(zarr_path, mode="r")

        needed = ["state", "action", "joint_states"] + self.cam_keys
        if self.use_target_ee:
            needed.append("target_ee")
        missing = [k for k in needed if k not in self.replay_buffer]
        if missing:
            raise KeyError(f"{zarr_path} is missing {missing}")

        self._key_first_k = {}
        if n_obs_steps is not None:
            self._key_first_k = {k: n_obs_steps + pad_before for k in self.cam_keys}

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
            keys=needed,
            episode_mask=train_mask)
        self.train_mask = train_mask
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after

        shape = self.replay_buffer[self.cam_keys[0]].shape
        cprint("--------------------------", "cyan")
        cprint(f"Image dataset: {self.n_cams} cameras {self.cam_keys}", "cyan")
        cprint(f"Frame size: {shape[1]}x{shape[2]} (HxW), {self.replay_buffer.n_steps} "
               f"frames over {self.replay_buffer.n_episodes} episodes", "cyan")
        cprint(f"pose_source: {pose_source}", "cyan")
        cprint(f"Augmentation: shift={self.image_shift_px}px, "
               f"colour_jitter={use_color_jitter}, state_noise={use_data_augmentation}, "
               f"rgb_pixel_noise={self.rgb_pixel_noise_std}",
               "cyan")
        cprint("Backing: zarr on disk (create_from_path), not resident", "cyan")
        cprint("--------------------------", "cyan")

    # -- validation ---------------------------------------------------------

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=self.horizon,
            pad_before=self.pad_before,
            pad_after=self.pad_after,
            key_first_k=self._key_first_k,
            keys=list(self.sampler.keys),
            episode_mask=~self.train_mask)
        val_set.train_mask = ~self.train_mask
        # Validate on clean pixels: otherwise val loss moves with the
        # augmentation draw and stops being comparable between epochs.
        val_set.use_data_augmentation = False
        val_set.use_color_jitter = False
        val_set.image_shift_px = 0
        # Redundant while the pixel noise is gated on use_data_augmentation
        # above, but stated anyway: validation must be a fixed target, and a
        # future ungating of that flag should not silently start perturbing it.
        val_set.rgb_pixel_noise_std = 0.0
        return val_set

    # -- pose tag -----------------------------------------------------------

    def _campose(self, i, joints_t, agent_pos_t):
        """The 9-vector for camera `i` at one frame."""
        cam = self.cams[i]
        if self.pose_source == "none":
            return np.zeros(9, dtype=np.float64)

        if self.pose_source == "proprio":
            sl = self.camera_arm_slice[i]
            if sl is None:
                # Static camera: rides on no arm, so its pose is a constant and
                # the identity embedding already stands in for it.
                return np.zeros(9, dtype=np.float64)
            a, b = sl
            arm = np.asarray(agent_pos_t[a:b], dtype=np.float64)
            return pose9_from_pos_quat(arm[:3], arm[3:7])

        # extrinsic
        if cam["mount"] != "wrist":
            return pose9_from_matrix(cam["extrinsics"])
        lo, hi = cam["joint_slice"]
        T = cam["base"] @ panda_fk_flange(joints_t[lo:hi]) @ cam["extrinsics"]
        return pose9_from_matrix(T)

    # -- sampling -----------------------------------------------------------

    def __len__(self):
        return len(self.sampler)

    def _sample_to_data(self, sample, rng):
        action = sample["action"].astype(np.float32)
        if self.use_target_ee:
            action = np.concatenate(
                [action, sample["target_ee"].astype(np.float32)], axis=-1)

        n_steps = sample[self.cam_keys[0]].shape[0]
        if self.n_obs_steps is not None:
            n_steps = min(n_steps, self.n_obs_steps)
        agent_pos = sample["state"][:n_steps].astype(np.float32)
        joints = np.asarray(sample["joint_states"][:n_steps], dtype=np.float64)

        obs = {"agent_pos": agent_pos}
        # All ones. A camera always produces pixels, so the 3D path's "this
        # wrist camera saw nothing" case has no image counterpart -- the key
        # stays so the token layout matches, and the `absent` embedding is
        # simply never exercised.
        obs["cam_valid"] = np.ones((n_steps, self.n_cams), dtype=np.float32)

        for i, key in enumerate(self.cam_keys):
            img = np.asarray(sample[key][:n_steps])            # (T, H, W, 3) uint8
            img = img.astype(np.float32) / 255.0

            if self.image_shift_px > 0:
                img = self._shift(img, rng)
            if self.use_color_jitter:
                # ONE draw per window per camera -- the function samples its
                # factors once per call, and per-frame draws would destroy the
                # temporal consistency the two observation steps carry.
                img = apply_color_jitter(
                    img, self.brightness_range, self.contrast_range,
                    self.saturation_range).astype(np.float32)
            if self.use_data_augmentation and self.rgb_pixel_noise_std > 0:
                # INDEPENDENT per pixel and per frame, unlike the jitter and the
                # shift above. Those two are drawn once per window because they
                # are global transforms whose frame-to-frame variation would
                # fake camera motion or an exposure change. Sensor noise is the
                # opposite: it genuinely is independent every frame, and that is
                # what `add_noise` does on the 3D side too.
                #
                # Clipped at 2*std, matching add_noise's default relationship.
                img = add_noise(img, self.rgb_pixel_noise_std,
                                2 * self.rgb_pixel_noise_std).astype(np.float32)
                # add_noise does not clamp to the valid range; jitter does. Keep
                # pixels in [0, 1] so the normalizer downstream sees the range
                # it expects.
                np.clip(img, 0.0, 1.0, out=img)

            obs[key] = np.ascontiguousarray(img.transpose(0, 3, 1, 2))
            obs[self.campose_keys[i]] = np.stack([
                self._campose(i, joints[t], agent_pos[t]) for t in range(n_steps)
            ]).astype(np.float32)

        return {"obs": obs, "action": action}

    def _shift(self, img, rng):
        """Random translation: reflect-pad by `image_shift_px`, then crop back.

        One offset per window, held across the observation steps, for the same
        reason the colour jitter is drawn once: an independent shift per frame
        injects apparent camera motion between the two observation steps, which
        is precisely the cue the policy reads for velocity.
        """
        p = self.image_shift_px
        padded = np.pad(img, ((0, 0), (p, p), (p, p), (0, 0)), mode="reflect")
        dy = int(rng.integers(0, 2 * p + 1))
        dx = int(rng.integers(0, 2 * p + 1))
        h, w = img.shape[1], img.shape[2]
        return padded[:, dy:dy + h, dx:dx + w, :]

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        rng = np.random.default_rng(np.random.randint(0, 2 ** 31 - 1))
        sample = self.sampler.sample_sequence(idx)
        data = self._sample_to_data(sample, rng)

        if self.use_data_augmentation:
            # ONE draw per window, broadcast across observation frames -- see
            # the 3D dataset for why independent per-frame noise destroys the
            # velocity cue.
            ap = data["obs"]["agent_pos"]
            jitter = np.clip(
                np.random.normal(0.0, self.agent_pos_noise_std, ap.shape[-1]),
                -2 * self.agent_pos_noise_std, 2 * self.agent_pos_noise_std)
            data["obs"]["agent_pos"] = ap + jitter

        out = {
            "obs": {k: torch.from_numpy(v.astype(np.float32))
                    for k, v in data["obs"].items()},
            "action": torch.from_numpy(data["action"].astype(np.float32)),
        }
        return out

    # -- normalizer ---------------------------------------------------------

    def get_normalizer(self, mode="limits", **kwargs):
        """Images pass through UNCHANGED; only agent_pos, action and campose move.

        The min-max fit must not touch pixels. The encoder applies ImageNet
        statistics itself, which is the one rescaling DINOv2 expects, and a
        second data-fitted map in front of it would make the input depend on
        which scenes happened to be recorded -- something inference, which never
        sees this zarr, could not reproduce.

        `agent_pos`, `action` and the campose translation scale are left EXACTLY
        as the unfused 3D run has them, including `METRIC_SCALE_M`. That factor
        is meaningless without point clouds to share a unit with, but matching
        it keeps the conditioning numerically identical between the two runs,
        which is worth more here than a tidier constant.
        """
        action = np.asarray(self.replay_buffer["action"])
        if self.use_target_ee:
            action = np.concatenate(
                [action, np.asarray(self.replay_buffer["target_ee"])], axis=-1)

        normalizer = LinearNormalizer()
        normalizer.fit(
            data={"action": action,
                  "agent_pos": np.asarray(self.replay_buffer["state"])},
            last_n_dims=1, mode=mode, **kwargs)

        for key in self.cam_keys:
            normalizer[key] = SingleFieldLinearNormalizer.create_identity()

        k = 1.0 / METRIC_SCALE_M
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

        normalizer["cam_valid"] = SingleFieldLinearNormalizer.create_identity()
        return normalizer

    def get_all_actions(self) -> torch.Tensor:
        return torch.from_numpy(np.asarray(self.replay_buffer["action"]))
