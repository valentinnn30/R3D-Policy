"""RealHybridDataset -- some cameras as point clouds, the rest as images.

Plan: ~/ros2_ws/hybrid_2d3d_ablations_plan.txt

    H1   : cameras_3d [0],    cameras_2d [1, 2]              top 3D, wrists 2D
    H2-U : cameras_3d [1, 2], cameras_2d [0]                 wrists 3D unfused
    H2-F : cameras_3d [1, 2], cameras_2d [0], fuse_3d: true  wrists 3D fused

COMPOSITION, not reimplementation. This holds the two single-modality datasets,
each restricted to its own cameras, and merges what they return. Every camera
block is therefore produced by exactly the code that produces it in the
non-hybrid runs -- the same stored FPS clouds, the same `select_camera_frame`
crop and uniform draw (or the same `fuse_cameras`), the same image shift /
jitter / pixel noise, the same normalizers, the same pose tags:

    ds3d = RealMultiCamUnfusedDataset(cameras=cameras_3d, ...)   # or the fused
    ds2d = RealMultiCamImageDataset(cameras=cameras_2d, ...)     #   RealMultiCamDataset

The two zarrs are frame-identical (the 2D sync reads its time grid from the 3D
output; verified equal `episode_ends`, `state`, `action`), so one sample index
is the same window in both. That is ASSERTED at construction, not assumed.

Pose tags come from `pose_source_3d` for the 3D blocks and `pose_source_2d` for
the image blocks, resolved per block by `r3d.common.hybrid_layout`.
"""

import copy
from typing import Dict

import numpy as np
import torch
from termcolor import cprint

from r3d.common.hybrid_layout import (
    describe,
    effective_dataset_source,
    hybrid_blocks,
    resolve_block_poses,
)
from r3d.dataset.base_dataset import BaseDataset
from r3d.dataset.real_image_dataset import RealMultiCamImageDataset
from r3d.dataset.real_multicam_dataset import (
    RealMultiCamDataset,
    RealMultiCamUnfusedDataset,
)
from r3d.model.common.normalizer import LinearNormalizer, SingleFieldLinearNormalizer


class RealHybridDataset(BaseDataset):

    def __init__(self,
            zarr_path_3d,
            zarr_path_2d,
            preprocess_config_3d,
            preprocess_config_2d,
            cameras_3d,
            cameras_2d,
            fuse_3d=False,
            horizon=1,
            pad_before=0,
            pad_after=0,
            seed=42,
            val_ratio=0.0,
            max_train_episodes=None,
            n_obs_steps=None,
            # one source per modality -- see hybrid_layout.resolve_block_poses
            pose_source_3d="extrinsic",
            pose_source_2d="none",
            # indexed by GLOBAL camera index, e.g. [null, [0, 7], [7, 14]]
            camera_arm_slice=None,
            # unfused 3D blocks, in cameras_3d order
            points_per_camera_encoder=None,
            num_groups_per_camera=None,
            # fused 3D block
            fused_num_points=None,
            fused_num_groups=None,
            # extrinsics randomization -- OFF for every run (2026-09-04)
            use_extrinsics_randomization=False,
            extrinsics_rot_sigma_deg=0.7,
            extrinsics_trans_sigma_m=0.004,
            anchor_first_camera=False,
            fuse_method="random",
            fuse_device="cpu",
            # augmentation: the union of the two single-modality task configs
            use_data_augmentation=False,
            pc_xyz_noise_std=0.002,
            pc_rgb_noise_std=0.01,
            agent_pos_noise_std=0.0002,
            use_color_jitter=False,
            brightness_range=(-0.125, 0.125),
            contrast_range=(0.5, 1.5),
            saturation_range=(0.5, 1.5),
            rgb_pixel_noise_std=0.0,
            image_shift_px=0,
            use_target_ee=False,
            ):
        super().__init__()

        self.cameras_3d = [int(c) for c in cameras_3d]
        self.cameras_2d = [int(c) for c in cameras_2d]
        self.fuse_3d = bool(fuse_3d)
        self.blocks = hybrid_blocks(self.cameras_3d, self.cameras_2d, self.fuse_3d)
        self.resolved = resolve_block_poses(
            self.blocks, pose_source_3d, pose_source_2d, camera_arm_slice)
        self.pose_source_3d = pose_source_3d
        self.pose_source_2d = pose_source_2d
        self.camera_arm_slice = camera_arm_slice

        def sub_slices(cams):
            if camera_arm_slice is None:
                return None
            return [camera_arm_slice[c] for c in cams]

        common = dict(horizon=horizon, pad_before=pad_before, pad_after=pad_after,
                      seed=seed, val_ratio=val_ratio,
                      max_train_episodes=max_train_episodes, n_obs_steps=n_obs_steps,
                      use_target_ee=use_target_ee)
        jitter = dict(use_data_augmentation=use_data_augmentation,
                      agent_pos_noise_std=agent_pos_noise_std,
                      use_color_jitter=use_color_jitter,
                      brightness_range=brightness_range,
                      contrast_range=contrast_range,
                      saturation_range=saturation_range)
        kw3d = dict(zarr_path=zarr_path_3d, preprocess_config=preprocess_config_3d,
                    cameras=self.cameras_3d,
                    use_extrinsics_randomization=use_extrinsics_randomization,
                    extrinsics_rot_sigma_deg=extrinsics_rot_sigma_deg,
                    extrinsics_trans_sigma_m=extrinsics_trans_sigma_m,
                    anchor_first_camera=anchor_first_camera,
                    fuse_method=fuse_method, fuse_device=fuse_device,
                    pc_xyz_noise_std=pc_xyz_noise_std,
                    pc_rgb_noise_std=pc_rgb_noise_std,
                    **common, **jitter)

        if self.fuse_3d:
            if fused_num_points is None or fused_num_groups is None:
                raise ValueError("fuse_3d needs fused_num_points and fused_num_groups")
            if int(fused_num_groups) > int(fused_num_points):
                raise ValueError(
                    f"fused_num_groups {fused_num_groups} exceeds fused_num_points "
                    f"{fused_num_points}: the encoder picks that many FPS centres")
            self.fused_num_points = int(fused_num_points)
            self.fused_num_groups = int(fused_num_groups)
            self.ds3d = RealMultiCamDataset(num_points=self.fused_num_points, **kw3d)
        else:
            self.ds3d = RealMultiCamUnfusedDataset(
                points_per_camera_encoder=points_per_camera_encoder,
                num_groups_per_camera=num_groups_per_camera,
                pose_source=effective_dataset_source(self.resolved, "3d"),
                camera_arm_slice=sub_slices(self.cameras_3d),
                **kw3d)

        self.ds2d = RealMultiCamImageDataset(
            zarr_path=zarr_path_2d, preprocess_config=preprocess_config_2d,
            cameras=self.cameras_2d,
            pose_source=effective_dataset_source(self.resolved, "2d"),
            camera_arm_slice=sub_slices(self.cameras_2d),
            rgb_pixel_noise_std=rgb_pixel_noise_std,
            image_shift_px=image_shift_px,
            **common, **jitter)

        self._check_aligned()

        cprint("--------------------------", "cyan")
        cprint(f"HYBRID 2D/3D: 3D cams {self.cameras_3d} "
               f"({'FUSED' if self.fuse_3d else 'unfused'}), 2D cams {self.cameras_2d}",
               "cyan")
        cprint(describe(self.blocks, self.resolved), "cyan")
        cprint("--------------------------", "cyan")

    # -- alignment ----------------------------------------------------------

    def _check_aligned(self):
        """The two zarrs must describe the same frames, or index i is two instants."""
        rb3, rb2 = self.ds3d.replay_buffer, self.ds2d.replay_buffer
        if not np.array_equal(rb3.episode_ends[:], rb2.episode_ends[:]):
            raise ValueError(
                "3D and 2D zarrs have different episode_ends -- they are not the "
                "same collection, so a sample index would be two different windows")
        # (N, 48) float32 is ~10 MB -- cheap enough to compare in full.
        if not np.array_equal(np.asarray(rb3["state"][:]), np.asarray(rb2["state"][:])):
            raise ValueError("3D and 2D zarrs disagree in `state` -- frames are not aligned")
        if not np.array_equal(self.ds3d.train_mask, self.ds2d.train_mask):
            raise ValueError("train/val episode split differs between the 3D and 2D datasets")
        if not np.array_equal(self.ds3d.sampler.indices, self.ds2d.sampler.indices):
            raise ValueError("sampler windows differ between the 3D and 2D datasets")

    # -- validation ---------------------------------------------------------

    def get_validation_dataset(self):
        val = copy.copy(self)
        val.ds3d = self.ds3d.get_validation_dataset()
        val.ds2d = self.ds2d.get_validation_dataset()
        return val

    # -- sampling -----------------------------------------------------------

    def __len__(self):
        return len(self.ds3d)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        d3 = self.ds3d[idx]
        d2 = self.ds2d[idx]
        if not torch.equal(d3["action"], d2["action"]):
            raise RuntimeError(f"index {idx}: 3D and 2D windows carry different actions")

        o3, o2 = d3["obs"], d2["obs"]
        # agent_pos from the 3D side ONLY, so exactly one window-level jitter is
        # applied, as in every other run. Both sides computed their proprio tags
        # from the un-jittered agent_pos inside _sample_to_data.
        obs = {"agent_pos": o3["agent_pos"]}
        T = o3["agent_pos"].shape[0]
        cam_valid = torch.ones((T, len(self.blocks)), dtype=torch.float32)
        for b_i, b in enumerate(self.blocks):
            if b["kind"] == "image":
                obs[b["key"]] = o2[b["key"]]
                obs[b["campose_key"]] = o2[b["campose_key"]]
                cam_valid[:, b_i] = o2["cam_valid"][:, self.cameras_2d.index(b["cameras"][0])]
            elif b["kind"] == "pc":
                obs[b["key"]] = o3[b["key"]]
                obs[b["campose_key"]] = o3[b["campose_key"]]
                cam_valid[:, b_i] = o3["cam_valid"][:, self.cameras_3d.index(b["cameras"][0])]
            else:  # pc_fused: always valid, exactly as in the fused run
                obs[b["key"]] = o3["point_cloud"]
        obs["cam_valid"] = cam_valid
        return {"obs": obs, "action": d3["action"]}

    # -- normalizer ---------------------------------------------------------

    def get_normalizer(self, mode="limits", **kwargs):
        """The union of both sub-datasets' normalizers, key for key.

        agent_pos / action are fitted identically by both (byte-identical
        arrays); the 3D fit is used and the 2D one is asserted equal to it.
        """
        n3 = self.ds3d.get_normalizer(mode=mode, **kwargs)
        n2 = self.ds2d.get_normalizer(mode=mode, **kwargs)

        for key in ("agent_pos", "action"):
            p3, p2 = n3.params_dict[key], n2.params_dict[key]
            for pk in ("scale", "offset"):
                if not torch.allclose(p3[pk], p2[pk], rtol=0, atol=1e-6):
                    raise ValueError(
                        f"{key}.{pk} normalizer differs between the 3D and 2D zarrs")

        out = LinearNormalizer()
        out["agent_pos"] = n3["agent_pos"]
        out["action"] = n3["action"]
        for b in self.blocks:
            src = n2 if b["kind"] == "image" else n3
            src_key = "point_cloud" if b["kind"] == "pc_fused" else b["key"]
            out[b["key"]] = src[src_key]
            if b["campose_key"] is not None:
                out[b["campose_key"]] = src[b["campose_key"]]
        out["cam_valid"] = SingleFieldLinearNormalizer.create_identity()
        return out

    def get_all_actions(self) -> torch.Tensor:
        return self.ds2d.get_all_actions()
