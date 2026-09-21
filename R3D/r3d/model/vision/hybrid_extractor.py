"""MultiModalEncoder -- point-cloud blocks and image blocks in one token sequence.

Plan: ~/ros2_ws/hybrid_2d3d_ablations_plan.txt

The hybrid counterpart of `MultiCamDP3Encoder` and `MultiCamImageEncoder`, built
ONLY from their parts. Nothing in `pointnet_extractor.py` or
`image_extractor.py` is modified; everything is imported read-only, following
the precedent the 2D encoder set:

* ONE shared Uni3D for every 3D block (per unfused camera, or the fused wrist
  cloud), called with that block's `num_groups` -- as MultiCamDP3Encoder does;
* ONE shared DinoV2Tokenizer for every image block, with the same freeze /
  partial-unfreeze flags -- as MultiCamImageEncoder does;
* TWO CameraPoseEmbedding instances, one per modality. Each is built with the
  existing derivation from ITS OWN pose source (`hybrid_layout`), so each
  modality's blocks are tagged exactly as its non-hybrid run tags them. One
  instance could not hold 3D 'proprio' (per-camera MLPs) next to 2D
  'extrinsic' (a shared MLP): `per_camera_mlp` is a single flag.

Return contract is unchanged -- `(features, positional_encoding)`, one PE entry
per patch token, `agent_pos` broadcast onto the feature axis under
`cat_on_token: false` -- so `dp3.py`'s conditioning path needs no change. Uni3D's
PE (PositionEmbeddingRandom(embed_dim // 2)) and the image PE
(PositionEmbeddingRandomND(out_channel // 2)) are both `out_channel` wide, which
is what lets the two modalities share one sequence; that is asserted.
"""

from typing import Dict

import torch
import torch.nn as nn
from termcolor import cprint

from r3d.common.hybrid_layout import (
    describe,
    hybrid_blocks,
    modality_pose_config,
    resolve_block_poses,
)
from r3d.model.vision.image_extractor import DinoV2Tokenizer, PositionEmbeddingRandomND
from r3d.model.vision.pointnet_extractor import (
    CameraPoseEmbedding,
    Uni3DPointcloudEncoder,
    create_mlp,
)


class MultiModalEncoder(nn.Module):

    def __init__(self,
                 observation_space: Dict,
                 cameras_3d,
                 cameras_2d,
                 fuse_3d=False,
                 out_channel=256,
                 state_mlp_size=(64, 64), state_mlp_activation_fn=nn.ReLU,
                 pointcloud_encoder_cfg=None,
                 image_encoder_cfg=None,
                 pointnet_type="uni3d",
                 fps_random_config=None,
                 cat_on_token=False,
                 num_groups_per_camera=None,
                 fused_num_groups=None,
                 pose_source_3d="extrinsic",
                 pose_source_2d="none",
                 camera_arm_slice=None,
                 ):
        super().__init__()
        pointcloud_encoder_cfg = dict(pointcloud_encoder_cfg or {})
        image_encoder_cfg = dict(image_encoder_cfg or {})
        state_mlp_size = (64, out_channel)
        self.cat_on_token = cat_on_token

        # -- layout -------------------------------------------------------------
        self.blocks = hybrid_blocks(cameras_3d, cameras_2d, fuse_3d)
        self.resolved = resolve_block_poses(
            self.blocks, pose_source_3d, pose_source_2d, camera_arm_slice)

        expected = {b["key"] for b in self.blocks} | {
            b["campose_key"] for b in self.blocks if b["campose_key"]} | {"cam_valid"}
        missing = sorted(expected - set(observation_space))
        if missing:
            raise RuntimeError(
                f"shape_meta is missing {missing} for cameras_3d={list(cameras_3d)} "
                f"cameras_2d={list(cameras_2d)} fuse_3d={fuse_3d}. The task config's "
                "shape_meta and its dataset camera lists disagree.")
        stray = sorted(k for k in observation_space if k not in expected and (
            k.startswith(("point_cloud", "image_cam", "campose_cam"))))
        if stray:
            raise RuntimeError(f"shape_meta has {stray}, which no block consumes")
        cv = tuple(observation_space["cam_valid"])
        if cv != (len(self.blocks),):
            raise RuntimeError(
                f"cam_valid shape {cv} != ({len(self.blocks)},): one entry per BLOCK")

        self.low_dim_keys = [k for k in observation_space if k not in expected]
        if len(self.low_dim_keys) != 1:
            raise RuntimeError(
                f"expected exactly one low-dim key (agent_pos), got {self.low_dim_keys}. "
                "dp3.py derives num_patches from the positional encoding, so a second "
                "one would misalign it against the tokens.")

        # -- 3D trunk: one shared Uni3D ------------------------------------------
        pc_blocks = [b for b in self.blocks if b["kind"] != "image"]
        self.uni3d = None
        self.block_groups = {}
        if pc_blocks:
            if pointnet_type not in ("uni3d", "uni3d_pretrained"):
                raise NotImplementedError(
                    f"MultiModalEncoder supports the uni3d encoders only, got {pointnet_type!r}")
            if pointcloud_encoder_cfg.get("feature_mode") != "pointsam":
                raise NotImplementedError(
                    "MultiModalEncoder needs feature_mode='pointsam' (per-patch tokens)")
            embed_dim = int(pointcloud_encoder_cfg.get("embed_dim", out_channel))
            if embed_dim != int(out_channel):
                raise ValueError(
                    f"pointcloud_encoder_cfg.embed_dim {embed_dim} != encoder_output_dim "
                    f"{out_channel}. Point and image tokens (and their PEs) share one "
                    "sequence, so both trunks must emit the same width.")
            self.fps_random_config = fps_random_config or {
                'use_random': True, 'random_start': True,
                'random_noise_scale': 0, 'shuffle_output': True,
            }
            uni3d_config = dict(pointcloud_encoder_cfg)
            uni3d_config["embed_dim"] = embed_dim
            if pointnet_type == "uni3d_pretrained":
                uni3d_config.setdefault("use_pretrained_weights", True)
            uni3d_config["fps_random_config"] = self.fps_random_config
            self.uni3d = Uni3DPointcloudEncoder(**uni3d_config)

            unfused = [b for b in pc_blocks if b["kind"] == "pc"]
            if unfused:
                if num_groups_per_camera is None or len(num_groups_per_camera) != len(unfused):
                    raise ValueError(
                        f"num_groups_per_camera={num_groups_per_camera} needs one entry "
                        f"per unfused 3D camera {[b['cameras'][0] for b in unfused]}")
                for b, g in zip(unfused, num_groups_per_camera):
                    self.block_groups[b["key"]] = int(g)
            for b in pc_blocks:
                if b["kind"] == "pc_fused":
                    if fused_num_groups is None:
                        raise ValueError("a fused 3D block needs fused_num_groups")
                    self.block_groups[b["key"]] = int(fused_num_groups)
            for b in pc_blocks:
                n_pts = int(observation_space[b["key"]][0])
                if self.block_groups[b["key"]] > n_pts:
                    raise ValueError(
                        f"{b['key']}: {self.block_groups[b['key']]} groups > {n_pts} points")

        # -- 2D trunk: one shared DinoV2Tokenizer --------------------------------
        img_blocks = [b for b in self.blocks if b["kind"] == "image"]
        self.dino = None
        if img_blocks:
            shapes = {tuple(observation_space[b["key"]]) for b in img_blocks}
            if len(shapes) != 1:
                raise RuntimeError(
                    f"image blocks disagree on shape {shapes}; the trunk is shared")
            shape = shapes.pop()
            if len(shape) != 3 or shape[0] != 3:
                raise RuntimeError(f"expected image shape (3, H, W), got {shape}")
            self.img_h, self.img_w = int(shape[1]), int(shape[2])
            image_encoder_cfg.setdefault("out_channel", out_channel)
            if int(image_encoder_cfg["out_channel"]) != int(out_channel):
                raise ValueError("image_encoder_cfg.out_channel must equal encoder_output_dim")
            self.dino = DinoV2Tokenizer(**image_encoder_cfg)
            self.grid_h, self.grid_w = self.dino.grid_shape(self.img_h, self.img_w)
            self.tokens_per_image = self.grid_h * self.grid_w
            # Patch centres in [-1, 1], row-major -- as MultiCamImageEncoder.
            rows = (torch.arange(self.grid_h, dtype=torch.float32) + 0.5) / self.grid_h
            cols = (torch.arange(self.grid_w, dtype=torch.float32) + 0.5) / self.grid_w
            vv, uu = torch.meshgrid(rows * 2 - 1, cols * 2 - 1, indexing="ij")
            self.register_buffer(
                "patch_coords", torch.stack([uu, vv], dim=-1).reshape(-1, 2),
                persistent=False)
            self.pe_layer_2d = PositionEmbeddingRandomND(out_channel // 2, coord_dim=2)

        # -- camera tags: one CameraPoseEmbedding per modality -------------------
        self.local_index = []
        counters = {"3d": 0, "2d": 0}
        for r in self.resolved:
            self.local_index.append(counters[r["modality"]])
            counters[r["modality"]] += 1
        self.pose_embed_3d = self.pose_embed_2d = None
        if counters["3d"]:
            n, per_cam, has = modality_pose_config(self.blocks, self.resolved, "3d")
            self.pose_embed_3d = CameraPoseEmbedding(
                n, out_channel, per_camera_mlp=per_cam, has_pose=has)
        if counters["2d"]:
            n, per_cam, has = modality_pose_config(self.blocks, self.resolved, "2d")
            self.pose_embed_2d = CameraPoseEmbedding(
                n, out_channel, per_camera_mlp=per_cam, has_pose=has)

        # -- agent_pos -----------------------------------------------------------
        output_dim = state_mlp_size[-1]
        net_arch = state_mlp_size[:-1] if len(state_mlp_size) > 1 else []
        self.low_dim_mlps = nn.ModuleDict()
        for key in self.low_dim_keys:
            s = observation_space[key]
            if len(s) != 1:
                raise RuntimeError(f"Low-dimensional obs '{key}' must be rank-1, got {s}")
            self.low_dim_mlps[key] = nn.Sequential(
                *create_mlp(s[0], output_dim, net_arch, state_mlp_activation_fn))

        self.n_output_channels = out_channel if cat_on_token else \
            out_channel + output_dim * len(self.low_dim_keys)

        n_tokens = sum(self.block_groups.get(b["key"], 0) if b["kind"] != "image"
                       else self.tokens_per_image for b in self.blocks)
        self.n_tokens = n_tokens
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen = sum(p.numel() for p in self.parameters() if not p.requires_grad)
        cprint("[MultiModalEncoder] blocks (token order):", "yellow")
        cprint(describe(self.blocks, self.resolved), "yellow")
        cprint(f"[MultiModalEncoder] tokens per block: " + ", ".join(
            f"{b['key']}={self.block_groups[b['key']] if b['kind'] != 'image' else self.tokens_per_image}"
            for b in self.blocks) + f"  (total {n_tokens})", "yellow")
        cprint(f"[MultiModalEncoder] params: {trainable/1e6:.2f}M trainable, "
               f"{frozen/1e6:.2f}M frozen; output dim {self.n_output_channels}", "yellow")

    def forward(self, observations: Dict, eval=False):
        cam_valid = observations["cam_valid"]
        feats, pes = [], []
        for b_i, b in enumerate(self.blocks):
            x = observations[b["key"]]
            if b["kind"] == "image":
                if x.ndim != 4 or x.shape[1] != 3:
                    raise AssertionError(f"{b['key']}: expected [B, 3, H, W], got {tuple(x.shape)}")
                tokens = self.dino(x)
                pe = self.pe_layer_2d(self.patch_coords).unsqueeze(0).repeat(
                    tokens.shape[0], 1, 1)
                embed = self.pose_embed_2d
            else:
                assert x.ndim == 3, f"{b['key']}: expected [B, N, C], got {x.shape}"
                if x.shape[-1] == 3:
                    x = torch.cat([x, torch.zeros_like(x)], dim=-1)
                elif x.shape[-1] > 6:
                    x = x[..., :6]
                tokens, pe = self.uni3d(x, eval, num_groups=self.block_groups[b["key"]])
                embed = self.pose_embed_3d

            if b["campose_key"] is not None:
                pose9 = observations[b["campose_key"]]
            else:
                pose9 = tokens.new_zeros((tokens.shape[0], 9))
            tokens = embed(tokens, self.local_index[b_i], pose9, cam_valid[..., b_i])
            feats.append(tokens)
            pes.append(pe)

        pn_feat = torch.cat(feats, dim=1)
        pc_pe = torch.cat(pes, dim=1)

        low_dim_features = []
        for key in self.low_dim_keys:
            f = self.low_dim_mlps[key](observations[key])
            if self.cat_on_token:
                f = f.unsqueeze(1)
            else:
                f = f.unsqueeze(1).expand(-1, pn_feat.shape[1], -1)
            low_dim_features.append(f)

        final_feat = torch.cat([pn_feat] + low_dim_features,
                               dim=-2 if self.cat_on_token else -1)
        return final_feat, pc_pe

    def output_shape(self):
        return self.n_output_channels
