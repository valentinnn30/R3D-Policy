"""Frozen DINOv2 image encoder -- the 2D counterpart of `MultiCamDP3Encoder`.

Design brief: `~/ros2_ws/tmp/dinov2_2d_baseline_design.md`.

This file exists so the 2D baseline changes ONLY the encoder. Everything
downstream is untouched: the tokens produced here go into the same
`one_way_transformer` cross-attention, tagged by the same
`CameraPoseEmbedding`, with the same `(features, positional_encoding)` return
contract that `dp3.py` already consumes. Nothing in `pointnet_extractor.py` is
modified or re-exported -- `CameraPoseEmbedding` and `create_mlp` are imported
read-only, so the 3D path is bit-for-bit unaffected.

Three things differ from the point-cloud encoder, and only three:

* the trunk is a ViT over pixels instead of Uni3D over point patches;
* the trunk is FROZEN, so the only trainable parameters here are the output
  projection, the camera tag and (optionally) nothing else;
* token positions are 2-d pixel centres instead of 3-d patch centres, which is
  why `PositionEmbeddingRandomND` exists rather than reusing
  `PositionEmbeddingRandom` -- that one hardcodes a (3, num_pos_feats) matrix.
"""

from typing import Dict, Optional

import numpy as np
import timm
import torch
import torch.nn as nn
from termcolor import cprint

# Read-only imports. Nothing below mutates them, so the 3D encoder is unaffected.
from r3d.model.vision.pointnet_extractor import CameraPoseEmbedding, create_mlp

# DINOv2 was trained with ImageNet statistics. The dataset hands images through
# an IDENTITY normalizer in [0, 1] precisely so this stays the one and only
# place pixels are rescaled -- see RealMultiCamImageDataset.get_normalizer.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class PositionEmbeddingRandomND(nn.Module):
    """Random-Fourier positional encoding over `coord_dim` normalised axes.

    Identical maths to `PositionEmbeddingRandom` in `pointnet_extractor.py`,
    with the coordinate dimension made a parameter. That one registers a
    (3, num_pos_feats) buffer, so it cannot take 2-d pixel centres; copying the
    six lines here is cheaper than making the 3D file's signature depend on the
    2D experiment.

    The matrix is a BUFFER, so the frequencies are drawn once and travel with
    the checkpoint. Re-drawing them on reload would silently re-encode every
    position.
    """

    def __init__(self, num_pos_feats: int = 64, coord_dim: int = 2,
                 scale: Optional[float] = None) -> None:
        super().__init__()
        if scale is None or scale <= 0.0:
            scale = 1.0
        self.coord_dim = int(coord_dim)
        self.register_buffer(
            "positional_encoding_gaussian_matrix",
            scale * torch.randn((self.coord_dim, num_pos_feats)),
        )

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """coords: (..., coord_dim), normalised to [-1, 1]. -> (..., 2*feats)."""
        if coords.shape[-1] != self.coord_dim:
            raise ValueError(
                f"expected last dim {self.coord_dim}, got {coords.shape[-1]}")
        if (coords < -1 - 1e-6).any() or (coords > 1 + 1e-6).any():
            raise ValueError("Input coordinates must be normalized to [-1, 1].")
        c = coords @ self.positional_encoding_gaussian_matrix
        c = 2 * np.pi * c
        return torch.cat([torch.sin(c), torch.cos(c)], dim=-1)


class DinoV2Tokenizer(nn.Module):
    """One image -> its patch tokens, projected to the policy's width.

    Registers matter here and are not cosmetic: this consumes PATCH tokens, and
    the high-norm artifact tokens that DINOv2 develops without registers live in
    exactly that set. The `reg4` variants are what the artifact paper fixes.
    They also make the ViT-S -> ViT-L scale-up a one-string change.

    CLS and the register tokens are dropped. They carry no pixel coordinate, so
    they have no entry in the positional encoding, and `dp3.py` derives
    `num_patches` from that encoding's length -- keeping them would misalign the
    two silently rather than raise.
    """

    def __init__(self,
                 model_name: str = "vit_small_patch14_reg4_dinov2.lvd142m",
                 out_channel: int = 256,
                 pretrained: bool = True,
                 freeze: bool = True,
                 drop_path_rate: float = 0.0):
        super().__init__()
        # dynamic_img_size lets the pretrained position embedding be
        # interpolated to a non-square, non-518 input. Without it timm asserts
        # on the 322x182 frames.
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,
            dynamic_img_size=True,
            drop_path_rate=drop_path_rate,
        )
        self.embed_dim = self.backbone.embed_dim
        self.num_prefix_tokens = int(getattr(self.backbone, "num_prefix_tokens", 1))
        patch = self.backbone.patch_embed.patch_size
        self.patch_size = int(patch[0] if isinstance(patch, (tuple, list)) else patch)

        self.freeze = bool(freeze)
        if self.freeze:
            for p in self.backbone.parameters():
                p.requires_grad_(False)
            self.backbone.eval()

        self.out_proj = nn.Linear(self.embed_dim, out_channel)

        self.register_buffer(
            "pixel_mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1), persistent=False)
        self.register_buffer(
            "pixel_std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1), persistent=False)

        cprint(f"[DinoV2Tokenizer] {model_name}: embed_dim {self.embed_dim}, "
               f"patch {self.patch_size}, prefix tokens {self.num_prefix_tokens}, "
               f"{'FROZEN' if self.freeze else 'trainable'}", "yellow")

    def train(self, mode: bool = True):
        """Keep a frozen trunk in eval mode whatever the parent does.

        `requires_grad_(False)` stops gradients but leaves stochastic depth and
        dropout active, which would make a "frozen" encoder return different
        features every call.
        """
        super().train(mode)
        if self.freeze:
            self.backbone.eval()
        return self

    def grid_shape(self, height: int, width: int):
        if height % self.patch_size or width % self.patch_size:
            raise ValueError(
                f"image {height}x{width} is not divisible by the patch size "
                f"{self.patch_size}. The stored resolution is chosen to be "
                "patch-aligned; resize at conversion time, not here.")
        return height // self.patch_size, width // self.patch_size

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        """img: [B, 3, H, W] in [0, 1]. -> [B, n_patches, out_channel]."""
        x = (img - self.pixel_mean) / self.pixel_std
        if self.freeze:
            with torch.no_grad():
                feats = self.backbone.forward_features(x)
        else:
            feats = self.backbone.forward_features(x)
        patches = feats[:, self.num_prefix_tokens:, :]
        return self.out_proj(patches)


class MultiCamImageEncoder(nn.Module):
    """Per-camera images -> one concatenated token sequence.

    Deliberately the same shape of object as `MultiCamDP3Encoder`: same
    constructor role, same `(features, pe)` return, same camera tagging, so
    `dp3.py`'s conditioning path needs no change at all.

    The trunk is SHARED across cameras -- one instance, N forward passes -- not
    one encoder per view, which is the original Diffusion Policy default
    (`share_rgb_model: False`). Per-view weights suit a small network trained
    from scratch; they are the wrong default for a large pretrained trunk, where
    they would finetune N independently drifting copies of one initialisation,
    each seeing 1/N of the visual diversity. It would also differ from the 3D
    policy in a second place, which is the confound this whole baseline exists
    to avoid.
    """

    def __init__(self,
                 observation_space: Dict,
                 out_channel=256,
                 state_mlp_size=(64, 64), state_mlp_activation_fn=nn.ReLU,
                 image_encoder_cfg=None,
                 cat_on_token=True,
                 per_camera_pose_mlp=False,
                 camera_has_pose=None,
                 ):
        super().__init__()
        image_encoder_cfg = dict(image_encoder_cfg or {})
        state_mlp_size = (64, out_channel)

        self.image_keys = sorted(
            [k for k in observation_space if k.startswith("image_cam")],
            key=lambda k: int(k.replace("image_cam", "")))
        if not self.image_keys:
            raise RuntimeError(
                "MultiCamImageEncoder found no image_cam* keys in shape_meta. "
                "A point-cloud zarr wants MultiCamDP3Encoder instead.")
        self.n_cams = len(self.image_keys)
        self.campose_keys = [
            k.replace("image_", "campose_") for k in self.image_keys]

        missing = [k for k in self.campose_keys if k not in observation_space]
        if missing:
            raise RuntimeError(
                f"shape_meta is missing {missing}. Every camera needs its pose "
                "vector; the dataset emits one per camera whatever pose_source "
                "is set to (zeros for 'none').")
        if "cam_valid" not in observation_space:
            raise RuntimeError(
                "shape_meta is missing 'cam_valid'. It is all-ones for images "
                "-- a camera always produces pixels -- but the key has to exist "
                "so the token layout matches the 3D path exactly.")

        consumed = set(self.image_keys) | set(self.campose_keys) | {"cam_valid"}
        self.low_dim_keys = [k for k in observation_space if k not in consumed]
        if len(self.low_dim_keys) != 1:
            raise RuntimeError(
                f"expected exactly one low-dim key (agent_pos), got "
                f"{self.low_dim_keys}. dp3.py computes num_patches as "
                "num_tokens - 1 under cat_on_token, so a second one would "
                "silently misalign the positional encoding against the tokens.")
        self.low_dim_shapes = {k: observation_space[k] for k in self.low_dim_keys}
        self.cat_on_token = cat_on_token

        # Every camera must be the same resolution, because they share a trunk
        # and their tokens are concatenated into one sequence.
        shapes = {tuple(observation_space[k]) for k in self.image_keys}
        if len(shapes) != 1:
            raise RuntimeError(
                f"cameras disagree on image shape: {shapes}. The trunk is "
                "shared, so they must match.")
        shape = shapes.pop()
        if len(shape) != 3 or shape[0] != 3:
            raise RuntimeError(
                f"expected image shape (3, H, W), got {shape}. The dataset "
                "emits channels-first float images in [0, 1].")
        self.img_h, self.img_w = int(shape[1]), int(shape[2])

        image_encoder_cfg.setdefault("out_channel", out_channel)
        self.extractor = DinoV2Tokenizer(**image_encoder_cfg)
        self.grid_h, self.grid_w = self.extractor.grid_shape(self.img_h, self.img_w)
        self.tokens_per_camera = self.grid_h * self.grid_w

        # Pixel centres of each patch, normalised to [-1, 1], in the token order
        # timm produces (row-major over the patch grid). A buffer rather than a
        # recomputation, and non-persistent because it is fully determined by
        # the image size.
        rows = (torch.arange(self.grid_h, dtype=torch.float32) + 0.5) / self.grid_h
        cols = (torch.arange(self.grid_w, dtype=torch.float32) + 0.5) / self.grid_w
        vv, uu = torch.meshgrid(rows * 2 - 1, cols * 2 - 1, indexing="ij")
        self.register_buffer(
            "patch_coords", torch.stack([uu, vv], dim=-1).reshape(-1, 2),
            persistent=False)

        self.pe_layer = PositionEmbeddingRandomND(out_channel // 2, coord_dim=2)

        if camera_has_pose is not None and len(camera_has_pose) != self.n_cams:
            raise ValueError(
                f"camera_has_pose has {len(camera_has_pose)} entries for "
                f"{self.n_cams} cameras")
        self.pose_embed = CameraPoseEmbedding(
            self.n_cams, out_channel,
            per_camera_mlp=per_camera_pose_mlp,
            has_pose=camera_has_pose)

        output_dim = state_mlp_size[-1]
        net_arch = state_mlp_size[:-1] if len(state_mlp_size) > 1 else []
        self.low_dim_mlps = nn.ModuleDict()
        for key in self.low_dim_keys:
            s = self.low_dim_shapes[key]
            if len(s) != 1:
                raise RuntimeError(f"Low-dimensional obs '{key}' must be rank-1, got {s}")
            self.low_dim_mlps[key] = nn.Sequential(
                *create_mlp(s[0], output_dim, net_arch, state_mlp_activation_fn))

        self.n_output_channels = out_channel if cat_on_token else \
            out_channel + output_dim * len(self.low_dim_keys)

        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen = sum(p.numel() for p in self.parameters() if not p.requires_grad)
        cprint(f"[MultiCamImageEncoder] cameras: {self.image_keys}", "yellow")
        cprint(f"[MultiCamImageEncoder] {self.img_h}x{self.img_w} -> "
               f"{self.grid_h}x{self.grid_w} = {self.tokens_per_camera} tokens per "
               f"camera (total {self.tokens_per_camera * self.n_cams})", "yellow")
        # Report what was BUILT, not the flag. With no camera carrying a pose
        # `CameraPoseEmbedding` builds no MLP at all, and printing "shared"
        # there says the opposite of what is running.
        if not any(self.pose_embed.has_pose):
            _mlp_desc = "none built -- each tag is a learned constant per camera"
        else:
            _mlp_desc = "one per camera" if per_camera_pose_mlp else "shared"
        cprint(f"[MultiCamImageEncoder] pose MLP: {_mlp_desc}; "
               f"cameras carrying a pose: {self.pose_embed.has_pose}", "yellow")
        cprint(f"[MultiCamImageEncoder] params: {trainable/1e6:.2f}M trainable, "
               f"{frozen/1e6:.2f}M frozen", "yellow")
        cprint(f"[MultiCamImageEncoder] Final output dim: {self.n_output_channels}", "yellow")

    def forward(self, observations: Dict, eval=False):
        cam_valid = observations["cam_valid"]

        feats = []
        for i, key in enumerate(self.image_keys):
            img = observations[key]
            if img.ndim != 4 or img.shape[1] != 3:
                raise AssertionError(
                    f"{key}: expected [B, 3, H, W], got {tuple(img.shape)}")
            tokens = self.extractor(img)
            tokens = self.pose_embed(
                tokens, i, observations[self.campose_keys[i]], cam_valid[..., i])
            feats.append(tokens)

        pn_feat = torch.cat(feats, dim=1)          # [B, n_cams * tokens, D]

        # One positional encoding per camera, repeated -- the pixel grid is the
        # same for every view. What distinguishes them is the camera tag added
        # above, exactly as the 3D path distinguishes identical camera-frame
        # coordinates from different cameras.
        pe_one = self.pe_layer(self.patch_coords)  # [tokens, D]
        pc_pe = pe_one.unsqueeze(0).repeat(
            pn_feat.shape[0], self.n_cams, 1)      # [B, n_cams * tokens, D]

        low_dim_features = []
        for key in self.low_dim_keys:
            f = self.low_dim_mlps[key](observations[key])
            if self.cat_on_token:
                f = f.unsqueeze(1)
            else:
                f = f.unsqueeze(1).expand(-1, pn_feat.shape[1], -1)
            low_dim_features.append(f)

        features = [pn_feat] + low_dim_features
        final_feat = torch.cat(features, dim=-2 if self.cat_on_token else -1)
        return final_feat, pc_pe

    def output_shape(self):
        return self.n_output_channels
