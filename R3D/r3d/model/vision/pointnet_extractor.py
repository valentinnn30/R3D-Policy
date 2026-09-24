import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import os
import numpy as np

from pytorch3d.ops import sample_farthest_points
from typing import Optional, Dict, Tuple, Union, List, Type
from termcolor import cprint


def create_mlp(
        input_dim: int,
        output_dim: int,
        net_arch: List[int],
        activation_fn: Type[nn.Module] = nn.ReLU,
        squash_output: bool = False,
) -> List[nn.Module]:
    """
    Create a multi layer perceptron (MLP), which is
    a collection of fully-connected layers each followed by an activation function.

    :param input_dim: Dimension of the input vector
    :param output_dim:
    :param net_arch: Architecture of the neural net
        It represents the number of units per layer.
        The length of this list is the number of layers.
    :param activation_fn: The activation function
        to use after each layer.
    :param squash_output: Whether to squash the output using a Tanh
        activation function
    :return:
    """

    if len(net_arch) > 0:
        modules = [nn.Linear(input_dim, net_arch[0]), activation_fn()]
    else:
        modules = []

    for idx in range(len(net_arch) - 1):
        modules.append(nn.Linear(net_arch[idx], net_arch[idx + 1]))
        modules.append(activation_fn())

    if output_dim > 0:
        last_layer_dim = net_arch[-1] if len(net_arch) > 0 else input_dim
        modules.append(nn.Linear(last_layer_dim, output_dim))
    if squash_output:
        modules.append(nn.Tanh())
    return modules


class PointNetEncoderXYZRGB(nn.Module):
    """Encoder for Pointcloud
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int = 1024,
                 use_layernorm: bool = False,
                 final_norm: str = 'none',
                 use_projection: bool = True,
                 **kwargs
                 ):
        """Initialize PointNet encoder for XYZ+RGB point clouds.

        Args:
            in_channels (int): Feature size of input (3 or 6).
            out_channels (int): Output feature dimension.
            use_layernorm (bool): Whether to use LayerNorm after each MLP layer.
            final_norm (str): Normalization after final projection ('layernorm' or 'none').
            use_projection (bool): Whether to apply the final projection layer.
        """
        super().__init__()
        block_channel = [64, 128, 256, 512]
        cprint("pointnet use_layernorm: {}".format(use_layernorm), 'cyan')
        cprint("pointnet use_final_norm: {}".format(final_norm), 'cyan')
        
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, block_channel[0]),
            nn.LayerNorm(block_channel[0]) if use_layernorm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(block_channel[0], block_channel[1]),
            nn.LayerNorm(block_channel[1]) if use_layernorm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(block_channel[1], block_channel[2]),
            nn.LayerNorm(block_channel[2]) if use_layernorm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(block_channel[2], block_channel[3]),
        )

        if final_norm == 'layernorm':
            self.final_projection = nn.Sequential(
                nn.Linear(block_channel[-1], out_channels),
                nn.LayerNorm(out_channels)
            )
        elif final_norm == 'none':
            self.final_projection = nn.Linear(block_channel[-1], out_channels)
        else:
            raise NotImplementedError(f"final_norm: {final_norm}")
         
    def forward(self, x, eval):
        x = self.mlp(x)
        x = torch.max(x, 1)[0]
        x = self.final_projection(x)
        return x
    

class PointNetEncoderXYZ(nn.Module):
    """Encoder for Pointcloud
    """

    def __init__(self,
                 in_channels: int = 3,
                 out_channels: int = 1024,
                 use_layernorm: bool = False,
                 final_norm: str = 'none',
                 use_projection: bool = True,
                 **kwargs
                 ):
        """Initialize PointNet encoder for XYZ-only point clouds.

        Args:
            in_channels (int): Feature size of input (must be 3).
            out_channels (int): Output feature dimension.
            use_layernorm (bool): Whether to use LayerNorm after each MLP layer.
            final_norm (str): Normalization after final projection ('layernorm' or 'none').
            use_projection (bool): Whether to apply the final projection layer.
        """
        super().__init__()
        block_channel = [64, 128, 256]
        cprint("[PointNetEncoderXYZ] use_layernorm: {}".format(use_layernorm), 'cyan')
        cprint("[PointNetEncoderXYZ] use_final_norm: {}".format(final_norm), 'cyan')
        
        assert in_channels == 3, cprint(f"PointNetEncoderXYZ only supports 3 channels, but got {in_channels}", "red")
       
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, block_channel[0]),
            nn.LayerNorm(block_channel[0]) if use_layernorm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(block_channel[0], block_channel[1]),
            nn.LayerNorm(block_channel[1]) if use_layernorm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(block_channel[1], block_channel[2]),
            nn.LayerNorm(block_channel[2]) if use_layernorm else nn.Identity(),
            nn.ReLU(),
        )

        if final_norm == 'layernorm':
            self.final_projection = nn.Sequential(
                nn.Linear(block_channel[-1], out_channels),
                nn.LayerNorm(out_channels)
            )
        elif final_norm == 'none':
            self.final_projection = nn.Linear(block_channel[-1], out_channels)
        else:
            raise NotImplementedError(f"final_norm: {final_norm}")

        self.use_projection = use_projection
        if not use_projection:
            self.final_projection = nn.Identity()
            cprint("[PointNetEncoderXYZ] not use projection", "yellow")

    def forward(self, x, eval):
        x = self.mlp(x)
        x = torch.max(x, 1)[0]
        x = self.final_projection(x)
        return x


class DP3Encoder(nn.Module):
    def __init__(self,
                 observation_space: Dict,
                 img_crop_shape=None,
                 out_channel=256,
                 state_mlp_size=(64, 64), state_mlp_activation_fn=nn.ReLU,
                 pointcloud_encoder_cfg=None,
                 use_pc_color=False,
                 pointnet_type='pointnet',
                 fps_random_config=None,
                 cat_on_token=False
                 ):
        super().__init__()
        self.imagination_key = 'imagin_robot'
        self.point_cloud_key = 'point_cloud'
        self.rgb_image_key = 'image'
        self.n_output_channels = out_channel
        state_mlp_size = (64, pointcloud_encoder_cfg['embed_dim'])

        self.use_imagined_robot = self.imagination_key in observation_space.keys()
        self.point_cloud_shape = observation_space[self.point_cloud_key]
        if self.use_imagined_robot:
            self.imagination_shape = observation_space[self.imagination_key]
        else:
            self.imagination_shape = None

        # `fingertip_anchors` are patch CENTRES for the fingertip tokens, not a
        # feature: without this entry every non-cloud key becomes a low-dim MLP
        # input (and a rank-2 (10, 3) key would raise there).
        self.fingertip_key = 'fingertip_anchors'
        self.use_fingertips = self.fingertip_key in observation_space.keys()
        ignored_obs_keys = {self.point_cloud_key, self.rgb_image_key, self.imagination_key,
                            self.fingertip_key}
        self.low_dim_keys = [key for key in observation_space.keys() if key not in ignored_obs_keys]
        if len(self.low_dim_keys) == 0:
            raise RuntimeError("DP3Encoder requires at least one low-dimensional observation key")
        # xyz + per-point features the encoder is BUILT for (6 = xyz rgb; 9 with
        # the camera one-hot). Anything else arriving is an error, never a trim.
        self.point_channels = int((pointcloud_encoder_cfg or {}).get('point_channels', 6))
        self.low_dim_shapes = {key: observation_space[key] for key in self.low_dim_keys}

        cprint(f"[DP3Encoder] point cloud shape: {self.point_cloud_shape}", "yellow")
        cprint(f"[DP3Encoder] low-dim keys: {self.low_dim_keys}", "yellow")
        cprint(f"[DP3Encoder] low-dim shapes: {self.low_dim_shapes}", "yellow")
        cprint(f"[DP3Encoder] imagination point shape: {self.imagination_shape}", "yellow")

        self.use_pc_color = use_pc_color
        self.pointnet_type = pointnet_type

        feature_mode = pointcloud_encoder_cfg.get('feature_mode', None)
        self.pc_encoder_extract_global_feature = feature_mode != 'pointsam'
        cprint(f"[DP3Encoder] extract_global_feature: {self.pc_encoder_extract_global_feature}", "yellow")

        self.fps_random_config = fps_random_config or {
            'use_random': True,
            'random_start': True,
            'random_noise_scale': 0,
            'shuffle_output': True
        }

        self.cat_on_token = cat_on_token
        pc_output_dim = self.n_output_channels

        if pointnet_type == "pointnet":
            pointnet_cfg = dict(pointcloud_encoder_cfg)
            pointnet_cfg.setdefault('out_channels', out_channel)
            if use_pc_color:
                pointnet_cfg['in_channels'] = 6
                self.extractor = PointNetEncoderXYZRGB(**pointnet_cfg)
            else:
                pointnet_cfg['in_channels'] = 3
                self.extractor = PointNetEncoderXYZ(**pointnet_cfg)
        elif pointnet_type == "uni3d":
            cprint(f"[DP3Encoder] Using Uni3D encoder", "yellow")
            uni3d_config = {
                'pc_model': 'eva02_large_patch14_448',
                'pc_feat_dim': 1024,
                'embed_dim': out_channel,
                'group_size': 32,
                'num_group': 512,
                'patch_dropout': 0.5,
                'drop_path_rate': 0.2,
                'pretrained_pc': None,
                'pc_encoder_dim': 512,
                'use_pretrained_weights': False,
                'pretrained_weights_path': None,
            }
            if pointcloud_encoder_cfg:
                uni3d_config.update(pointcloud_encoder_cfg)
            uni3d_config['fps_random_config'] = self.fps_random_config
            self.extractor = Uni3DPointcloudEncoder(**uni3d_config)
            pc_output_dim = uni3d_config['embed_dim']
        elif pointnet_type == "uni3d_pretrained":
            cprint(f"[DP3Encoder] Using pretrained Uni3D encoder", "yellow")
            uni3d_config = {
                'pc_model': 'eva02_large_patch14_448',
                'pc_feat_dim': 1024,
                'embed_dim': out_channel,
                'group_size': 32,
                'num_group': 512,
                'patch_dropout': 0.5,
                'drop_path_rate': 0.2,
                'pretrained_pc': None,
                'pc_encoder_dim': 512,
                'use_pretrained_weights': True,
                'pretrained_weights_path': 'Uni3D_large/model.pt',
            }
            if pointcloud_encoder_cfg:
                uni3d_config.update(pointcloud_encoder_cfg)
            uni3d_config['fps_random_config'] = self.fps_random_config
            self.extractor = Uni3DPointcloudEncoder(**uni3d_config)
            pc_output_dim = uni3d_config['embed_dim']
        else:
            raise NotImplementedError(f"pointnet_type: {pointnet_type}")

        if len(state_mlp_size) == 0:
            raise RuntimeError(f"State mlp size is empty")
        elif len(state_mlp_size) == 1:
            net_arch = []
        else:
            net_arch = state_mlp_size[:-1]
        output_dim = state_mlp_size[-1]

        self.low_dim_mlps = nn.ModuleDict()
        for key in self.low_dim_keys:
            shape = self.low_dim_shapes[key]
            if len(shape) != 1:
                raise RuntimeError(f"Low-dimensional obs '{key}' must be rank-1, got {shape}")
            self.low_dim_mlps[key] = nn.Sequential(
                *create_mlp(shape[0], output_dim, net_arch, state_mlp_activation_fn)
            )

        if self.cat_on_token:
            self.n_output_channels = pc_output_dim
        else:
            self.n_output_channels = pc_output_dim + output_dim * len(self.low_dim_keys)

        cprint(f"[DP3Encoder] Final output dim: {self.n_output_channels}", "yellow")

    def forward(self, observations: Dict, eval=False) -> torch.Tensor:
        points = observations[self.point_cloud_key]
        assert len(points.shape) == 3, cprint(f"point cloud shape: {points.shape}, length should be 3", "red")
        if self.use_imagined_robot:
            img_points = observations[self.imagination_key][..., :points.shape[-1]]
            points = torch.concat([points, img_points], dim=1)

        if self.pointnet_type in ["uni3d", "uni3d_pretrained"]:
            if points.shape[-1] == 3:
                colors = torch.zeros_like(points)
                points = torch.cat([points, colors], dim=-1)
            elif points.shape[-1] != self.point_channels:
                # Used to be `points[..., :6]` for anything wider -- which would
                # silently drop a camera one-hot. Width is a contract now.
                raise ValueError(
                    f"point cloud has {points.shape[-1]} channels, encoder built for "
                    f"{self.point_channels} (pointcloud_encoder_cfg.point_channels)")

        extra = {}
        if self.use_fingertips:
            extra["anchors"] = observations[self.fingertip_key]
        if not self.pc_encoder_extract_global_feature:
            pn_feat, pc_pe = self.extractor(points, eval, **extra)
        else:
            pn_feat = self.extractor(points, eval, **extra)

        low_dim_features = []
        for key in self.low_dim_keys:
            low_dim_value = observations[key]
            low_dim_feat = self.low_dim_mlps[key](low_dim_value)
            if not self.pc_encoder_extract_global_feature:
                if self.cat_on_token:
                    low_dim_feat = low_dim_feat.unsqueeze(1)
                else:
                    low_dim_feat = low_dim_feat.unsqueeze(1).expand(-1, pn_feat.shape[1], -1)
            low_dim_features.append(low_dim_feat)

        features = [pn_feat] + low_dim_features
        if self.cat_on_token:
            final_feat = torch.cat(features, dim=-2)
        else:
            final_feat = torch.cat(features, dim=-1)
        if not self.pc_encoder_extract_global_feature:
            return final_feat, pc_pe
        return final_feat

    def output_shape(self):
        return self.n_output_channels


# =============================================================================
# PointSAM encoder components (adapted from Uni3D)
# =============================================================================

def fps(data, number, use_random=True, random_start=True, random_noise_scale=0, shuffle_output=True):
    '''
    Enhanced FPS with randomness options
    Args:
        data: B N 3 (or more channels)
        number: int, number of points to sample
        use_random: bool, whether to enable randomness
        random_start: bool, whether to use random starting point
        random_noise_scale: float, scale of random noise added to distances
        shuffle_output: bool, whether to randomly shuffle the output order
    '''
    xyz_coordinates = data[:, :, :3]
    B, N, _ = xyz_coordinates.shape
    
    if not use_random:
        # Original deterministic FPS
        _, fps_idx = sample_farthest_points(xyz_coordinates, K=number)
    else:
        # Enhanced FPS with randomness
        if random_start:
            # Randomly select starting points for each batch
            start_indices = torch.randint(0, N, (B,), device=data.device)
            
            # Create modified coordinates with random starting points moved to front
            modified_xyz = xyz_coordinates.clone()
            for b in range(B):
                start_idx = start_indices[b]
                # Swap the randomly selected point to the first position
                modified_xyz[b, [0, start_idx]] = modified_xyz[b, [start_idx, 0]]
        else:
            modified_xyz = xyz_coordinates
        
        if random_noise_scale > 0:
            # Add small random noise to coordinates for FPS computation
            noise = torch.randn_like(modified_xyz) * random_noise_scale
            noisy_xyz = modified_xyz + noise
        else:
            noisy_xyz = modified_xyz
        
        # Perform FPS on modified/noisy coordinates
        _, fps_idx = sample_farthest_points(noisy_xyz, K=number)
        
        # If we used random start, we need to map back the indices
        if random_start:
            for b in range(B):
                start_idx = start_indices[b]
                # Map indices back to original positions
                mask_0 = fps_idx[b] == 0
                mask_start = fps_idx[b] == start_idx
                fps_idx[b][mask_0] = start_idx
                fps_idx[b][mask_start] = 0
        
        if shuffle_output:
            # Randomly shuffle the order of selected indices
            for b in range(B):
                perm = torch.randperm(number, device=data.device)
                fps_idx[b] = fps_idx[b][perm]
    
    # Gather the selected points using the (possibly randomized) indices
    fps_data = torch.gather(
        data, 1, fps_idx.unsqueeze(-1).long().expand(-1, -1, data.shape[-1]))
    
    return fps_data

def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist


def random_point_dropout(batch_pc, max_dropout_ratio=0.875):
    ''' batch_pc: BxNx3 '''
    B, N, _ = batch_pc.shape
    result = torch.clone(batch_pc)
    for b in range(B):
        dropout_ratio = torch.rand(1).item() * max_dropout_ratio  # 0 ~ 0.875
        drop_idx = torch.where(torch.rand(N) <= dropout_ratio)[0]
        if len(drop_idx) > 0:
            result[b, drop_idx, :] = batch_pc[b, 0, :].unsqueeze(0)  # set to the first point
    return result


class PatchDropout(nn.Module):
    """
    Patch dropout for Uni3D
    https://arxiv.org/abs/2212.00794
    """
    def __init__(self, prob, exclude_first_token=True):
        super().__init__()
        assert 0 <= prob < 1.
        self.prob = prob
        self.exclude_first_token = exclude_first_token  # exclude CLS token

    def forward(self, x):
        if self.exclude_first_token:
            cls_tokens, x = x[:, :1], x[:, 1:]
        else:
            cls_tokens = torch.jit.annotate(torch.Tensor, x[:, :1])

        batch = x.size()[0]
        num_tokens = x.size()[1]

        batch_indices = torch.arange(batch)
        batch_indices = batch_indices[..., None]

        keep_prob = 1 - self.prob
        num_patches_keep = max(1, int(num_tokens * keep_prob))

        rand = torch.randn(batch, num_tokens)
        patch_indices_keep = rand.topk(num_patches_keep, dim=-1).indices

        x = x[batch_indices, patch_indices_keep]

        if self.exclude_first_token:
            x = torch.cat((cls_tokens, x), dim=1)

        return x

class KNNGrouper(nn.Module):
    """Group points based on K nearest neighbors.

    A number of points are sampled as centers by farthest point sampling (FPS).
    Each group is formed by the center and its k nearest neighbors.
    """

    def __init__(self, num_groups, group_size, radius=None, centralize_features=False, fps_random_config=None):
        super().__init__()
        self.num_groups = num_groups
        self.group_size = group_size
        self.radius = radius
        self.centralize_features = centralize_features
        self.fps_random_config = fps_random_config or {}
        cprint(f"[Group] FPS randomness config: {fps_random_config}", "cyan")

    def forward(self, xyz: torch.Tensor, features: torch.Tensor, use_fps=True,
                num_groups=None):
        """
        Args:
            xyz: [B, N, 3]. Input point clouds.
            features: [B, N, C]. Point features.
            use_fps: bool. Whether to use farthest point sampling.
                If not, `xyz` should already be sampled by FPS.

        Returns:
            dict: {
                features: [B, G, K, 3 + C]. Group features.
                centers: [B, G, 3]. Group centers.
                knn_idx: [B, G, K]. The indices of k nearest neighbors.
            }
        """
        batch_size, num_points, _ = xyz.shape
        # `num_groups` overrides the constructed default for this call only.
        # Nothing downstream has a weight whose shape depends on it -- the patch
        # encoder is a shared PointNet applied per group, the positional embed
        # runs per centre, and the ViT is sequence-length agnostic (this code
        # bypasses timm's own fixed positional table). So the unfused pipeline
        # can give each camera a different token budget for free.
        G = self.num_groups if num_groups is None else int(num_groups)
        with torch.no_grad():
            centers = fps(xyz, G, **self.fps_random_config) # B G 3
            _, knn_idx = knn_points(centers, xyz, self.group_size)  # [B, G, K]

        batch_offset = torch.arange(batch_size, device=xyz.device) * num_points
        batch_offset = batch_offset.reshape(-1, 1, 1)
        knn_idx_flat = (knn_idx + batch_offset).reshape(-1)  # [B * G * K]

        nbr_xyz = xyz.reshape(-1, 3)[knn_idx_flat]
        nbr_xyz = nbr_xyz.reshape(batch_size, G, self.group_size, 3)
        nbr_xyz = nbr_xyz - centers.unsqueeze(2)  # [B, G, K, 3]
        # NOTE: Follow PointNext to normalize the relative position
        if self.radius is not None:
            nbr_xyz = nbr_xyz / self.radius

        nbr_feats = features.reshape(-1, features.shape[-1])[knn_idx_flat]
        nbr_feats = nbr_feats.reshape(
            batch_size, G, self.group_size, features.shape[-1]
        )

        group_feats = torch.cat([nbr_xyz, nbr_feats], dim=-1)
        return dict(
            features=group_feats, centers=centers, knn_idx=knn_idx
        )

class PatchEncoder(nn.Module):
    """Encode point patches following the PointNet structure for segmentation."""

    def __init__(self, in_channels, out_channels, hidden_dims: list[int]):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # NOTE: The original Uni3D implementation uses BatchNorm1d, while we use LayerNorm.
        self.conv1 = nn.Sequential(
            nn.Linear(in_channels, hidden_dims[0]),
            nn.LayerNorm(hidden_dims[0]),
            nn.GELU(),
            nn.Linear(hidden_dims[0], hidden_dims[0]),
        )
        self.conv2 = nn.Sequential(
            nn.Linear(hidden_dims[0] * 2, hidden_dims[1]),
            nn.LayerNorm(hidden_dims[1]),
            nn.GELU(),
            nn.Linear(hidden_dims[1], out_channels),
        )

    def forward(self, point_patches: torch.Tensor):
        # point_patches: [B, L, K, C_in]
        x = self.conv1(point_patches)
        y = torch.max(x, dim=-2, keepdim=True).values
        x = torch.cat([y.expand_as(x), x], dim=-1)
        x = self.conv2(x)  # [B, L, K, C_out]
        y = torch.max(x, dim=-2).values  # [B, L, C_out]
        return y

class PatchEmbed(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        num_patches,
        patch_size,
        radius: float = None,
        centralize_features=False,
        fps_random_config=None,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.grouper = KNNGrouper(
            num_patches,
            patch_size,
            radius=radius,
            centralize_features=centralize_features,
            fps_random_config=fps_random_config
        )

        self.patch_encoder = PatchEncoder(in_channels, out_channels, [128, 512])
        self.fps_random_config = fps_random_config or {}

    def forward(self, coords: torch.Tensor, features: torch.Tensor,
                num_groups=None):
        patches = self.grouper(coords, features, num_groups=num_groups)
        patch_features = patches["features"]  # [B, L, K, C_in]
        x = self.patch_encoder(patch_features)
        patches["embeddings"] = x
        return patches


def knn_points(
    query: torch.Tensor,
    key: torch.Tensor,
    k: int,
    sorted: bool = False,
    transpose: bool = False,
):
    """Compute k nearest neighbors.

    Args:
        query: [B, N1, D], query points. [B, D, N1] if @transpose is True.
        key:  [B, N2, D], key points. [B, D, N2] if @transpose is True.
        k: the number of nearest neighbors.
        sorted: whether to sort the results
        transpose: whether to transpose the last two dimensions.

    Returns:
        torch.Tensor: [B, N1, K], distances to the k nearest neighbors in the key.
        torch.Tensor: [B, N1, K], indices of the k nearest neighbors in the key.
    """
    if transpose:
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
    # Compute pairwise distances, [B, N1, N2]
    distance = torch.cdist(query, key)
    if k == 1:
        knn_dist, knn_ind = torch.min(distance, dim=2, keepdim=True)
    else:
        knn_dist, knn_ind = torch.topk(distance, k, dim=2, largest=False, sorted=sorted)
    return knn_dist, knn_ind

class PositionEmbeddingRandom(nn.Module):
    """
    Positional encoding using random spatial frequencies.
    """

    def __init__(self, num_pos_feats: int = 64, scale: Optional[float] = None) -> None:
        super().__init__()
        if scale is None or scale <= 0.0:
            scale = 1.0
        self.register_buffer(
            "positional_encoding_gaussian_matrix",
            scale * torch.randn((3, num_pos_feats)),
        )

    def _pe_encoding(self, coords: torch.Tensor) -> torch.Tensor:
        """Positionally encode points that are normalized to [-1,1]."""
        # assuming coords are in [-1, 1] and have d_1 x ... x d_n x D shape
        coords = coords @ self.positional_encoding_gaussian_matrix
        # TODO: Why using 2 * np.pi?
        coords = 2 * np.pi * coords
        # outputs d_1 x ... x d_n x C shape
        return torch.cat([torch.sin(coords), torch.cos(coords)], dim=-1)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Args:
            coords: shape (..., coord_dim), normalized coordinates in [-1, 1].

        Returns:
            torch.Tensor: shape (..., num_pos_feats), positional encoding.
        """
        if (coords < -1 - 1e-6).any() or (coords > 1 + 1e-6).any():
            print("Bounds: ", (coords.min(), coords.max()))
            raise ValueError(f"Input coordinates must be normalized to [-1, 1].")
        # TODO: whether to convert to float?
        return self._pe_encoding(coords)


class Uni3DPointcloudEncoder(nn.Module):
    """
    Uni3D point cloud encoder.
    Supports both pretrained weight loading and training from scratch.
    """
    def __init__(self,
                 pc_model='eva02_large_patch14_448',
                 pc_feat_dim=1024,
                 embed_dim=1024,
                 group_size=32,
                 num_group=512,
                 patch_dropout=0.5,
                 drop_path_rate=0.2,
                 pretrained_pc=None,
                 pc_encoder_dim=512,
                 use_pretrained_weights=False,
                 pretrained_weights_path=None,
                 normalization_type="batch_norm",
                 feature_mode="pointsam",
                 extract_global_feature=True,
                 fps_random_config=None,
                 point_channels=6,
                 fingertip_tokens=None,
                 **kwargs):
        super().__init__()

        # vit backbone
        self.transformer = timm.create_model(pc_model, checkpoint_path=pretrained_pc, drop_path_rate=drop_path_rate)
        self.transformer_dim = self.transformer.embed_dim
        self.embed_dim = embed_dim
        self.num_group = num_group
        self.group_size = group_size
        self.use_pretrained_weights = use_pretrained_weights

        # `point_channels` = xyz + per-point features. 6 (xyz rgb) is the
        # pretrained layout; 9 adds the per-camera one-hot. The group features
        # the patch encoder sees are [rel_xyz, feats], so its input width is
        # point_channels either way. Extra input columns are ZERO-initialised
        # (see _zero_extra_input_columns) so step 0 equals the 6-channel model.
        if point_channels < 6:
            raise ValueError(f"point_channels must be >= 6 (xyz rgb ...), got {point_channels}")
        self.point_channels = int(point_channels)
        self.patch_embed = PatchEmbed(in_channels=self.point_channels, out_channels=512, num_patches=num_group, patch_size=group_size, fps_random_config=fps_random_config)
        self._zero_extra_input_columns()

        # 7 = xyz + rgb + dist
        self.pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, self.transformer_dim)
        )

        self.extract_global_feature = feature_mode != 'pointsam'

        # for pointsam output pc_pe
        self.pe_layer = PositionEmbeddingRandom(embed_dim // 2)

        # Patch dropout
        self.patch_dropout = PatchDropout(patch_dropout, exclude_first_token=(feature_mode=="cls")) if patch_dropout > 0. else nn.Identity()
        # Project transformer output to embedding dim
        self.out_proj = nn.Linear(self.transformer_dim, self.embed_dim)
        self.patch_proj = nn.Linear(self.patch_embed.out_channels, self.transformer_dim)
        self.feature_mode = feature_mode

        if self.feature_mode == "cls":
            cprint(f"[Uni3DPointcloudEncoder] use cls token", "red")

            self.cls_token = nn.Parameter(torch.zeros(1, 1, self.transformer_dim))
            self.cls_pos = nn.Parameter(torch.randn(1, 1, self.transformer_dim))
        elif self.feature_mode == "max_pooling":
            cprint(f"[Uni3DPointcloudEncoder] use max pooling", "red")
        else:  # pointsam
            cprint(f"[Uni3DPointcloudEncoder] use pointsam, do not extract global feature", "red")

        # Load pretrained weights if specified
        if use_pretrained_weights:
            self._load_pretrained_weights_selective(pretrained_weights_path, normalization_type)
        else:
            cprint(f"[Uni3DPointcloudEncoder] Using random initialization (training from scratch)", "red")

        self.fingertip_cfg = None
        if fingertip_tokens and fingertip_tokens.get("enabled", False):
            self._init_fingertip_tokens(fingertip_tokens, patch_dropout, feature_mode)

    # -------------------------------------------------------------------------
    # Extra per-point input channels (camera one-hot)
    # -------------------------------------------------------------------------

    _CONV1_KEY = "patch_embed.patch_encoder.conv1.0.weight"

    def _zero_extra_input_columns(self):
        """Zero the patch encoder's input columns beyond xyz+rgb, so the extra
        channels contribute exactly nothing until training moves them."""
        if self.point_channels > 6:
            with torch.no_grad():
                self.patch_embed.patch_encoder.conv1[0].weight[:, 6:] = 0.0

    # -------------------------------------------------------------------------
    # Fingertip tokens (see ~/ros2_ws/fingertip_tokens_design_rationale.txt)
    # -------------------------------------------------------------------------

    N_HANDS, N_FINGERS = 2, 5
    TAG_SITES = ("vit", "head")

    def _init_fingertip_tokens(self, cfg, patch_dropout, feature_mode):
        # Guards: each of these would silently break the token bookkeeping.
        if feature_mode != "pointsam":
            raise ValueError("fingertip tokens need feature_mode 'pointsam' (per-token output)")
        if patch_dropout and patch_dropout > 0:
            raise ValueError("fingertip tokens need patch_dropout 0: PatchDropout keeps a "
                             "random subset of tokens, which would drop and reorder them")
        tag_at = tuple(cfg.get("tag_at", self.TAG_SITES))
        if not set(tag_at) <= set(self.TAG_SITES) or not tag_at:
            raise ValueError(f"fingertip_tokens.tag_at must be a non-empty subset of "
                             f"{self.TAG_SITES}, got {tag_at}")
        self.fingertip_tag_at = tag_at
        self.fingertip_radius_m = float(cfg.get("radius_m", 0.03))
        # HOW the K points of a patch are drawn from inside its radius:
        #   nearest -- the K nearest (the reference; what every FPS patch does)
        #   fps     -- farthest-point sampling over up to `fps_candidates`
        #              nearest in-radius points, in METRES, starting at the
        #              point nearest the centre (deterministic: training and
        #              inference draw identically). Spreads the K points over
        #              the whole ball instead of its dense core.
        self.ft_sampling = str(cfg.get("sampling", "nearest"))
        if self.ft_sampling not in ("nearest", "fps"):
            raise ValueError(f"fingertip_tokens.sampling must be nearest | fps, "
                             f"got {self.ft_sampling!r}")
        self.ft_fps_candidates = int(cfg.get("fps_candidates", 512))
        # fps only: start the FPS at a RANDOM in-radius point while training
        # (eval=False) -- a fresh, still evenly spread draw of the same ball
        # every step, like the scene tokens' fps_random_config. eval=True
        # (predict_action, i.e. inference) always starts at the nearest point.
        self.ft_fps_random_start = bool(cfg.get("fps_random_start", False))
        if self.ft_fps_random_start and self.ft_sampling != "fps":
            raise ValueError("fingertip_tokens.fps_random_start needs sampling: fps")
        if self.ft_fps_candidates < self.group_size:
            raise ValueError("fingertip_tokens.fps_candidates must be >= group_size")
        # `anchors` absent: the reference run -- 10 tips, one radius, no type
        # tag, state_dict unchanged (old checkpoints load). Present: an anchor
        # SET (real_preprocess.ANCHOR_TYPES), in anchor_layout order.
        self.ft_anchor_spec = cfg.get("anchors")
        if self.ft_anchor_spec is None:
            self.n_tips = self.N_HANDS * self.N_FINGERS
            idx = torch.arange(self.n_tips)
            # anchor order [L thumb..pinky, R thumb..pinky] (real_preprocess.HandFK)
            hand_idx, finger_idx = idx // self.N_FINGERS, idx % self.N_FINGERS
            self.ft_type_names = None
        else:
            from r3d.common.real_preprocess import (
                ANCHOR_TYPES, normalize_anchor_spec, anchor_layout)
            entries = normalize_anchor_spec(
                [dict(e) for e in self.ft_anchor_spec], self.fingertip_radius_m)
            lay = anchor_layout(entries)
            self.n_tips = len(lay)
            hand_idx = torch.tensor([h for h, _, _, _ in lay])
            finger_idx = torch.tensor([f for _, f, _, _ in lay])
            self.ft_type_names = ANCHOR_TYPES
            self.register_buffer("ft_type_idx", torch.tensor([t for _, _, t, _ in lay]),
                                 persistent=False)
            self.register_buffer("ft_radius", torch.tensor([r for *_, r in lay]),
                                 persistent=False)
        self.register_buffer("ft_hand_idx", hand_idx, persistent=False)
        self.register_buffer("ft_finger_idx", finger_idx, persistent=False)

        Dv, Dh = self.transformer_dim, self.embed_dim
        # All zero-init: at step 0 the tags change nothing; training decides.
        self.ft_empty = nn.Parameter(torch.zeros(Dv))
        if "vit" in tag_at:
            self.ft_tip_vit = nn.Parameter(torch.zeros(Dv))
            self.ft_hand_vit = nn.Parameter(torch.zeros(self.N_HANDS, Dv))
            self.ft_finger_vit = nn.Parameter(torch.zeros(self.N_FINGERS, Dv))
        if "head" in tag_at:
            self.ft_tip_head = nn.Parameter(torch.zeros(Dh))
            self.ft_hand_head = nn.Parameter(torch.zeros(self.N_HANDS, Dh))
            self.ft_finger_head = nn.Parameter(torch.zeros(self.N_FINGERS, Dh))
        # Anchor TYPE (tip / pad cluster / axial / pad_front / pinch): without
        # it a pinch token and the tip token of its partner finger would carry
        # identical tags. Zero-init like the rest; only in an anchor-set run.
        if self.ft_type_names is not None:
            n_types = len(self.ft_type_names)
            if "vit" in tag_at:
                self.ft_type_vit = nn.Parameter(torch.zeros(n_types, Dv))
            if "head" in tag_at:
                self.ft_type_head = nn.Parameter(torch.zeros(n_types, Dh))

        # Metres per normalized unit, per axis. The encoder sees NORMALIZED
        # coordinates and the workspace normalizer is anisotropic (0.65/1.0/
        # 0.6 m spans), so a radius in normalized space would be an ellipsoid.
        # DP3.set_normalizer fills this; it is saved in the state_dict, so
        # inference reads the same value. The flag makes a forgotten fill loud.
        self.register_buffer("xyz_half_range", torch.ones(3))
        self.register_buffer("xyz_half_range_set", torch.zeros((), dtype=torch.bool))
        self.fingertip_cfg = dict(cfg)
        cprint(f"[Uni3DPointcloudEncoder] fingertip tokens: {self.n_tips}, "
               f"radius {self.fingertip_radius_m * 100:.1f} cm, tag at {tag_at}, "
               f"sampling {self.ft_sampling}"
               f"{' (random start in training)' if self.ft_fps_random_start else ''}", "red")

    def set_xyz_half_range(self, half_range):
        half_range = torch.as_tensor(half_range, dtype=torch.float32).flatten()
        if half_range.shape != (3,) or not torch.all(half_range > 0):
            raise ValueError(f"xyz_half_range must be 3 positive values, got {half_range}")
        self.xyz_half_range.copy_(half_range.to(self.xyz_half_range.device))
        self.xyz_half_range_set.fill_(True)

    def _tag(self, site):
        tip = getattr(self, f"ft_tip_{site}")
        hand = getattr(self, f"ft_hand_{site}")[self.ft_hand_idx]
        finger = getattr(self, f"ft_finger_{site}")[self.ft_finger_idx]
        tag = tip + hand + finger  # [n_tips, D]
        if self.ft_type_names is not None:
            tag = tag + getattr(self, f"ft_type_{site}")[self.ft_type_idx]
        return tag

    def _fingertip_patches(self, pts, feats, anchors, eval=True):
        """Anchored radius patches.

        pts [B, N, 3] and anchors [B, T, 3] are NORMALIZED; feats [B, N, C].
        Returns (embeddings [B, T, 512], centers [B, T, 3] clamped to [-1, 1],
        empty [B, T] bool).

        * neighbours are chosen by METRIC distance (radius_m), never a point
          beyond it: plain kNN always returns K points and would fill an
          occluded fingertip with palm / table;
        * which K: the nearest, or FPS-spread over the ball (`sampling`);
        * empty slots are refilled by cycling the in-radius neighbours --
          the patch encoder max-pools, so duplicates are exactly neutral;
        * features are the same NORMALIZED relative coords + channels an FPS
          patch gets, so the pretrained patch encoder sees familiar input;
        * a patch with no point in radius, or an anchor outside the workspace
          box, is flagged empty (the caller substitutes a learned token).
        """
        if not bool(self.xyz_half_range_set):
            raise RuntimeError(
                "fingertip tokens: xyz_half_range was never set. DP3.set_normalizer "
                "fills it from the point_cloud normalizer; without it the 3 cm "
                "radius would be measured in normalized units.")
        B, N, _ = pts.shape
        K = self.group_size
        inside = (anchors.abs() <= 1.0).all(-1)                    # [B, T]
        centers = anchors.clamp(-1.0, 1.0)
        hr = self.xyz_half_range.to(pts.dtype)
        T = anchors.shape[1]
        with torch.no_grad():
            # nearest: exactly K candidates; fps: a larger pool to spread over
            Kc = K if self.ft_sampling == "nearest" else min(self.ft_fps_candidates, N)
            dist, idx = knn_points(anchors * hr, pts * hr, Kc, sorted=True)  # metres
            # per-anchor radius in an anchor-set run, else the one scalar
            radius = (self.ft_radius.to(dist.dtype)[None, :, None]
                      if self.ft_type_names is not None else self.fingertip_radius_m)
            valid = dist <= radius                                 # sorted -> prefix
            n_valid = valid.sum(-1)                                # [B, T]
            slot = torch.arange(K, device=pts.device).expand(B, T, K)
            if self.ft_sampling == "nearest":
                fill = slot % n_valid.clamp(min=1).unsqueeze(-1)
                idx = torch.gather(idx, 2, fill)
            else:
                # FPS over the in-radius prefix of each candidate list, in
                # metres. Start at candidate 0 (the nearest point) -> the draw
                # is deterministic -- or, training with fps_random_start, at a
                # random in-radius point. Short patches come back -1 padded
                # after min(n, K) picks; refill by cycling, as above.
                cand = torch.gather((pts * hr).unsqueeze(1).expand(B, T, N, 3), 2,
                                    idx.unsqueeze(-1).expand(-1, -1, -1, 3))
                _, pick = sample_farthest_points(
                    cand.reshape(B * T, Kc, 3).float(),
                    lengths=n_valid.reshape(-1).clamp(min=1), K=K,
                    random_start_point=self.ft_fps_random_start and not eval)
                n_pick = n_valid.clamp(min=1, max=K).reshape(-1, 1)
                pick = torch.gather(pick, 1, slot.reshape(B * T, K) % n_pick)
                idx = torch.gather(idx, 2, pick.reshape(B, T, K))
        empty = (n_valid == 0) | ~inside

        flat = idx.reshape(B, -1)                                  # [B, T*K]
        nbr_xyz = torch.gather(pts, 1, flat.unsqueeze(-1).expand(-1, -1, 3))
        nbr_feat = torch.gather(feats, 1, flat.unsqueeze(-1).expand(-1, -1, feats.shape[-1]))
        nbr_xyz = nbr_xyz.reshape(B, T, K, 3) - centers.unsqueeze(2)
        nbr_feat = nbr_feat.reshape(B, T, K, -1)
        emb = self.patch_embed.patch_encoder(torch.cat([nbr_xyz, nbr_feat], dim=-1))
        return emb, centers, empty

    def _load_pretrained_weights_selective(self, pretrained_weights_path, normalization_type):
        """
        Selectively load pretrained weights based on normalization_type.

        Args:
            pretrained_weights_path: Path to pretrained weights
            normalization_type: Normalization type ("batch_norm", "layer_norm", "none")
        """
        load_weight_path = pretrained_weights_path
        if not os.path.exists(load_weight_path):
            cprint(f"[Uni3DPointcloudEncoder] Pretrained weights file not found: {load_weight_path}", "red")
            return

        # Load pretrained weights
        from safetensors.torch import load_file
        checkpoint = load_file(os.path.join(load_weight_path, "model.safetensors"))
        # Remap key names
        processed_state_dict = {}
        for key in list(checkpoint.keys()):
            if key.startswith('pc_encoder.'):
                new_key = key.replace('pc_encoder.', '')
                processed_state_dict[new_key] = checkpoint[key]
        # Extra per-point channels (camera one-hot): the pretrained first layer
        # takes 6 inputs. strict=False does NOT forgive a shape mismatch, so
        # widen it here: pretrained weights in the first 6 columns, zeros in the
        # rest -- the model is then exactly the 6-channel one at step 0.
        w = processed_state_dict.get(self._CONV1_KEY)
        if w is not None and self.point_channels > w.shape[1]:
            cur = self.state_dict()[self._CONV1_KEY]
            widened = torch.zeros_like(cur)
            widened[:, :w.shape[1]] = w.to(widened.dtype)
            processed_state_dict[self._CONV1_KEY] = widened
            cprint(f"  widened {self._CONV1_KEY} {tuple(w.shape)} -> {tuple(cur.shape)} "
                   "(new columns zero)", "yellow")
        missing_keys, unexpected_keys = self.load_state_dict(processed_state_dict, strict=False)
        cprint(f"  Missing keys: {missing_keys}", "yellow")
        cprint(f"  Unexpected keys: {unexpected_keys}", "yellow")

        cprint(f"[Uni3DPointcloudEncoder] Pretrained weights loaded: {load_weight_path}", "red")

    def forward(self, pcd, eval, num_groups=None, anchors=None):
        """`anchors` [B, 10, 3] (normalized) -- fingertip-token run only.

        The fingertip tokens are APPENDED after the FPS tokens of this frame, so
        the per-frame token block stays contiguous and the action head's
        per-obs-step temporal encoding (num_tokens = L // n_obs_steps) still
        lines up.
        """
        if (anchors is not None) != (self.fingertip_cfg is not None):
            raise ValueError("fingertip anchors given to an encoder without fingertip "
                             "tokens, or missing for one that has them")
        if anchors is not None and anchors.shape[1] != self.n_tips:
            raise ValueError(f"encoder built for {self.n_tips} fingertip anchors, "
                             f"got {anchors.shape[1]} (anchor spec mismatch)")
        if pcd.shape[-1] != self.point_channels:
            raise ValueError(f"encoder built for {self.point_channels} point channels, "
                             f"got {pcd.shape[-1]}")
        # Apply point cloud dropout (data augmentation)
        if not eval:
            pcd = random_point_dropout(pcd, max_dropout_ratio=0.8)

        pts = pcd[..., :3].contiguous()
        colors = pcd[..., 3:].contiguous()
        # Group points into patches and get embeddings
        patches = self.patch_embed(pts, colors, num_groups=num_groups)
        if isinstance(patches, list):
            patch_embed = patches[-1]["embeddings"]
            centers = patches[-1]["centers"]
        else:
            patch_embed = patches["embeddings"]  # [B, L, D]
            centers = patches["centers"]  # [B, L, 3]
        patch_embed = self.patch_proj(patch_embed)

        if anchors is not None:
            # After the dropout on purpose: a dropped point is moved onto point
            # 0, far away, so the metric radius test excludes it by itself.
            ft_emb, ft_centers, ft_empty = self._fingertip_patches(
                pts, colors, anchors.to(pts.dtype), eval=eval)
            ft_emb = self.patch_proj(ft_emb)
            ft_emb = torch.where(ft_empty.unsqueeze(-1),
                                 self.ft_empty.to(ft_emb.dtype).expand_as(ft_emb), ft_emb)
            if "vit" in self.fingertip_tag_at:
                ft_emb = ft_emb + self._tag("vit").to(ft_emb.dtype)
            patch_embed = torch.cat([patch_embed, ft_emb], dim=1)
            centers = torch.cat([centers, ft_centers], dim=1)

        # Add positional embedding
        pos_embed = self.pos_embed(centers)

        if self.feature_mode == "cls":

            # prepare cls
            cls_tokens = self.cls_token.expand(patch_embed.size(0), -1, -1)  
            cls_pos = self.cls_pos.expand(pos_embed.size(0), -1, -1) 

            # final input
            patch_embed = torch.cat((cls_tokens, patch_embed), dim=1)
            pos_embed = torch.cat((cls_pos, pos_embed), dim=1)
        
        x = patch_embed + pos_embed
        # patch dropout
        if not eval:
            x = self.patch_dropout(x)
            x = self.transformer.pos_drop(x)

        for block in self.transformer.blocks:
            x = block(x)

        if self.extract_global_feature:

            # Extract features based on whether CLS token is used
            if self.feature_mode == "cls":
                # Use CLS token (first token) for classification
                x = self.transformer.norm(x[:, 0, :])
            elif self.feature_mode == "max_pooling":
                # Use global max pooling over all patch tokens
                x = self.transformer.norm(torch.max(x, dim=1)[0])
        else: 
            # pointsam, do not extract global feature
            x = self.transformer.norm(x)
        
        x = self.transformer.fc_norm(x)
        x = self.out_proj(x)

        if not self.extract_global_feature:
            pc_pe = self.pe_layer(centers)
            if anchors is not None and "head" in self.fingertip_tag_at:
                # the action head adds pc_pe to its point-token KEYS in every
                # cross-attention layer, so the tag is visible in all of them
                tag = self._tag("head").to(pc_pe.dtype)
                pc_pe = torch.cat([pc_pe[:, :-self.n_tips], pc_pe[:, -self.n_tips:] + tag], dim=1)
            return x, pc_pe
        else:
            return x

# =============================================================================
# Unfused multi-camera encoder -- merge in FEATURE space, not in point space
# =============================================================================


class CameraPoseEmbedding(nn.Module):
    """Tags a camera's tokens with where that camera is, and which one it is.

    Without this, a patch at (0.1, 0.0, 0.5) in the left wrist camera and one at
    (0.1, 0.0, 0.5) in the top camera are indistinguishable, because each
    camera's geometry arrives in its own frame.

    Two separate signals, deliberately not merged into one MLP input:

    * `cam_embed` -- identity. Lets the model learn per-camera noise and
      reliability characteristics, and (under pose_source='proprio') is the
      whole of a static camera's pose information, since that transform is a
      constant.
    * `pose_mlp` -- the 9-vector [translation(3), 6D rotation(6)]. 6D rather
      than a quaternion because every parameterisation of SO(3) in <=4
      dimensions is discontinuous, which is hard to embed smoothly.

    `absent` replaces a camera's tokens entirely when it contributed nothing
    this frame (a wrist camera looking away from the table). Added rather than
    masking the attention, which would mean threading a key-padding mask
    through OneWayTransformer -- worth doing later if empty cameras turn out to
    matter, but they are measured at 2 of 30 episodes.

    `per_camera_mlp` gives each camera its OWN pose MLP instead of one shared
    across all of them, and it is not cosmetic -- it decides what the tag can
    represent:

    * Shared: the MLP sees only `pose9` and cannot know which camera sent it,
      so it computes one function for all. The per-camera freedom is then just
      `cam_embed`, a constant OFFSET ON THE OUTPUT. Composing a rigid transform
      is not an addition in embedding space, so `f(pose) + e_i` cannot equal
      `f(pose @ T_i)`; the decoder has to disentangle the sum instead.
    * Per camera: the tag becomes a genuinely per-camera FUNCTION of pose, so a
      constant right-composition -- the unknown flange->camera mount -- folds
      into the learned weights.

    That distinction is the whole content of Ablation A' (`pose_source:
    proprio`), which tags each wrist camera with its ARM's recorded EE pose and
    asks the network to learn the constant mount offset itself. With a shared
    MLP that ablation is testing a hypothesis the architecture cannot cleanly
    express. Run B (`extrinsic`) does not need it -- the calibrated mount is
    already inside the 4x4 -- and keeps the shared MLP so its parameter count
    is unchanged against the runs already done.

    `has_pose` marks the cameras whose `pose9` actually carries information. A
    camera without one gets NO pose MLP at all, only its identity embedding.
    That is the static top camera under `proprio`: it rides on no arm, the
    dataset hands it `zeros(9)`, and pushing a constant through an MLP to
    obtain a constant is a way to spend parameters on nothing. Under
    `pose_source: none` no camera has a pose and the module degenerates to
    identity embeddings, which is exactly what that control means.
    """

    def __init__(self, n_cams: int, embed_dim: int, pose_dim: int = 9,
                 per_camera_mlp: bool = False, has_pose=None):
        super().__init__()
        if has_pose is None:
            has_pose = [True] * n_cams
        if len(has_pose) != n_cams:
            raise ValueError(
                f"has_pose has {len(has_pose)} entries for {n_cams} cameras")
        self.has_pose = [bool(v) for v in has_pose]
        self.per_camera_mlp = bool(per_camera_mlp)
        self.cam_embed = nn.Embedding(n_cams, embed_dim)

        def _mlp():
            return nn.Sequential(
                nn.Linear(pose_dim, embed_dim),
                nn.GELU(),
                nn.Linear(embed_dim, embed_dim),
            )

        # A ModuleDict keyed by camera index rather than a ModuleList, because
        # the cameras without a pose get no module at all -- a ModuleList would
        # need a placeholder, and a placeholder is how a silently-unused MLP
        # ends up in the state dict.
        if self.per_camera_mlp:
            self.pose_mlps = nn.ModuleDict(
                {str(i): _mlp() for i in range(n_cams) if self.has_pose[i]})
            self.pose_mlp = None
        else:
            self.pose_mlps = None
            self.pose_mlp = _mlp() if any(self.has_pose) else None
        self.absent = nn.Parameter(torch.zeros(embed_dim))

    def _mlp_for(self, cam_index: int):
        """The pose MLP for this camera, or None when it carries no pose."""
        if not self.has_pose[cam_index]:
            return None
        return self.pose_mlps[str(cam_index)] if self.per_camera_mlp else self.pose_mlp

    def forward(self, tokens, cam_index, pose9, valid):
        """tokens [B, T, D]; pose9 [B, 9]; valid [B] float 0/1 -> [B, T, D]."""
        idx = torch.full((tokens.shape[0],), cam_index,
                         dtype=torch.long, device=tokens.device)
        tag = self.cam_embed(idx)                             # [B, D]
        mlp = self._mlp_for(cam_index)
        if mlp is not None:
            tag = tag + mlp(pose9)
        out = tokens + tag.unsqueeze(1)
        keep = valid.reshape(-1, 1, 1).to(out.dtype)
        return keep * out + (1.0 - keep) * self.absent.view(1, 1, -1)


class MultiCamDP3Encoder(nn.Module):
    """Per-camera clouds -> one concatenated token sequence.

    The fused `DP3Encoder` takes a single world-frame `point_cloud`. This one
    takes `point_cloud_cam0..N` in their own camera frames, runs the SAME
    encoder over each (shared weights -- no new encoder parameters, and the
    pretrained Uni3D init stays intact), tags each block with its camera pose
    and identity, and concatenates along the token axis.

    That concatenation is free because the conditioning path is attention over
    an order-free token set: `dp3.py` reshapes to
    [B, n_obs_steps * num_tokens, D] and the UNet cross-attends into it. The
    diffusion head needs no change at all.

    Only `feature_mode='pointsam'` (extract_global_feature False) is supported:
    a per-camera *global* vector would defeat the point of the exercise, which
    is to let attention weight individual patches across cameras.
    """

    def __init__(self,
                 observation_space: Dict,
                 out_channel=256,
                 state_mlp_size=(64, 64), state_mlp_activation_fn=nn.ReLU,
                 pointcloud_encoder_cfg=None,
                 use_pc_color=False,
                 pointnet_type='uni3d_pretrained',
                 fps_random_config=None,
                 cat_on_token=True,
                 num_groups_per_camera=None,
                 per_camera_pose_mlp=False,
                 camera_has_pose=None,
                 ):
        super().__init__()
        state_mlp_size = (64, pointcloud_encoder_cfg['embed_dim'])

        self.point_cloud_keys = sorted(
            [k for k in observation_space if k.startswith("point_cloud_cam")],
            key=lambda k: int(k.replace("point_cloud_cam", "")))
        if not self.point_cloud_keys:
            raise RuntimeError(
                "MultiCamDP3Encoder found no point_cloud_cam* keys in shape_meta. "
                "A fused zarr wants DP3Encoder instead.")
        self.n_cams = len(self.point_cloud_keys)
        self.campose_keys = [
            k.replace("point_cloud_", "campose_") for k in self.point_cloud_keys]

        missing = [k for k in self.campose_keys if k not in observation_space]
        if missing:
            raise RuntimeError(
                f"shape_meta is missing {missing}. Every camera needs its pose "
                "vector; the dataset emits one per camera whatever pose_source "
                "is set to (zeros for 'none').")
        if "cam_valid" not in observation_space:
            raise RuntimeError(
                "shape_meta is missing 'cam_valid'. Without it a camera that saw "
                "nothing this frame is indistinguishable from one that did, and "
                "its degenerate tokens reach the conditioning.")

        # campose_* and cam_valid are consumed HERE, not as their own token
        # blocks -- leaving them in low_dim_keys would give the conditioning
        # four extra tokens and break dp3.py's `num_patches = num_tokens - 1`.
        consumed = set(self.point_cloud_keys) | set(self.campose_keys) | {"cam_valid"}
        self.low_dim_keys = [k for k in observation_space if k not in consumed]
        if len(self.low_dim_keys) != 1:
            raise RuntimeError(
                f"expected exactly one low-dim key (agent_pos), got "
                f"{self.low_dim_keys}. dp3.py computes num_patches as "
                "num_tokens - 1 under cat_on_token, so a second one would "
                "silently misalign pc_pe against the tokens.")
        self.low_dim_shapes = {k: observation_space[k] for k in self.low_dim_keys}

        self.use_pc_color = use_pc_color
        self.pointnet_type = pointnet_type
        self.cat_on_token = cat_on_token

        feature_mode = pointcloud_encoder_cfg.get('feature_mode', None)
        if feature_mode == 'pointsam':
            self.pc_encoder_extract_global_feature = False
        else:
            raise NotImplementedError(
                "MultiCamDP3Encoder needs feature_mode='pointsam' (per-patch "
                f"tokens); got {feature_mode!r}. A per-camera global vector "
                "cannot be attended into patch-wise.")

        self.fps_random_config = fps_random_config or {
            'use_random': True, 'random_start': True,
            'random_noise_scale': 0, 'shuffle_output': True,
        }

        if num_groups_per_camera is None:
            num_groups_per_camera = [
                pointcloud_encoder_cfg.get('num_group', 512) // self.n_cams
            ] * self.n_cams
        if len(num_groups_per_camera) != self.n_cams:
            raise ValueError(
                f"num_groups_per_camera has {len(num_groups_per_camera)} entries "
                f"for {self.n_cams} cameras")
        self.num_groups_per_camera = [int(g) for g in num_groups_per_camera]

        if pointnet_type not in ("uni3d", "uni3d_pretrained"):
            raise NotImplementedError(
                f"MultiCamDP3Encoder supports the uni3d encoders only, got "
                f"{pointnet_type!r}")
        uni3d_config = {
            'pc_model': 'eva02_large_patch14_448',
            'pc_feat_dim': 1024,
            'embed_dim': out_channel,
            'group_size': 32,
            'num_group': 512,
            'patch_dropout': 0.5,
            'drop_path_rate': 0.2,
            'pretrained_pc': None,
            'pc_encoder_dim': 512,
            'use_pretrained_weights': pointnet_type == "uni3d_pretrained",
            'pretrained_weights_path':
                'Uni3D_large/model.pt' if pointnet_type == "uni3d_pretrained" else None,
        }
        if pointcloud_encoder_cfg:
            uni3d_config.update(pointcloud_encoder_cfg)
        uni3d_config['fps_random_config'] = self.fps_random_config
        self.extractor = Uni3DPointcloudEncoder(**uni3d_config)
        pc_output_dim = uni3d_config['embed_dim']

        # Which cameras carry a meaningful pose, and whether each gets its own
        # MLP. Both are decided by `pose_source` upstream in dp3.py, not set
        # here, so the encoder still never needs to know which experiment is
        # running -- it only needs the consequences.
        if camera_has_pose is not None and len(camera_has_pose) != self.n_cams:
            raise ValueError(
                f"camera_has_pose has {len(camera_has_pose)} entries for "
                f"{self.n_cams} cameras")
        self.pose_embed = CameraPoseEmbedding(
            self.n_cams, pc_output_dim,
            per_camera_mlp=per_camera_pose_mlp,
            has_pose=camera_has_pose)

        output_dim = state_mlp_size[-1]
        net_arch = state_mlp_size[:-1] if len(state_mlp_size) > 1 else []
        self.low_dim_mlps = nn.ModuleDict()
        for key in self.low_dim_keys:
            shape = self.low_dim_shapes[key]
            if len(shape) != 1:
                raise RuntimeError(f"Low-dimensional obs '{key}' must be rank-1, got {shape}")
            self.low_dim_mlps[key] = nn.Sequential(
                *create_mlp(shape[0], output_dim, net_arch, state_mlp_activation_fn))

        self.n_output_channels = pc_output_dim if cat_on_token else \
            pc_output_dim + output_dim * len(self.low_dim_keys)

        cprint(f"[MultiCamDP3Encoder] cameras: {self.point_cloud_keys}", "yellow")
        cprint(f"[MultiCamDP3Encoder] groups per camera: {self.num_groups_per_camera} "
               f"(total {sum(self.num_groups_per_camera)} patch tokens)", "yellow")
        cprint(f"[MultiCamDP3Encoder] pose MLP: "
               f"{'one per camera' if per_camera_pose_mlp else 'shared'}; "
               f"cameras carrying a pose: {self.pose_embed.has_pose}", "yellow")
        cprint(f"[MultiCamDP3Encoder] low-dim keys: {self.low_dim_keys}", "yellow")
        cprint(f"[MultiCamDP3Encoder] Final output dim: {self.n_output_channels}", "yellow")

    def forward(self, observations: Dict, eval=False):
        cam_valid = observations["cam_valid"]

        feats, pes = [], []
        for i, key in enumerate(self.point_cloud_keys):
            points = observations[key]
            assert points.ndim == 3, f"{key}: expected [B, N, C], got {points.shape}"
            if points.shape[-1] == 3:
                points = torch.cat([points, torch.zeros_like(points)], dim=-1)
            elif points.shape[-1] != 6:
                # was a silent `[..., :6]` trim; extra channels are unsupported here
                raise ValueError(f"{key}: expected 3 or 6 channels, got {points.shape[-1]}")

            tokens, pe = self.extractor(
                points, eval, num_groups=self.num_groups_per_camera[i])
            tokens = self.pose_embed(
                tokens, i, observations[self.campose_keys[i]], cam_valid[..., i])
            feats.append(tokens)
            pes.append(pe)

        pn_feat = torch.cat(feats, dim=1)   # [B, sum(G_i), D]
        pc_pe = torch.cat(pes, dim=1)       # [B, sum(G_i), D_pe]

        low_dim_features = []
        for key in self.low_dim_keys:
            low_dim_feat = self.low_dim_mlps[key](observations[key])
            if self.cat_on_token:
                low_dim_feat = low_dim_feat.unsqueeze(1)
            else:
                low_dim_feat = low_dim_feat.unsqueeze(1).expand(-1, pn_feat.shape[1], -1)
            low_dim_features.append(low_dim_feat)

        features = [pn_feat] + low_dim_features
        final_feat = torch.cat(features, dim=-2 if self.cat_on_token else -1)
        return final_feat, pc_pe

    def output_shape(self):
        return self.n_output_channels
