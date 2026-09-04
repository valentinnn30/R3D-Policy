from typing import Dict
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from termcolor import cprint
import copy
import time
import pytorch3d.ops as torch3d_ops

from r3d.model.common.normalizer import LinearNormalizer
from r3d.policy.base_policy import BasePolicy
from r3d.model.diffusion.diffusion_backbone import ConditionalUnet1D
from r3d.model.diffusion.mask_generator import LowdimMaskGenerator
from r3d.common.pytorch_util import dict_apply
from r3d.common.model_util import print_params
from r3d.model.vision.pointnet_extractor import DP3Encoder, MultiCamDP3Encoder

class DP3(BasePolicy):
    def __init__(self,
            shape_meta: dict,
            noise_scheduler: DDPMScheduler,
            horizon,
            n_action_steps,
            n_obs_steps,
            num_inference_steps=None,
            obs_as_global_cond=True,
            diffusion_step_embed_dim=256,
            down_dims=(256,512,1024),
            kernel_size=5,
            n_groups=8,
            condition_type="film",
            use_down_condition=True,
            use_mid_condition=True,
            use_up_condition=True,
            encoder_output_dim=256,
            crop_shape=None,
            use_pc_color=False,
            pointnet_type="pointnet",
            pointcloud_encoder_cfg=None,
            fps_random_config=None,
            transformer_config=None,
            use_target_ee=False,
            cat_on_token=False,
            # Unfused pipeline only: patch tokens per camera. Ignored for a
            # fused shape_meta. None -> an even split of `num_group`.
            num_groups_per_camera=None,
            # Unfused pipeline only. Mirrored from task.dataset so the encoder's
            # pose handling cannot disagree with what the dataset actually
            # writes into campose_cam*.
            pose_source=None,
            camera_arm_slice=None,
            # None -> derived from pose_source (per-camera MLPs under 'proprio').
            per_camera_pose_mlp=None,
            # parameters passed to step
            **kwargs):
        super().__init__()

        self.use_target_ee = use_target_ee

        self.cat_on_token = cat_on_token

        cprint(f"[Diffusion] cat on token: {self.cat_on_token}", "green")

        self.condition_type = condition_type

        cprint(f"[Diffusion] condition_type: {self.condition_type}", "green")

        _feature_mode = pointcloud_encoder_cfg.get('feature_mode', None)
        self.pc_encoder_extract_global_feature = _feature_mode != 'pointsam'

        # parse shape_meta
        action_shape = shape_meta['action']['shape']
        self.action_shape = action_shape
        if len(action_shape) == 1:
            action_dim = action_shape[0]
        elif len(action_shape) == 2: # use multiple hands
            action_dim = action_shape[0] * action_shape[1]
        else:
            raise NotImplementedError(f"Unsupported action shape {action_shape}")
            
        obs_shape_meta = shape_meta['obs']
        obs_dict = dict_apply(obs_shape_meta, lambda x: x['shape'])

        # Which encoder to build is decided by the DATA, not by a flag: a
        # shape_meta carrying point_cloud_cam* is an unfused zarr and there is
        # no `point_cloud` for DP3Encoder to read. Auto-detecting means the
        # encoder cannot be set inconsistently with the dataset that feeds it.
        self.unfused = any(k.startswith("point_cloud_cam") for k in obs_dict)
        # Every observation key holding a cloud. One entry when fused; one per
        # camera when not. Used by the clip / drop-colour steps below, which
        # would otherwise KeyError on 'point_cloud' for an unfused batch.
        self.pc_keys = sorted(
            [k for k in obs_dict if k.startswith("point_cloud_cam")],
            key=lambda k: int(k.replace("point_cloud_cam", ""))
        ) if self.unfused else ["point_cloud"]
        if self.unfused:
            cprint("[Diffusion] unfused multi-camera observation detected -> "
                   "MultiCamDP3Encoder (merge in feature space)", "green")

            # What the camera tag is allowed to represent follows `pose_source`,
            # which lives on the DATASET. Deriving both switches from it here,
            # rather than exposing a second independent one, is what makes
            # `task.dataset.pose_source=proprio` on the command line reconfigure
            # the whole experiment. The failure it prevents is a proprio run
            # training with a shared pose MLP -- which cannot fold in the
            # constant flange->camera offset that A' exists to test, so the
            # ablation would answer a different question than the one asked,
            # and nothing would report that it had.
            n_cams = len(self.pc_keys)
            pose_source = "extrinsic" if pose_source is None else str(pose_source)
            if pose_source == "extrinsic":
                camera_has_pose = [True] * n_cams
            elif pose_source == "none":
                camera_has_pose = [False] * n_cams
            elif pose_source == "proprio":
                if camera_arm_slice is None:
                    raise ValueError(
                        "pose_source='proprio' needs camera_arm_slice, so the "
                        "encoder knows which cameras carry a pose at all. Wire "
                        "it from task.dataset.camera_arm_slice -- the dataset "
                        "hands a camera without an arm zeros(9), and an MLP on "
                        "a constant is parameters spent on nothing.")
                if len(camera_arm_slice) != n_cams:
                    raise ValueError(
                        f"camera_arm_slice has {len(camera_arm_slice)} entries "
                        f"for {n_cams} cameras")
                camera_has_pose = [s is not None for s in camera_arm_slice]
                if not any(camera_has_pose):
                    raise ValueError(
                        "pose_source='proprio' but no camera has an arm slice, "
                        "so every tag would collapse to its identity embedding "
                        "-- that is the 'none' control, not A'.")
            else:
                raise ValueError(
                    f"unknown pose_source {pose_source!r} (expected "
                    "'extrinsic', 'proprio' or 'none')")

            if per_camera_pose_mlp is None:
                per_camera_pose_mlp = pose_source == "proprio"
            cprint(f"[Diffusion] pose_source: {pose_source} -> "
                   f"per-camera pose MLP: {per_camera_pose_mlp}, "
                   f"cameras with a pose: {camera_has_pose}", "green")

            obs_encoder = MultiCamDP3Encoder(
                observation_space=obs_dict,
                out_channel=encoder_output_dim,
                pointcloud_encoder_cfg=pointcloud_encoder_cfg,
                use_pc_color=use_pc_color,
                pointnet_type=pointnet_type,
                fps_random_config=fps_random_config,
                cat_on_token=cat_on_token,
                num_groups_per_camera=num_groups_per_camera,
                per_camera_pose_mlp=per_camera_pose_mlp,
                camera_has_pose=camera_has_pose,
            )
        else:
            obs_encoder = DP3Encoder(
                observation_space=obs_dict,
                img_crop_shape=crop_shape,
                out_channel=encoder_output_dim,
                pointcloud_encoder_cfg=pointcloud_encoder_cfg,
                use_pc_color=use_pc_color,
                pointnet_type=pointnet_type,
                fps_random_config=fps_random_config,
                cat_on_token=cat_on_token,
            )

        # create diffusion model
        obs_feature_dim = obs_encoder.output_shape() # embed_dim + robot_state_embed_dim = 512
        input_dim = action_dim + obs_feature_dim
        global_cond_dim = None
        if obs_as_global_cond:
            input_dim = action_dim
            if "cross_attention" in self.condition_type or self.condition_type == "one_way_transformer":
                global_cond_dim = obs_feature_dim
            else:
                global_cond_dim = obs_feature_dim * n_obs_steps
        

        self.use_pc_color = use_pc_color
        self.pointnet_type = pointnet_type
        cprint(f"[DiffusionUnetHybridPointcloudPolicy] use_pc_color: {self.use_pc_color}", "yellow")
        cprint(f"[DiffusionUnetHybridPointcloudPolicy] pointnet_type: {self.pointnet_type}", "yellow")

        # Hint: ensure encoder_output_dim matches Uni3D output dimension
        if pointnet_type in ["uni3d", "uni3d_pretrained"]:
            cprint(f"[DP3] Uni3D encoder detected, ensure encoder_output_dim matches Uni3D output dim", "cyan")

        model = ConditionalUnet1D(
            input_dim=input_dim,
            local_cond_dim=None,
            global_cond_dim=global_cond_dim,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            down_dims=down_dims,
            kernel_size=kernel_size,
            n_groups=n_groups,
            condition_type=condition_type,
            use_down_condition=use_down_condition,
            use_mid_condition=use_mid_condition,
            use_up_condition=use_up_condition,
            transformer_config=transformer_config,
            use_target_ee=self.use_target_ee,
            cat_on_token=self.cat_on_token
        )

        self.obs_encoder = obs_encoder
        self.model = model
        self.noise_scheduler = noise_scheduler
        self.noise_scheduler_pc = copy.deepcopy(noise_scheduler)
        self.mask_generator = LowdimMaskGenerator(
            action_dim=action_dim,
            obs_dim=0 if obs_as_global_cond else obs_feature_dim,
            max_n_obs_steps=n_obs_steps,
            fix_obs_steps=True,
            action_visible=False
        )
        
        self.normalizer = LinearNormalizer()
        self.horizon = horizon
        self.obs_feature_dim = obs_feature_dim
        self.action_dim = action_dim
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.obs_as_global_cond = obs_as_global_cond
        self.kwargs = kwargs

        if num_inference_steps is None:
            num_inference_steps = noise_scheduler.config.num_train_timesteps
        self.num_inference_steps = num_inference_steps


        print_params(self)

    # ========= inference  ============
    def conditional_sample(self,
            condition_data, condition_mask,
            condition_data_pc=None, condition_mask_pc=None,
            local_cond=None, global_cond=None,
            pc_pe=None,
            n_obs_steps=None,
            generator=None,
            # keyword arguments to scheduler.step
            **kwargs
            ):
        model = self.model
        scheduler = self.noise_scheduler

        trajectory = torch.randn(
            size=condition_data.shape,
            dtype=condition_data.dtype,
            device=condition_data.device)

        # set step values
        scheduler.set_timesteps(self.num_inference_steps)

        for t in scheduler.timesteps:
            # 1. apply conditioning
            trajectory[condition_mask] = condition_data[condition_mask].to(trajectory.dtype)

            model_output = model(sample=trajectory,
                                timestep=t,
                                local_cond=local_cond, global_cond=global_cond, pc_pe=pc_pe,
                                n_obs_steps=n_obs_steps)

            # 3. compute previous image: x_t -> x_t-1
            trajectory = scheduler.step(model_output, t, trajectory).prev_sample

        # finally make sure conditioning is enforced
        trajectory[condition_mask] = condition_data[condition_mask].to(trajectory.dtype)

        return trajectory


    def predict_action(self, obs_dict) -> Dict[str, torch.Tensor]:
        """
        obs_dict: must include "obs" key
        result: must include "action" key
        """
        # normalize input
        nobs = self.normalizer.normalize(obs_dict)

        # Clip point cloud to ensure it's within [-1-1e-6, 1+1e-6].
        #
        # DELIBERATELY SKIPPED for the unfused pipeline. Its xyz already lands
        # inside [-1, 1] (one shared isotropic METRIC_SCALE_M; measured max
        # |xyz| 0.93), so the clamp would be a no-op -- but only while that
        # scale is right. If someone shrinks METRIC_SCALE_M, clamping would
        # SILENTLY flatten the far points onto the cube faces, whereas letting
        # them through makes the encoder's PositionEmbeddingRandom raise
        # "Input coordinates must be normalized to [-1, 1]" and say so. A loud
        # failure beats a silent clip, so the guard stays off here.
        #
        # The fused path keeps it: there the workspace box makes [-1, 1] true
        # by construction, so the clamp only ever absorbs float error.
        for _k in self.pc_keys:
            if _k in nobs:
                if not self.unfused:
                    nobs[_k] = torch.clamp(nobs[_k], min=-1-1e-6, max=1+1e-6)
                if not self.use_pc_color:
                    nobs[_k] = nobs[_k][..., :3]

        value = next(iter(nobs.values()))
        B, To = value.shape[:2]
        T = self.horizon
        Da = self.action_dim
        Do = self.obs_feature_dim
        To = self.n_obs_steps

        # build input
        device = self.device
        dtype = self.dtype

        # handle different ways of passing observation
        local_cond = None
        global_cond = None
        pc_pe = None
        if self.obs_as_global_cond:
            # condition through global feature
            this_nobs = dict_apply(nobs,
                lambda x: x[:, :To, ...].reshape(-1, *x.shape[2:]))

            if not self.pc_encoder_extract_global_feature:
                nobs_features, pc_pe = self.obs_encoder(this_nobs, eval=True) # [B*num_obs_steps, num_patches, embed_dim]
                num_patches = pc_pe.shape[1]
                num_tokens = nobs_features.shape[1]
            else:
                nobs_features = self.obs_encoder(this_nobs, eval=True)

            if "cross_attention" in self.condition_type or self.condition_type == "one_way_transformer":
                # treat as a sequence
                if not self.pc_encoder_extract_global_feature:
                    global_cond = nobs_features.reshape(B, self.n_obs_steps * num_tokens, -1)
                    pc_pe = pc_pe.reshape(B, self.n_obs_steps * num_patches, -1)
                else:
                    global_cond = nobs_features.reshape(B, self.n_obs_steps, -1)
            else:
                # reshape back to B, Do
                global_cond = nobs_features.reshape(B, -1)
            # Initialize empty action data
            cond_data = torch.zeros(size=(B, T, Da), device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
        else:
            # condition through inpainting
            this_nobs = dict_apply(nobs,
                lambda x: x[:, :To, ...].reshape(-1, *x.shape[2:]))

            nobs_features = self.obs_encoder(this_nobs, eval=True)
            # reshape back to B, T, Do
            nobs_features = nobs_features.reshape(B, To, -1)
            cond_data = torch.zeros(size=(B, T, Da+Do), device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
            cond_data[:,:To,Da:] = nobs_features
            cond_mask[:,:To,Da:] = True
        
        # run sampling
        nsample = self.conditional_sample(
            cond_data, 
            cond_mask,
            local_cond=local_cond,
            global_cond=global_cond,
            pc_pe=pc_pe,
            n_obs_steps=self.n_obs_steps,
            **self.kwargs)
        
        # unnormalize prediction
        naction_pred = nsample[...,:Da]
        action_pred = self.normalizer['action'].unnormalize(naction_pred)

        # get action
        start = To - 1
        end = start + self.n_action_steps
        action = action_pred[:,start:end]
        
        # If ee auxiliary task is enabled, only return the joint portion for execution
        if self.use_target_ee:
            ee_dim = Da // 2
            joint_dim = Da - ee_dim

            # First half of action dims = joint, second half = ee
            joint_action = action[:, :, :joint_dim]  # (B, Ta, 14)
            ee_action = action[:, :, joint_dim:]     # (B, Ta, 14)

            result = {
                'action': joint_action,           # joint only for execution
                'action_pred': action_pred,       # full prediction (28-dim)
                'ee_pred': ee_action,             # ee prediction (for logging)
            }
            
        else:
            result = {
                'action': action,                 # (B, Ta, 14)
                'action_pred': action_pred,       # (B, T, 14)
            }
        
        return result

    # ========= training  ============
    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def compute_loss(self, batch):
        # normalize input
        obs_dict = batch['obs']
        nobs = self.normalizer.normalize(obs_dict)

        # Clip point cloud to ensure it's within [-1-1e-6, 1+1e-6].
        #
        # DELIBERATELY SKIPPED for the unfused pipeline. Its xyz already lands
        # inside [-1, 1] (one shared isotropic METRIC_SCALE_M; measured max
        # |xyz| 0.93), so the clamp would be a no-op -- but only while that
        # scale is right. If someone shrinks METRIC_SCALE_M, clamping would
        # SILENTLY flatten the far points onto the cube faces, whereas letting
        # them through makes the encoder's PositionEmbeddingRandom raise
        # "Input coordinates must be normalized to [-1, 1]" and say so. A loud
        # failure beats a silent clip, so the guard stays off here.
        #
        # The fused path keeps it: there the workspace box makes [-1, 1] true
        # by construction, so the clamp only ever absorbs float error.
        for _k in self.pc_keys:
            if _k in nobs:
                if not self.unfused:
                    nobs[_k] = torch.clamp(nobs[_k], min=-1-1e-6, max=1+1e-6)
                if not self.use_pc_color:
                    nobs[_k] = nobs[_k][..., :3]

        nactions = self.normalizer['action'].normalize(batch['action'])
        
        batch_size = nactions.shape[0]
        horizon = nactions.shape[1]

        # handle different ways of passing observation
        local_cond = None
        global_cond = None
        trajectory = nactions
        cond_data = trajectory
        pc_pe = None

        if self.obs_as_global_cond:
            # reshape B, T, ... to B*T
            this_nobs = dict_apply(nobs,
                lambda x: x[:, :self.n_obs_steps, ...].reshape(-1, *x.shape[2:]))

            if not self.pc_encoder_extract_global_feature:

                nobs_features, pc_pe = self.obs_encoder(this_nobs) # [B*num_obs_steps, num_patches, embed_dim(*2)]
                num_patches = pc_pe.shape[1]
                num_tokens = nobs_features.shape[1]
            else:
                nobs_features = self.obs_encoder(this_nobs)

            if "cross_attention" in self.condition_type or self.condition_type == "one_way_transformer":
                # treat as a sequence
                if not self.pc_encoder_extract_global_feature:
                    global_cond = nobs_features.reshape(batch_size, self.n_obs_steps * num_tokens, -1)
                    pc_pe = pc_pe.reshape(batch_size, self.n_obs_steps * num_patches, -1)
                else:
                    global_cond = nobs_features.reshape(batch_size, self.n_obs_steps, -1)
            else:
                # reshape back to B, Do
                global_cond = nobs_features.reshape(batch_size, -1)
        else:
            # reshape B, T, ... to B*T
            this_nobs = dict_apply(nobs, lambda x: x.reshape(-1, *x.shape[2:]))

            nobs_features = self.obs_encoder(this_nobs)
            # reshape back to B, T, Do
            nobs_features = nobs_features.reshape(batch_size, horizon, -1)
            cond_data = torch.cat([nactions, nobs_features], dim=-1)
            trajectory = cond_data.detach()

        # generate impainting mask
        condition_mask = self.mask_generator(trajectory.shape)

        # Sample noise that we'll add to the images
        noise = torch.randn(trajectory.shape, device=trajectory.device)

        bsz = trajectory.shape[0]
        # Sample a random timestep for each image
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps, 
            (bsz,), device=trajectory.device
        ).long()

        # Add noise to the clean images according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_trajectory = self.noise_scheduler.add_noise(
            trajectory, noise, timesteps)

        # compute loss mask
        loss_mask = ~condition_mask

        # apply conditioning
        noisy_trajectory[condition_mask] = cond_data[condition_mask]

        # Predict the noise residual
        pred = self.model(sample=noisy_trajectory,
                        timestep=timesteps,
                        local_cond=local_cond,
                        global_cond=global_cond,
                        pc_pe=pc_pe,
                        n_obs_steps=self.n_obs_steps)


        pred_type = self.noise_scheduler.config.prediction_type 
        if pred_type == 'epsilon':
            target = noise
        elif pred_type == 'sample':
            target = trajectory
        elif pred_type == 'v_prediction':
            self.noise_scheduler.alpha_t = self.noise_scheduler.alpha_t.to(self.device)
            self.noise_scheduler.sigma_t = self.noise_scheduler.sigma_t.to(self.device)
            alpha_t, sigma_t = self.noise_scheduler.alpha_t[timesteps], self.noise_scheduler.sigma_t[timesteps]
            alpha_t = alpha_t.unsqueeze(-1).unsqueeze(-1)
            sigma_t = sigma_t.unsqueeze(-1).unsqueeze(-1)
            v_t = alpha_t * noise - sigma_t * trajectory
            target = v_t
        else:
            raise ValueError(f"Unsupported prediction type {pred_type}")

        loss = F.mse_loss(pred, target, reduction='none')
        loss = loss * loss_mask.type(loss.dtype)

        # If ee auxiliary task is enabled, compute joint and ee losses separately
        if self.use_target_ee:
            ee_dim = self.action_dim // 2
            joint_dim = self.action_dim - ee_dim
            joint_loss = loss[:, :, :joint_dim]  # first half: joint dims
            ee_loss = loss[:, :, joint_dim:]     # second half: ee dims

            joint_loss_mean = reduce(joint_loss, 'b ... -> b (...)', 'mean').mean()
            ee_loss_mean = reduce(ee_loss, 'b ... -> b (...)', 'mean').mean()

            ee_loss_weight = 1
            total_loss = joint_loss_mean + ee_loss_weight * ee_loss_mean
            
            loss_dict = {
                'bc_loss': total_loss.item(),
                'joint_loss': joint_loss_mean.item(),
                'ee_loss': ee_loss_mean.item(),
            }
            
            loss = total_loss

        else:
            loss = reduce(loss, 'b ... -> b (...)', 'mean')
            loss = loss.mean()
            

            loss_dict = {
                'bc_loss': loss.item(),
            }

        return loss, loss_dict

