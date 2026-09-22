"""DP3Hybrid -- the hybrid 2D/3D policy. Only the encoder differs from `DP3`.

Plan: ~/ros2_ws/hybrid_2d3d_ablations_plan.txt

Same construction as `DP3Image`, for the same reason: `dp3.py` is NOT modified.
`__init__` rebuilds only the encoder; `conditional_sample`, `predict_action`,
`compute_loss` and `set_normalizer` are inherited verbatim.

The one piece of inherited point-cloud handling is the loop

    for _k in self.pc_keys:
        if not self.unfused: clamp to [-1, 1]
        if not self.use_pc_color: drop rgb

With `pc_keys` = this policy's 3D keys and `unfused = not fuse_3d`, that loop
does exactly what each source run does: the fused wrist cloud is clamped (as in
run A, where the workspace box makes [-1, 1] true by construction), unfused
camera-frame clouds are not (as in Run B, where a loud PE range error beats a
silent clip), and image keys are never touched.
"""

import copy

from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from termcolor import cprint

from r3d.common.model_util import print_params
from r3d.model.common.normalizer import LinearNormalizer
from r3d.model.diffusion.diffusion_backbone import ConditionalUnet1D
from r3d.model.diffusion.mask_generator import LowdimMaskGenerator
from r3d.model.vision.hybrid_extractor import MultiModalEncoder
from r3d.policy.dp3 import DP3


class DP3Hybrid(DP3):
    def __init__(self,
            shape_meta: dict,
            noise_scheduler: DDPMScheduler,
            horizon,
            n_action_steps,
            n_obs_steps,
            num_inference_steps=None,
            obs_as_global_cond=True,
            diffusion_step_embed_dim=256,
            down_dims=(256, 512, 1024),
            kernel_size=5,
            n_groups=8,
            condition_type="one_way_transformer",
            use_down_condition=True,
            use_mid_condition=True,
            use_up_condition=True,
            encoder_output_dim=256,
            use_pc_color=True,
            pointnet_type="uni3d",
            pointcloud_encoder_cfg=None,
            image_encoder_cfg=None,
            fps_random_config=None,
            transformer_config=None,
            use_target_ee=False,
            cat_on_token=False,
            # Mirrored from task.dataset, so one command-line override
            # reconfigures dataset AND encoder -- no second switch to forget.
            cameras_3d=None,
            cameras_2d=None,
            fuse_3d=False,
            num_groups_per_camera=None,
            fused_num_groups=None,
            pose_source_3d="extrinsic",
            pose_source_2d="none",
            camera_arm_slice=None,
            **kwargs):
        # Skip DP3.__init__: it would build the single-modality encoder.
        super(DP3, self).__init__()

        self.use_target_ee = use_target_ee
        self.cat_on_token = cat_on_token
        self.condition_type = condition_type
        cprint(f"[DP3Hybrid] cat on token: {self.cat_on_token}", "green")
        cprint(f"[DP3Hybrid] condition_type: {self.condition_type}", "green")

        # Per-patch tokens, always -- both source pipelines are 'pointsam'.
        self.pc_encoder_extract_global_feature = False
        self.use_pc_color = use_pc_color
        self.pointnet_type = pointnet_type
        self.fuse_3d = bool(fuse_3d)
        self.unfused = not self.fuse_3d

        action_shape = shape_meta['action']['shape']
        self.action_shape = action_shape
        if len(action_shape) == 1:
            action_dim = action_shape[0]
        elif len(action_shape) == 2:
            action_dim = action_shape[0] * action_shape[1]
        else:
            raise NotImplementedError(f"Unsupported action shape {action_shape}")

        obs_dict = {k: v['shape'] for k, v in shape_meta['obs'].items()}
        if cameras_3d is None or cameras_2d is None:
            raise ValueError(
                "DP3Hybrid needs cameras_3d and cameras_2d; wire them from "
                "task.dataset like pose_source.")

        obs_encoder = MultiModalEncoder(
            observation_space=obs_dict,
            cameras_3d=list(cameras_3d),
            cameras_2d=list(cameras_2d),
            fuse_3d=self.fuse_3d,
            out_channel=encoder_output_dim,
            pointcloud_encoder_cfg=pointcloud_encoder_cfg,
            image_encoder_cfg=image_encoder_cfg,
            pointnet_type=pointnet_type,
            fps_random_config=fps_random_config,
            cat_on_token=cat_on_token,
            num_groups_per_camera=(None if num_groups_per_camera is None
                                   else list(num_groups_per_camera)),
            fused_num_groups=fused_num_groups,
            pose_source_3d=str(pose_source_3d),
            pose_source_2d=str(pose_source_2d),
            camera_arm_slice=(None if camera_arm_slice is None else
                              [None if s is None else list(s) for s in camera_arm_slice]),
        )
        self.blocks = obs_encoder.blocks
        self.resolved = obs_encoder.resolved
        # The inherited clip / drop-colour loop runs over the 3D keys only.
        self.pc_keys = [b["key"] for b in self.blocks if b["kind"] != "image"]

        obs_feature_dim = obs_encoder.output_shape()
        input_dim = action_dim + obs_feature_dim
        global_cond_dim = None
        if obs_as_global_cond:
            input_dim = action_dim
            if "cross_attention" in self.condition_type \
                    or self.condition_type == "one_way_transformer":
                global_cond_dim = obs_feature_dim
            else:
                global_cond_dim = obs_feature_dim * n_obs_steps

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
            cat_on_token=self.cat_on_token,
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
            action_visible=False,
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
