"""DP3Image -- the 2D baseline policy. Only the encoder differs from `DP3`.

Design brief: `~/ros2_ws/tmp/dinov2_2d_baseline_design.md`.

`dp3.py` is NOT modified. This subclasses `DP3` and rebuilds only the parts of
`__init__` that mention point clouds; every method after construction --
`conditional_sample`, `predict_action`, `compute_loss`, `set_normalizer` -- is
inherited verbatim and runs unchanged.

That inheritance works because the point-cloud specific code in those methods is
confined to one loop:

    for _k in self.pc_keys:
        if _k in nobs: ...clip / drop colour...

Setting `self.pc_keys = []` makes it a no-op, which is why images need no fork
of the sampling or the loss. The rest of the conditioning path is already
modality-agnostic: it consumes a token sequence plus a positional encoding, and
`MultiCamImageEncoder` returns exactly the pair `MultiCamDP3Encoder` returns.
"""

import copy

import torch.nn as nn
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from termcolor import cprint

from r3d.common.model_util import print_params
from r3d.model.common.normalizer import LinearNormalizer
from r3d.model.diffusion.diffusion_backbone import ConditionalUnet1D
from r3d.model.diffusion.mask_generator import LowdimMaskGenerator
from r3d.model.vision.image_extractor import MultiCamImageEncoder
from r3d.policy.dp3 import DP3


class DP3Image(DP3):
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
            image_encoder_cfg=None,
            transformer_config=None,
            use_target_ee=False,
            cat_on_token=True,
            # Mirrored from task.dataset so the encoder's pose handling cannot
            # disagree with what the dataset writes into campose_cam*.
            pose_source=None,
            camera_arm_slice=None,
            # None -> derived from pose_source (per-camera MLPs under 'proprio').
            per_camera_pose_mlp=None,
            **kwargs):
        # Skip DP3.__init__ deliberately: it builds a Uni3D trunk
        # unconditionally, which would download weights and allocate a point
        # encoder this policy never uses. Go straight to BasePolicy.
        super(DP3, self).__init__()

        self.use_target_ee = use_target_ee
        self.cat_on_token = cat_on_token
        self.condition_type = condition_type
        cprint(f"[DP3Image] cat on token: {self.cat_on_token}", "green")
        cprint(f"[DP3Image] condition_type: {self.condition_type}", "green")

        # Per-patch tokens, always. There is no image equivalent of the
        # `feature_mode` switch: pooling a view to one vector is the original
        # Diffusion Policy conditioning, and adopting it would change the
        # conditioning path as well as the encoder.
        self.pc_encoder_extract_global_feature = False
        # No point-cloud keys, so the clip / drop-colour loops inherited from
        # DP3.predict_action and DP3.compute_loss iterate over nothing.
        self.pc_keys = []
        self.unfused = True
        self.use_pc_color = False

        action_shape = shape_meta['action']['shape']
        self.action_shape = action_shape
        if len(action_shape) == 1:
            action_dim = action_shape[0]
        elif len(action_shape) == 2:
            action_dim = action_shape[0] * action_shape[1]
        else:
            raise NotImplementedError(f"Unsupported action shape {action_shape}")

        obs_shape_meta = shape_meta['obs']
        obs_dict = {k: v['shape'] for k, v in obs_shape_meta.items()}

        n_cams = len([k for k in obs_dict if k.startswith("image_cam")])
        if n_cams == 0:
            raise RuntimeError(
                "DP3Image needs image_cam* keys in shape_meta. A point-cloud "
                "task config wants r3d.policy.dp3.DP3 instead.")

        # Same derivation as the unfused 3D policy, and for the same reason:
        # `pose_source` lives on the DATASET, so deriving both switches from it
        # here means `task.dataset.pose_source=proprio` on the command line
        # reconfigures the whole experiment with no second switch to forget.
        pose_source = "extrinsic" if pose_source is None else str(pose_source)
        if pose_source == "extrinsic":
            camera_has_pose = [True] * n_cams
        elif pose_source == "none":
            camera_has_pose = [False] * n_cams
        elif pose_source == "proprio":
            if camera_arm_slice is None:
                raise ValueError(
                    "pose_source='proprio' needs camera_arm_slice, so the "
                    "encoder knows which cameras carry a pose at all. Wire it "
                    "from task.dataset.camera_arm_slice.")
            if len(camera_arm_slice) != n_cams:
                raise ValueError(
                    f"camera_arm_slice has {len(camera_arm_slice)} entries for "
                    f"{n_cams} cameras")
            camera_has_pose = [s is not None for s in camera_arm_slice]
            if not any(camera_has_pose):
                raise ValueError(
                    "pose_source='proprio' but no camera has an arm slice, so "
                    "every tag would collapse to its identity embedding -- that "
                    "is the 'none' control, not the kinematic ablation.")
        else:
            raise ValueError(
                f"unknown pose_source {pose_source!r} (expected 'extrinsic', "
                "'proprio' or 'none')")

        if per_camera_pose_mlp is None:
            per_camera_pose_mlp = pose_source == "proprio"
        cprint(f"[DP3Image] pose_source: {pose_source} -> per-camera pose MLP: "
               f"{per_camera_pose_mlp}, cameras with a pose: {camera_has_pose}",
               "green")

        obs_encoder = MultiCamImageEncoder(
            observation_space=obs_dict,
            out_channel=encoder_output_dim,
            image_encoder_cfg=image_encoder_cfg,
            cat_on_token=cat_on_token,
            per_camera_pose_mlp=per_camera_pose_mlp,
            camera_has_pose=camera_has_pose,
        )

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
