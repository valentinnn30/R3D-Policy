"""Observation transform for real bimanual Franka data.

This module is *the* definition of the point-cloud transform. The converter
(``scripts/convert_h5_to_zarr.py``) and the real-robot inference loop -- which
lives in a separate repo -- must both import this file. Crop, scale and frame
convention have to be bit-identical between training and inference; a silently
different workspace crop degrades the policy with no error anywhere.

Two stages, in this order:

*Conversion* (``preprocess_camera_frame``), once per camera per frame:
  scale to metres -> nominal extrinsics -> crop to the workspace box grown by
  ``margin`` -> farthest-point-sample to ``points_per_camera`` -> RGB carried
  through the FPS indices. The points come back in CAMERA frame, because the
  perturbation has to be applied to them later as an extrinsic.

*Sample time* (``fuse_cameras``), shared by the dataset and the robot:
  apply each camera's transform -> concatenate -> crop to the exact box ->
  subsample to ``num_points``.

Frame convention
----------------
``extrinsics`` is a standard 4x4 homogeneous camera->base transform acting on
*column* vectors, i.e. ``p_base = T @ p_cam``. This is what a ChArUco hand-eye
solve gives you. Note that the DP3 template this replaces
(``scripts/convert_real_robot_data.py``) used ``p_row @ M``, which is the same
thing only if ``M`` is the transpose of ``T`` -- do not copy a matrix between
the two without transposing it.

RGB convention
--------------
RGB is stored in ``[0, 1]``, not ``[0, 255]``. This is not cosmetic:
``RobotwinDataset.apply_color_jitter`` clips to ``[0, 1]``, so 0-255 colours
would be flattened to solid white by the training-time augmentation.
"""

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
import yaml
from termcolor import cprint

try:
    import pytorch3d.ops as torch3d_ops
except ImportError as e:  # pragma: no cover - pytorch3d is a hard dependency
    raise ImportError(
        "pytorch3d is required (it is used on every policy forward pass too). "
        "See CLAUDE.md for the CUDA_HOME / TORCH_CUDA_ARCH_LIST build notes."
    ) from e


@dataclass
class PointCloudPreprocessConfig:
    """Calibration + cropping constants. Load from yaml, never hardcode."""

    # [[x_min, x_max], [y_min, y_max], [z_min, z_max]] in the robot base frame
    workspace: List[List[float]]
    # multiplier taking raw xyz to metres. The ros2_ws clouds are already in
    # metres, so this stays 1.0 unless a source publishes millimetres.
    point_scale: float = 1.0
    # value that means "full intensity" in the source RGB. 255 for uint8.
    rgb_max: float = 255.0
    num_points: int = 1024

    # ------------------------------------------------------------- floor cut --
    # Fixed table plane in the BASE frame, fitted once offline (RANSAC + least
    # squares over 150 frames) and then held constant, exactly like the
    # extrinsics. Points at or below `floor_margin` above it are dropped.
    #
    # Why a plane and not a z bound: the table is tilted 2.67 deg in this world
    # frame, so over the 0.65 m workspace a flat z-cut is uneven by ~28 mm --
    # it would leave floor at one end or eat the cube at the other.
    #
    # Why this is worth doing at all: the clouds are fused in the WORLD frame,
    # so static geometry is not informative the way a moving-camera image
    # background is. Worse, which PART of the floor is visible changes with arm
    # pose (the wrist cameras ride on the arms), making it a nuisance variable
    # correlated with proprioception. Measured: 62% of every fused cloud was
    # table, and the policy demonstrably responded to it.
    #
    # STALE IF THE TABLE OR CAMERAS MOVE, same failure mode as the extrinsics.
    # Must be byte-identical at inference or the observation silently diverges.
    floor_normal: Optional[List[float]] = None
    floor_offset: Optional[float] = None
    floor_margin: float = 0.0

    @classmethod
    def from_yaml(cls, path: str) -> "PointCloudPreprocessConfig":
        with open(path, "r") as f:
            raw = yaml.safe_load(f)
        # tolerate the converter's yaml carrying unrelated sections
        known = {k: v for k, v in raw.items() if k in cls.__dataclass_fields__}
        return cls(**known)

    @property
    def bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        ws = np.asarray(self.workspace, dtype=np.float64)
        assert ws.shape == (3, 2), f"workspace must be 3x2, got {ws.shape}"
        return ws[:, 0], ws[:, 1]


def crop_to_workspace(points: np.ndarray, cfg: PointCloudPreprocessConfig) -> np.ndarray:
    """Step 3. ``points`` is (N, >=3) with xyz first; the crop keeps all columns."""
    lo, hi = cfg.bounds
    xyz = points[:, :3]
    keep = np.all((xyz > lo) & (xyz < hi), axis=-1)
    return points[keep]


def farthest_point_sample(
    points: np.ndarray, num_points: int, device: str = "cuda"
) -> Tuple[np.ndarray, np.ndarray]:
    """Step 4. Returns (sampled_xyz, indices into ``points``)."""
    pts = torch.as_tensor(np.ascontiguousarray(points[:, :3]), dtype=torch.float32)
    pts = pts.to(device).unsqueeze(0)
    sampled, idx = torch3d_ops.sample_farthest_points(points=pts, K=[num_points])
    return sampled.squeeze(0).cpu().numpy(), idx.squeeze(0).cpu().numpy()


def pad_to_min_points(points: np.ndarray, num_points: int) -> np.ndarray:
    """Guard the crop against starving FPS.

    ``sample_farthest_points`` silently returns padded (-1) indices when asked
    for more points than it was given, which would write garbage into the zarr.
    If the crop left too few points we duplicate at random instead, and shout
    about it -- a frame that trips this usually means the extrinsics or the
    workspace box are wrong.
    """
    n = points.shape[0]
    if n == 0:
        raise ValueError(
            "workspace crop removed every point. Check that `extrinsics` really "
            "maps camera -> robot base (column-vector convention) and that "
            "`workspace` covers the table volume in base coordinates."
        )
    if n >= num_points:
        return points
    cprint(
        f"[real_preprocess] only {n} points survived the crop (need {num_points}); "
        "padding with duplicates",
        "yellow",
    )
    extra = np.random.randint(0, n, size=num_points - n)
    return np.concatenate([points, points[extra]], axis=0)


# ------------------------------------------------------- multi-camera fusion --
#
# A single fused cloud cannot express calibration error: once the cameras are
# merged, a per-camera extrinsic mistake is baked in, and any transform applied
# afterwards moves every point together. To train for
# robustness to *misregistration between cameras*, the clouds are kept separate,
# in their own camera frames, and fused at sample time under perturbed
# extrinsics. See CLAUDE.md, "Extrinsics randomization -- built".
#
# Both the training dataset and the on-robot inference loop call `fuse_cameras`.
# Inference passes the real solved extrinsics and perturbs nothing.


def random_se3(rot_sigma_deg: float, trans_sigma_m: float, rng=None) -> np.ndarray:
    """A small random rigid transform, for perturbing one camera's extrinsics.

    Rotation is sampled per-axis in the small-angle regime and mapped through
    Rodrigues. Note the lever arm: 0.5 deg displaces a point 1 m from the camera
    by ~8.7 mm, so the rotation term usually dominates the translation term even
    when the latter looks comparable in millimetres.
    """
    rng = np.random.default_rng() if rng is None else rng
    omega = np.deg2rad(rng.normal(0.0, rot_sigma_deg, size=3))
    theta = np.linalg.norm(omega)
    T = np.eye(4)
    if theta > 1e-12:
        k = omega / theta
        K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
        T[:3, :3] = np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)
    T[:3, 3] = rng.normal(0.0, trans_sigma_m, size=3)
    return T


def preprocess_camera_frame(
    points: np.ndarray,
    nominal_extrinsics: np.ndarray,
    cfg: PointCloudPreprocessConfig,
    num_points: int,
    margin: float = 0.05,
    device: str = "cuda",
    rgb_already_normalized: bool = False,
    max_input_points: Optional[int] = None,
    rng=None,
) -> np.ndarray:
    """Conversion-time per-camera reduction. Returns (num_points, 6) in CAMERA frame.

    The workspace crop still happens in the base frame -- using the *nominal*
    extrinsics and a box grown by `margin` -- so that points which a plausible
    perturbation would later pull into the workspace are not thrown away now.
    Only the crop uses the base frame; the points that come back are untouched
    camera-frame coordinates, because the perturbation has to be applied to
    them later as an extrinsic.
    """
    points = np.asarray(points, dtype=np.float64)
    assert points.ndim == 2 and points.shape[1] >= 6, (
        f"expected (N, 6) xyz+rgb, got {points.shape}"
    )

    xyz_cam = points[:, :3] * cfg.point_scale
    rgb = points[:, 3:6]
    if not rgb_already_normalized:
        rgb = rgb / cfg.rgb_max

    T = np.asarray(nominal_extrinsics, dtype=np.float64)
    xyz_base = xyz_cam @ T[:3, :3].T + T[:3, 3]

    lo, hi = cfg.bounds
    keep = np.all((xyz_base > lo - margin) & (xyz_base < hi + margin), axis=-1)
    # Floor cut, in the BASE frame like the box test, applied to the same mask so
    # the points that come back are still untouched camera-frame coordinates.
    # Runs BEFORE the FPS below, which is the whole point: sampling first and
    # cutting after would spend the per-camera budget on table and leave far too
    # few points (measured 3520 of 15360 survive that ordering). Cutting first
    # means the budget lands entirely on dynamic content.
    if cfg.floor_normal is not None:
        n = np.asarray(cfg.floor_normal, dtype=np.float64)
        keep &= (xyz_base @ n + float(cfg.floor_offset)) > cfg.floor_margin
    kept = np.concatenate([xyz_cam[keep], rgb[keep]], axis=-1)

    if len(kept) == 0:
        # A camera seeing NOTHING inside the workspace is normal, not a fault.
        # The wrist cameras ride on the arms, so when an arm lifts or turns away
        # the box contains none of what it sees -- and the floor cut removes the
        # table that used to be there. Measured on the 30-episode refine set:
        # 2 of 30 episodes, always the left wrist, 8 and 94 frames.
        #
        # Emit points placed far OUTSIDE the workspace in world coordinates, so
        # `fuse_cameras`' crop deletes them and this camera contributes nothing
        # for this frame -- which is the truth. The pool is then short and the
        # existing padding path fills it from the cameras that did see something.
        #
        # Returned in CAMERA frame like everything else, so the per-camera
        # perturbation can still be applied later without special-casing.
        # xyz_base = xyz_cam @ R.T + t  =>  xyz_cam = (xyz_base - t) @ R
        # This mirrors what the inference builder already does (it skips an
        # empty camera and reports it via FrameDiag.cam_short); conversion
        # should not be stricter than inference.
        far = (hi + 100.0) - T[:3, 3]
        sentinel = np.tile(far @ T[:3, :3], (num_points, 1))
        cprint(f"[real_preprocess] no points inside the workspace for this "
               f"camera/frame; contributing nothing", "yellow")
        return np.concatenate(
            [sentinel, np.zeros((num_points, 3))], axis=-1).astype(np.float32)

    kept = pad_to_min_points(kept, num_points)

    # Farthest-point sampling is O(N x K), so running it on a raw 170k-point
    # cloud costs ~875M distance evaluations per frame per camera. Thinning to a
    # few times the target first makes conversion tractable and costs almost
    # nothing in coverage: the random subset is already dense relative to
    # `num_points`, which is all FPS needs to spread points evenly.
    if max_input_points is not None and len(kept) > max_input_points:
        rng = np.random.default_rng() if rng is None else rng
        kept = kept[rng.choice(len(kept), size=max_input_points, replace=False)]

    sampled_xyz, idx = farthest_point_sample(kept, num_points, device=device)
    out = np.concatenate([sampled_xyz, kept[idx, 3:6]], axis=-1).astype(np.float32)
    np.clip(out[:, 3:6], 0.0, 1.0, out=out[:, 3:6])
    return out


def fuse_cameras(
    clouds_cam: Sequence[np.ndarray],
    extrinsics: Sequence[np.ndarray],
    cfg: PointCloudPreprocessConfig,
    method: str = "random",
    rng=None,
    device: str = "cuda",
) -> np.ndarray:
    """Fuse per-camera clouds into one (cfg.num_points, 6) cloud in base frame.

    `clouds_cam[i]` is (M, 6) in camera i's frame; `extrinsics[i]` is the 4x4
    to apply -- nominal at inference, nominal composed with a perturbation at
    training time. `method` selects the final downsample:

    - "random": uniform subsample. Cheap enough to run in a dataloader worker,
      and the resampling is itself a mild augmentation.
    - "fps": farthest-point. Uniform density, but pays a pytorch3d call per
      frame per sample -- with horizon 16 that is 16 calls per __getitem__.

    Whichever you pick, **inference must pick the same one**: the density of the
    cloud handed to the encoder is part of the observation.
    """
    rng = np.random.default_rng() if rng is None else rng

    transformed = []
    for cloud, T in zip(clouds_cam, extrinsics):
        T = np.asarray(T, dtype=np.float64)
        xyz = np.asarray(cloud[:, :3], dtype=np.float64) @ T[:3, :3].T + T[:3, 3]
        transformed.append(np.concatenate([xyz, cloud[:, 3:6]], axis=-1))

    stacked = np.concatenate(transformed, axis=0)
    fused = crop_to_workspace(stacked, cfg)
    if len(fused) == 0 and not np.isfinite(stacked).all():
        # Diagnose before pad_to_min_points' generic message. NaN compares false
        # in every crop test, so a NaN cloud is indistinguishable from a badly
        # placed workspace box -- and the generic message sends you to check the
        # extrinsics, which is the wrong place entirely.
        raise ValueError(
            "NaN in the camera clouds before fusion. SequenceSampler's "
            "`key_first_k` NaN-fills the frames it did not load, so this almost "
            "always means more observation frames were read than were loaded -- "
            "see RealMultiCamDataset._key_first_k and n_obs_steps."
        )
    fused = pad_to_min_points(fused, cfg.num_points)

    if method == "fps":
        xyz, idx = farthest_point_sample(fused, cfg.num_points, device=device)
        out = np.concatenate([xyz, fused[idx, 3:6]], axis=-1)
    elif method == "random":
        idx = rng.choice(len(fused), size=cfg.num_points, replace=False)
        out = fused[idx]
    else:
        raise ValueError(f"unknown fuse method '{method}' (expected 'random' or 'fps')")
    return np.ascontiguousarray(out, dtype=np.float32)


def workspace_limits(cfg: PointCloudPreprocessConfig):
    """Analytic (min, max) per channel of any fused cloud, as 6-vectors.

    Everything `fuse_cameras` returns has been cropped to the workspace box and
    has rgb in [0, 1], so the normalizer's limits are known from the config and
    need not be estimated from the data. That keeps normalization identical
    between training and inference even if they see different scenes.
    """
    lo, hi = cfg.bounds
    return (
        np.concatenate([lo, np.zeros(3)]).astype(np.float32),
        np.concatenate([hi, np.ones(3)]).astype(np.float32),
    )


# ------------------------------------------------------- Franka Panda FK ----
#
# Needed to place a wrist camera: its extrinsic is given relative to link 8, so
# the base-frame pose requires T_link0<-link8(t). Going via the recorded EE pose
# instead would inherit whatever world<-link0 offset the collection stack
# assumed, which is not necessarily the measured one.
#
# Verified against this rig: FK + world<-link0 + flange->ee reproduces the
# recorded `qpos_arm` to 0.02 mm / 0.002 deg on a stationary arm.

# Modified (Craig) DH: (a, d, alpha) per joint, then the fixed link7->link8 hop.
_PANDA_DH = [
    (0.0, 0.333, 0.0),
    (0.0, 0.0, -np.pi / 2),
    (0.0, 0.316, np.pi / 2),
    (0.0825, 0.0, np.pi / 2),
    (-0.0825, 0.384, -np.pi / 2),
    (0.0, 0.0, np.pi / 2),
    (0.088, 0.0, np.pi / 2),
]
_PANDA_FLANGE = (0.0, 0.107, 0.0)


def _dh_transform(a, d, alpha, theta):
    ca, sa = np.cos(alpha), np.sin(alpha)
    ct, st = np.cos(theta), np.sin(theta)
    return np.array([
        [ct, -st, 0.0, a],
        [st * ca, ct * ca, -sa, -d * sa],
        [st * sa, ct * sa, ca, d * ca],
        [0.0, 0.0, 0.0, 1.0],
    ])


def panda_fk_flange(joints: Sequence[float]) -> np.ndarray:
    """T_link0<-link8 for a Franka Panda, from its 7 joint angles (radians)."""
    joints = np.asarray(joints, dtype=np.float64).reshape(-1)
    if joints.shape != (7,):
        raise ValueError(f"Panda FK needs 7 joint angles, got {joints.shape}")
    T = np.eye(4)
    for (a, d, alpha), theta in zip(_PANDA_DH, joints):
        T = T @ _dh_transform(a, d, alpha, theta)
    a, d, alpha = _PANDA_FLANGE
    return T @ _dh_transform(a, d, alpha, 0.0)


def depth_to_point_cloud(
    depth: np.ndarray,
    rgb: np.ndarray,
    intrinsics: Sequence[float],
    depth_scale: float = 1.0,
) -> np.ndarray:
    """Optional step 0: unproject an organised depth+rgb pair into (N, 6).

    Only needed when the recording stores depth images rather than clouds.
    ``intrinsics`` is (fx, fy, cx, cy); ``depth_scale`` takes raw depth to the
    same unit the extrinsics expect. Invalid (<=0) depths are dropped.
    """
    fx, fy, cx, cy = intrinsics
    h, w = depth.shape[:2]
    vs, us = np.mgrid[0:h, 0:w]
    z = depth.astype(np.float64) * depth_scale
    valid = z > 0
    z = z[valid]
    x = (us[valid] - cx) * z / fx
    y = (vs[valid] - cy) * z / fy
    colors = rgb.reshape(h, w, -1)[valid][:, :3]
    return np.concatenate([np.stack([x, y, z], axis=-1), colors], axis=-1)
