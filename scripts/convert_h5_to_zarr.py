"""Convert ros2_ws per-episode HDF5 recordings into the R3D zarr ReplayBuffer.

This replaces scripts/convert_real_robot_data.py, which was an unmodified DP3
leftover with another lab's paths, extrinsics and depth scale hardcoded and no
argparse. The transform lives in `r3d.common.real_preprocess` so the inference
repo imports the exact same code.

Clouds are stored **per camera, in that camera's own frame**, alongside the
nominal extrinsics. They are fused at training time under a random per-camera
perturbation of those extrinsics -- see r3d/dataset/real_multicam_dataset.py.
Fusing here instead would bake the calibration in and make misregistration
impossible to augment.

Usage
-----
    # 1. discover the layout of a recording
    python scripts/convert_h5_to_zarr.py --inspect data/raw/episode_000.h5

    # 2. fill in R3D/r3d/config/preprocess/real_bimanual.yaml, then
    python scripts/convert_h5_to_zarr.py \
        --input-dir data/raw \
        --output R3D/data/real_bimanual.zarr \
        --config R3D/r3d/config/preprocess/real_bimanual.yaml

One HDF5 file == one episode. `img` is deliberately not written: R3D reads
point clouds only (RobotwinDataset has the `img` key commented out).
"""

import argparse
import glob
import os
import sys

import h5py
import numpy as np
import yaml
import zarr
from termcolor import cprint

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "R3D"))

from r3d.common.real_preprocess import (  # noqa: E402
    PointCloudPreprocessConfig,
    depth_to_point_cloud,
    panda_fk_flange,
    preprocess_camera_frame,
)


# ---------------------------------------------------------------- inspection --

def inspect_h5(path):
    """Print the dataset tree so the config's key paths can be filled in."""
    cprint(f"\n{path}", "cyan")
    with h5py.File(path, "r") as f:
        def visit(name, obj):
            if isinstance(obj, h5py.Dataset):
                print(f"  {name:<60} {str(obj.shape):<24} {obj.dtype}")
            else:
                print(f"  {name}/")
        f.visititems(visit)
        if f.attrs:
            cprint("  attrs:", "cyan")
            for k, v in f.attrs.items():
                print(f"    {k} = {v}")


# ------------------------------------------------------------------- loading --

def _read(f, path):
    if path not in f:
        raise KeyError(
            f"'{path}' not found in {f.filename}. "
            f"Run with --inspect to print the actual layout."
        )
    return f[path]


def _first_data_path(spec):
    """`data` may be a single dataset path or a list to concatenate."""
    d = spec.get("data") or spec.get("xyz")
    if d is None:
        raise KeyError(f"stream spec has neither `data` nor `xyz`: {spec}")
    return d[0] if isinstance(d, (list, tuple)) else d


def stream_length(f, spec):
    node = _read(f, _first_data_path(spec))
    return len(node.keys()) if isinstance(node, h5py.Group) else node.shape[0]


def load_stamps(f, spec, n_frames):
    """Timestamps in seconds, or a synthetic index when the stream has none."""
    if not spec.get("stamps"):
        return np.arange(n_frames, dtype=np.float64)
    t = np.asarray(_read(f, spec["stamps"])[...], dtype=np.float64).reshape(-1)
    if t.max() > 1e12:      # rosbag2 stamps are often integer nanoseconds
        t = t / 1e9
    if len(t) != n_frames:
        raise ValueError(
            f"{spec['stamps']} has {len(t)} stamps but its data stream has "
            f"{n_frames} frames"
        )
    return t


def frame_keys(node):
    """Per-frame datasets sorted numerically, so frame_9 precedes frame_10."""
    def as_int(k):
        digits = "".join(c for c in k if c.isdigit())
        return int(digits) if digits else 0
    return sorted(node.keys(), key=as_int)


def _read_one(f, path, i):
    node = _read(f, path)
    if isinstance(node, h5py.Group):
        return np.asarray(node[frame_keys(node)[i]][...])
    return np.asarray(node[i])


def read_frame(f, spec, i):
    """One frame of a stream.

    Handles (T, ...) arrays, groups of per-frame datasets, and a *list* of
    dataset paths, which are concatenated along the feature axis. The list form
    is how the bimanual state is assembled: left arm, right arm, left hand,
    right hand are four separate datasets in the recording.
    """
    d = spec["data"]
    if isinstance(d, (list, tuple)):
        return np.concatenate(
            [np.asarray(_read_one(f, p, i)).reshape(-1) for p in d], axis=0)
    return _read_one(f, d, i)


def read_point_cloud_frame(f, spec, i):
    """Raw (N, 6) xyz+rgb for frame `i`.

    `layout: xyz_rgb_split` is the ros2_ws format: separate `xyz` and `rgb`
    datasets, both allocated to a per-episode maximum and ZERO-PADDED, with
    `num_points[i]` giving the valid count. Slicing to that count is mandatory --
    the padding is (0, 0, 0), which sits inside any workspace box containing the
    origin and would survive the crop as a phantom blob at the robot base.
    """
    layout = spec.get("layout", "xyzrgb")

    if layout == "xyz_rgb_split":
        xyz = _read_one(f, spec["xyz"], i)
        rgb = _read_one(f, spec["rgb"], i)
        if spec.get("num_points"):
            n = int(np.asarray(_read(f, spec["num_points"])[i]))
            xyz, rgb = xyz[:n], rgb[:n]
        return np.concatenate([np.asarray(xyz, np.float64),
                               np.asarray(rgb, np.float64)], axis=-1)

    if layout == "depth":
        return depth_to_point_cloud(
            read_frame(f, spec, i),
            _read_one(f, spec["rgb"], i),
            intrinsics=spec["intrinsics"],
            depth_scale=spec.get("depth_scale", 1.0),
        )

    pc = read_frame(f, spec, i)
    if pc.ndim != 2 or pc.shape[1] < 6:
        raise ValueError(
            f"expected (N, >=6) xyz+rgb per frame, got {pc.shape}. Use "
            "`layout: xyz_rgb_split` for separate xyz/rgb datasets, or "
            "`layout: depth` for depth images."
        )
    return pc


# --------------------------------------------------------------------- sync --

def nearest_index(ref_t, other_t, max_dt):
    """Match each reference stamp to the closest stamp in another stream.

    Returns (indices, valid_mask). Reference frames with no partner inside
    `max_dt` are dropped from the episode -- with several cameras this is the
    main thing standing between you and silently misaligned observations.
    """
    order = np.argsort(other_t)
    sorted_t = other_t[order]
    if len(sorted_t) == 1:
        idx = np.zeros(len(ref_t), dtype=int)
    else:
        pos = np.clip(np.searchsorted(sorted_t, ref_t), 1, len(sorted_t) - 1)
        left, right = pos - 1, pos
        take_left = np.abs(ref_t - sorted_t[left]) <= np.abs(sorted_t[right] - ref_t)
        idx = np.where(take_left, left, right)
    return order[idx], np.abs(ref_t - sorted_t[idx]) <= max_dt


# --------------------------------------------------------------- conversion --

def as_matrix(pose, quat_order="xyzw"):
    """A recorded pose -> 4x4. Accepts a 4x4, or a 7-vector of xyz + quaternion.

    `quat_order` must be stated explicitly and matched to the recording: the two
    conventions differ only in element order, so getting it wrong yields a
    perfectly valid-looking -- orthonormal, det +1 -- rotation that is simply
    wrong. Note this function normalizes the quaternion, so a well-formed matrix
    is NOT evidence that the order is right; check the resulting axes against a
    pose you know instead. ROS 2's geometry_msgs/Quaternion is x, y, z, w.
    """
    pose = np.asarray(pose, dtype=np.float64).squeeze()
    if pose.shape == (4, 4):
        return pose
    if pose.shape == (16,):
        return pose.reshape(4, 4)
    if pose.shape != (7,):
        raise ValueError(
            f"cannot read a transform from shape {pose.shape}; expected a 4x4 "
            "or a 7-vector of xyz + quaternion"
        )
    if quat_order == "wxyz":
        t, (w, x, y, z) = pose[:3], pose[3:]
    elif quat_order == "xyzw":
        t, (x, y, z, w) = pose[:3], pose[3:]
    else:
        raise ValueError(f"quat_order must be 'wxyz' or 'xyzw', got {quat_order!r}")
    n = np.sqrt(x * x + y * y + z * z + w * w)
    if n < 1e-9:
        raise ValueError("zero-norm quaternion in recorded pose")
    x, y, z, w = x / n, y / n, z / n, w / n
    T = np.eye(4)
    T[:3, :3] = np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w),     2 * (x * z + y * w)],
        [2 * (x * y + z * w),     1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w),     2 * (y * z + x * w),     1 - 2 * (x * x + y * y)],
    ])
    T[:3, 3] = t
    return T


def flange_stream_name(cam_name):
    return f"{cam_name}__flange"


def collect_streams(cfg):
    """Flatten cameras, their flange poses, and proprioception into one mapping."""
    streams, cam_names = {}, []
    for cam in cfg["cameras"]:
        streams[cam["name"]] = cam
        cam_names.append(cam["name"])
        if cam.get("mount", "static") == "wrist":
            if not cam.get("flange_pose"):
                raise KeyError(
                    f"camera '{cam['name']}' is mount: wrist but has no "
                    "`flange_pose` stream. A wrist camera's base-frame extrinsic "
                    "changes every frame; without T_base<-flange(t) it cannot be "
                    "reconstructed."
                )
            fp = dict(cam["flange_pose"])
            if fp.get("source") == "fk":
                fp.setdefault("data", fp["joints"])
            streams[flange_stream_name(cam["name"])] = fp
    for name, spec in cfg.get("h5", {}).items():
        if spec:
            streams[name] = spec
    return streams, cam_names


def convert_episode(path, cfg, pc_cfg, streams, cam_names, device):
    """One HDF5 file -> (per-camera clouds, state, action, target_ee)."""
    ref_name = cfg["sync"]["reference"]
    max_dt = cfg["sync"]["max_dt"]
    if ref_name not in streams:
        raise KeyError(
            f"sync.reference '{ref_name}' is not a camera name or an h5 stream. "
            f"Known: {sorted(streams)}"
        )

    with h5py.File(path, "r") as f:
        stamps = {k: load_stamps(f, s, stream_length(f, s)) for k, s in streams.items()}
        ref_t = stamps[ref_name]

        indices = {ref_name: np.arange(len(ref_t))}
        valid = np.ones(len(ref_t), dtype=bool)
        for name, t in stamps.items():
            if name == ref_name:
                continue
            idx, ok = nearest_index(ref_t, t, max_dt)
            indices[name] = idx
            valid &= ok

        # Decimate to a lower control rate. The frame rate is not free: horizon,
        # n_obs_steps and n_action_steps are counted in FRAMES, so it sets how
        # far ahead the policy plans and how long it commits open-loop. Halving
        # the rate doubles both. Record at the higher rate and decimate here --
        # the reverse is not possible.
        stride = int(cfg.get("frame_stride") or 1)

        if cfg["action"]["source"] == "next_state":
            # action[t] := state[t+stride], so the tail has no successor
            valid[-stride:] = False

        keep = np.flatnonzero(valid)[::stride]
        if len(keep) == 0:
            raise ValueError(
                f"{path}: no frame survived time alignment. Check sync.max_dt "
                f"({max_dt}s) and that the stamp datasets are the right ones."
            )
        if len(keep) < len(ref_t):
            cprint(f"  dropped {len(ref_t) - len(keep)}/{len(ref_t)} unsynced frames", "yellow")

        clouds, nominals = {}, {}
        for name in cam_names:
            spec = streams[name]
            hand_eye = np.asarray(spec["extrinsics"], dtype=np.float64)

            if spec.get("mount", "static") == "wrist":
                # T_base<-cam(t) = T_base<-ee(t) @ T_ee<-flange @ T_flange<-cam.
                # Only the last factor is miscalibrated, and it is what the
                # training-time perturbation acts on (applied on the right) --
                # see RealMultiCamDataset._sample_deltas. The recording gives
                # T_base<-ee(t) as an EE pose; T_ee<-flange is a fixed, known
                # offset down the kinematic chain.
                fspec = streams[flange_stream_name(name)]
                sname = flange_stream_name(name)

                if fspec.get("source") == "fk":
                    # T_world<-cam(t) = T_world<-link0 @ T_link0<-link8(t) @ T_link8<-cam
                    # Every factor measured; the recorded EE pose never enters, so
                    # whatever world<-link0 offset the collection stack assumed is
                    # irrelevant here.
                    base = np.asarray(fspec["base"], dtype=np.float64)
                    lo, hi = fspec["joint_slice"]
                    nominals[name] = np.stack([
                        base @ panda_fk_flange(read_frame(f, fspec, indices[sname][i])[lo:hi])
                        @ hand_eye
                        for i in keep
                    ])
                else:
                    # Recorded EE pose route: T_world<-ee(t) @ T_ee<-flange @ T_flange<-cam
                    quat_order = fspec.get("quat_order", "xyzw")
                    ee_to_flange = np.asarray(
                        spec.get("ee_to_flange", np.eye(4)), dtype=np.float64)
                    nominals[name] = np.stack([
                        as_matrix(read_frame(f, fspec, indices[sname][i]),
                                  quat_order=quat_order)
                        @ ee_to_flange @ hand_eye
                        for i in keep
                    ])
            else:
                nominals[name] = None

            clouds[name] = np.stack([
                preprocess_camera_frame(
                    read_point_cloud_frame(f, spec, indices[name][i]),
                    nominal_extrinsics=(hand_eye if nominals[name] is None
                                        else nominals[name][j]),
                    cfg=pc_cfg,
                    num_points=cfg["points_per_camera"],
                    margin=cfg["crop_margin"],
                    device=device,
                    max_input_points=cfg.get("max_input_points"),
                    rgb_already_normalized=(pc_cfg.rgb_max == 1.0),
                )
                for j, i in enumerate(keep)
            ])

        def gather(name, offset=0):
            spec = streams[name]
            return np.stack([
                np.asarray(read_frame(f, spec, indices[name][i + offset])).reshape(-1)
                for i in keep
            ]).astype(np.float32)

        state = gather("state")
        action = gather("state", offset=stride) if cfg["action"]["source"] == "next_state" \
            else gather("action")
        target_ee = gather("target_ee") if "target_ee" in streams else None

    return clouds, nominals, state, action, target_ee


def canonicalize_quaternions(arr, slices, label):
    """Resolve the q == -q double cover, in place, across the whole dataset.

    Both signs encode the same rotation, so a dataset containing each for
    similar poses gives the L2 diffusion loss two contradictory targets, and it
    averages them toward a degenerate near-zero quaternion.

    The sign is chosen so the component with the largest mean magnitude is
    positive. Do NOT canonicalize on w: these are flange-down poses with
    w ~ 0 (measured: -0.008 on the right arm), so a w >= 0 rule would flip on
    sensor noise -- which is the very failure being prevented.
    """
    for lo, hi in slices:
        q = arr[:, lo:hi]
        if hi - lo != 4:
            raise ValueError(f"quaternion slice [{lo}, {hi}) is not 4 wide")
        dominant = int(np.argmax(np.abs(q).mean(axis=0)))
        flip = q[:, dominant] < 0
        if flip.any():
            q[flip] *= -1.0
            cprint(f"  canonicalized {label}[{lo}:{hi}]: flipped {flip.sum()}/{len(q)} "
                   f"frames on component {dominant}", "yellow")
    return arr


def write_zarr(out_path, clouds, nominals, states, actions, target_ees,
               episode_ends, cfg, cam_names):
    if os.path.exists(out_path):
        cprint(f"{out_path} already exists -- delete it first, refusing to overwrite.", "red")
        sys.exit(1)

    root = zarr.group(out_path)
    data, meta = root.create_group("data"), root.create_group("meta")
    comp = zarr.Blosc(cname="zstd", clevel=3, shuffle=1)

    def put(group, name, arr, dtype="float32"):
        group.create_dataset(name, data=arr, dtype=dtype,
                             chunks=(100,) + arr.shape[1:], compressor=comp)

    for i, name in enumerate(cam_names):
        put(data, f"point_cloud_cam{i}", clouds[name])
        # The nominal extrinsics travel with the data: fusion cannot be
        # reproduced without them, at training time or on the robot. A wrist
        # camera needs one matrix per frame, a static one needs a single matrix.
        if nominals[name] is not None:
            data.create_dataset(f"extrinsics_cam{i}", data=nominals[name],
                                dtype="float64", chunks=(100, 4, 4), compressor=comp)
        else:
            meta.create_dataset(
                f"extrinsics_cam{i}",
                data=np.asarray(cfg["cameras"][i]["extrinsics"], dtype=np.float64),
                dtype="float64")
    put(data, "state", states)
    put(data, "action", actions)
    if target_ees is not None:
        put(data, "target_ee", target_ees)
    meta.create_dataset("episode_ends", data=episode_ends, dtype="int64", chunks=(100,))
    meta.attrs["camera_names"] = list(cam_names)
    # Inference MUST run at the rate the policy was trained on: executing
    # n_action_steps predicted steps assumes each lasts 1/control_hz seconds, so
    # a mismatch makes the arm traverse the chunk at the wrong speed.
    if cfg.get("_control_hz"):
        meta.attrs["control_hz"] = float(cfg["_control_hz"])
    # The inference repo has to slice the predicted 48-vector the same way this
    # one assembled it. Record the layout with the data rather than relying on
    # both sides agreeing from memory.
    for key in ("state", "action"):
        paths = cfg["h5"][key]["data"]
        if isinstance(paths, (list, tuple)):
            meta.attrs[f"{key}_layout"] = list(paths)
    if cfg.get("quaternion_slices"):
        meta.attrs["quaternion_slices"] = [list(s) for s in cfg["quaternion_slices"]]
        meta.attrs["quaternion_order"] = "xyzw"


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--inspect", metavar="H5", help="print one file's layout and exit")
    p.add_argument("--input-dir", help="directory of per-episode .h5 files")
    p.add_argument("--pattern", default="*.h5", help="glob inside --input-dir")
    p.add_argument("--output", help="destination .zarr path")
    p.add_argument("--config", default="R3D/r3d/config/preprocess/real_bimanual.yaml")
    p.add_argument("--device", default="cuda", help="device for farthest-point sampling")
    p.add_argument("--max-episodes", type=int, default=None, help="stop early (smoke tests)")
    args = p.parse_args()

    if args.inspect:
        inspect_h5(args.inspect)
        return
    if not (args.input_dir and args.output):
        p.error("--input-dir and --output are required unless --inspect is given")

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    pc_cfg = PointCloudPreprocessConfig.from_yaml(args.config)
    streams, cam_names = collect_streams(cfg)

    for cam in cfg["cameras"]:
        if np.allclose(np.asarray(cam["extrinsics"], dtype=float), np.eye(4)):
            cprint(f"{cam['name']} extrinsics are identity -- is the ChArUco solve "
                   "filled in?", "yellow")

    files = sorted(glob.glob(os.path.join(args.input_dir, args.pattern)))
    if args.max_episodes:
        files = files[: args.max_episodes]
    if not files:
        cprint(f"no files matching {args.pattern} in {args.input_dir}", "red")
        sys.exit(1)

    stride = int(cfg.get("frame_stride") or 1)
    with h5py.File(files[0], "r") as f0:
        source_hz = f0.attrs.get("sampling_frequency_hz")
    if source_hz:
        cfg["_control_hz"] = float(source_hz) / stride
        cprint(f"control rate: {source_hz} Hz / stride {stride} = "
               f"{cfg['_control_hz']:.1f} Hz  "
               f"(horizon 16 = {16 / cfg['_control_hz']:.2f} s, "
               f"n_action_steps 8 = {8 / cfg['_control_hz']:.2f} s open-loop)", "cyan")

    per_cam = {name: [] for name in cam_names}
    per_nominal = {name: [] for name in cam_names}
    states, actions, target_ees, episode_ends, total = [], [], [], [], 0
    for path in files:
        cprint(f"[{len(episode_ends) + 1}/{len(files)}] {path}", "green")
        clouds, nominals, st, ac, ee = convert_episode(
            path, cfg, pc_cfg, streams, cam_names, args.device)
        for name in cam_names:
            per_cam[name].append(clouds[name])
            if nominals[name] is not None:
                per_nominal[name].append(nominals[name])
        states.append(st)
        actions.append(ac)
        if ee is not None:
            target_ees.append(ee)
        total += len(st)
        episode_ends.append(total)  # cumulative, per the ReplayBuffer format

    all_states, all_actions = np.concatenate(states), np.concatenate(actions)
    if cfg.get("quaternion_slices"):
        qs = cfg["quaternion_slices"]
        canonicalize_quaternions(all_states, qs, "state")
        canonicalize_quaternions(all_actions, qs, "action")

    write_zarr(
        args.output,
        {name: np.concatenate(v) for name, v in per_cam.items()},
        {name: (np.concatenate(v) if v else None) for name, v in per_nominal.items()},
        all_states, all_actions,
        np.concatenate(target_ees) if target_ees else None,
        np.asarray(episode_ends, dtype=np.int64), cfg, cam_names)

    cprint(f"\nwrote {args.output}", "green")
    cprint(f"  episodes    {len(episode_ends)}", "green")
    cprint(f"  frames      {total}", "green")
    for i, name in enumerate(cam_names):
        mount = cfg["cameras"][i].get("mount", "static")
        cprint(f"  cam{i} ({name})  {per_cam[name][0].shape[1:]} per frame, "
               f"camera frame, {mount} mount", "green")
    cprint(f"  state       D_state  = {states[0].shape[1]}", "green")
    cprint(f"  action      D_action = {actions[0].shape[1]}", "green")
    cprint(f"\nNext: python scripts/inspect_zarr.py {args.output}", "cyan")


if __name__ == "__main__":
    main()
