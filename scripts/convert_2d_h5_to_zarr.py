#!/usr/bin/env python3
"""Build the 2D image zarr from the synced 322x182 HDF5 episodes.

The 2D counterpart of `convert_h5_to_zarr.py`. That script is NOT modified;
this one imports its stream helpers so the state and action vectors are
assembled by the same code, from the same config block, as the point-cloud run
they are compared against.

    python scripts/convert_2d_h5_to_zarr.py \
        --input-dir ~/ros2_ws/tmp/1-90_RED_2d_322x182 \
        --output R3D/data/real_bimanual_2d_red90.zarr

`--input-dir` may be repeated; episodes are ordered by directory, then by
sorted filename within each.

Layout written:

    data/image_cam{i}   uint8   (N, H, W, 3)   RGB, one chunk PER FRAME
    data/state          float32 (N, 48)
    data/action         float32 (N, 48)
    data/joint_states   float32 (N, 14)
    meta/episode_ends   int64

Two choices worth not undoing:

* **Episodes are streamed to disk, not accumulated.** 150 episodes is ~28.5 GB
  of pixels, and building it in RAM before one big `np.concatenate` peaks near
  twice that on a 62 GB box. Each episode is appended to the zarr as it is
  converted; only the 48-vectors are held in memory, which is ~24 MB.
* **One chunk per frame.** The training dataset opens this zarr with
  `ReplayBuffer.create_from_path` and reads it off disk rather than loading it
  resident -- 3 cameras of 322x182 uint8 over a full collection is ~57 GB, and
  the box has 62 GB. With the 100-frame chunks the 3D converter uses, reading a
  2-frame window would decompress 100 frames per camera.
* **No `extrinsics_cam*`.** Unlike the point-cloud zarr, nothing here needs a
  per-frame transform: the only consumer is the pose tag, which the dataset
  composes from `joint_states` and the measured constants in the preprocess
  config. Storing the joint angles instead keeps the mount in config, where the
  kinematic ablation can override it.
"""

import argparse
import os
import sys

import h5py
import numpy as np
import yaml
import zarr
from termcolor import cprint

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# Read-only reuse. Assembling state/action a second way is a second chance to
# differ from the run this dataset exists to be compared against.
from convert_h5_to_zarr import (  # noqa: E402
    canonicalize_quaternions,
    load_stamps,
    read_frame,
    stream_length,
)


def load_config(path):
    """Load the 2D config, merging the point-cloud config it inherits."""
    with open(path) as f:
        cfg = yaml.safe_load(f)
    inherit = cfg.get("inherit")
    if inherit:
        with open(inherit) as f:
            base = yaml.safe_load(f)
        base.update({k: v for k, v in cfg.items() if k != "inherit"})
        cfg = base
    for required in ("images", "joint_states", "cameras", "h5"):
        if required not in cfg:
            raise KeyError(f"{path} (after inherit) has no `{required}` block")
    return cfg


def convert_episode(path, cfg, cam_names, image_paths):
    """One HDF5 file -> (per-camera images, state, action, joint_states)."""
    stride = int(cfg.get("frame_stride") or 1)

    with h5py.File(path, "r") as f:
        state_spec = cfg["h5"]["state"]
        n_frames = stream_length(f, state_spec)
        ref_t = load_stamps(f, state_spec, n_frames)

        # The 2D h5 is written by syncronize_data_2d.py, which takes its time
        # grid verbatim from the 3D output -- every stream here shares one
        # `timestamps_ns`. So alignment is an identity mapping and the only
        # useful check is that the lengths actually agree. A silent length
        # mismatch is how images and actions come apart by one frame.
        for name, ipath in image_paths.items():
            n_img = f[ipath].shape[0]
            if n_img != n_frames:
                raise ValueError(
                    f"{path}: camera '{name}' has {n_img} frames but state has "
                    f"{n_frames}. These files are pre-synced; a mismatch means "
                    "the wrong file pairing, not a sync problem to paper over.")

        if cfg["action"]["source"] == "next_state":
            keep = np.arange(0, n_frames - stride, stride)
        else:
            keep = np.arange(0, n_frames, stride)

        images = {}
        for name in cam_names:
            arr = f[image_paths[name]]
            images[name] = np.asarray(arr[...])[keep]
            if images[name].dtype != np.uint8:
                raise TypeError(
                    f"{path}: '{image_paths[name]}' is {images[name].dtype}, "
                    "expected uint8 RGB as the 2D sync writes it.")
            if images[name].ndim != 4 or images[name].shape[-1] != 3:
                raise ValueError(
                    f"{path}: '{image_paths[name]}' has shape "
                    f"{images[name].shape}, expected (T, H, W, 3).")

        def gather(spec, offset=0):
            return np.stack([
                np.asarray(read_frame(f, spec, i + offset)).reshape(-1)
                for i in keep
            ]).astype(np.float32)

        state = gather(state_spec)
        if cfg["action"]["source"] == "next_state":
            action = gather(state_spec, offset=stride)
        else:
            action = gather(cfg["h5"]["action"])

        joints = np.asarray(f[cfg["joint_states"]][...], dtype=np.float32)[keep]

    return images, state, action, joints


class ZarrWriter:
    """Streams episodes to disk as they are converted.

    The point-cloud converter can afford to build everything in memory and
    write once; 150 episodes of three-camera 322x182 uint8 is ~28.5 GB, and the
    concatenate step would need roughly twice that. So the image arrays are
    created on the first episode and appended to thereafter, and only the small
    per-frame vectors are accumulated -- ~24 MB for the whole set.
    """

    def __init__(self, out_path, cam_names):
        if os.path.exists(out_path):
            cprint(f"{out_path} already exists -- delete it first, refusing to "
                   "overwrite.", "red")
            sys.exit(1)
        self.out_path = out_path
        self.cam_names = list(cam_names)
        self.root = zarr.group(out_path)
        self.data = self.root.create_group("data")
        self.meta = self.root.create_group("meta")
        self.comp = zarr.Blosc(cname="zstd", clevel=3, shuffle=1)
        self.images = None
        self.state, self.action, self.joints, self.ends = [], [], [], []
        self.total = 0

    def add_episode(self, images, state, action, joints):
        if self.images is None:
            self.images = {}
            for i, name in enumerate(self.cam_names):
                shape = images[name].shape[1:]
                # One chunk per frame: the dataset reads 2-frame windows off
                # disk, so a 100-frame chunk would decompress 50x what is used.
                self.images[name] = self.data.create_dataset(
                    f"image_cam{i}", shape=(0,) + shape, chunks=(1,) + shape,
                    dtype="uint8", compressor=self.comp)
        for name in self.cam_names:
            self.images[name].append(images[name])
        self.state.append(state)
        self.action.append(action)
        self.joints.append(joints)
        self.total += len(state)
        self.ends.append(self.total)

    def finalize(self, cfg):
        states = np.concatenate(self.state, axis=0)
        actions = np.concatenate(self.action, axis=0)
        joints = np.concatenate(self.joints, axis=0)

        # Same canonicalisation as the point-cloud converter, over the same
        # slices: q and -q are one rotation, and a dataset holding both for
        # similar poses gives the L2 diffusion loss two contradictory targets.
        if cfg.get("quaternion_slices"):
            slices = [tuple(s) for s in cfg["quaternion_slices"]]
            canonicalize_quaternions(states, slices, "state")
            canonicalize_quaternions(actions, slices, "action")

        for name, arr in (("state", states), ("action", actions),
                          ("joint_states", joints)):
            self.data.create_dataset(name, data=arr, dtype="float32",
                                     chunks=(100,) + arr.shape[1:],
                                     compressor=self.comp)

        self.meta.create_dataset("episode_ends",
                                 data=np.asarray(self.ends, dtype=np.int64),
                                 dtype="int64", chunks=(100,))
        self.meta.attrs["camera_names"] = list(self.cam_names)
        first = self.images[self.cam_names[0]]
        self.meta.attrs["image_size"] = [int(first.shape[1]), int(first.shape[2])]
        if cfg.get("_control_hz"):
            self.meta.attrs["control_hz"] = float(cfg["_control_hz"])
        for key in ("state", "action"):
            paths = cfg["h5"][key]["data"]
            if isinstance(paths, (list, tuple)):
                self.meta.attrs[f"{key}_layout"] = list(paths)
        if cfg.get("quaternion_slices"):
            self.meta.attrs["quaternion_slices"] = [list(s) for s in cfg["quaternion_slices"]]
            self.meta.attrs["quaternion_order"] = "xyzw"
        return states, actions, joints


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input-dir", action="append", required=True,
                   help="directory of per-episode .h5 files (repeatable)")
    p.add_argument("--pattern", default="*.h5", help="glob inside each --input-dir")
    p.add_argument("--output", required=True, help="destination .zarr path")
    p.add_argument("--config", default="R3D/r3d/config/preprocess/real_bimanual_2d.yaml")
    p.add_argument("--max-episodes", type=int, default=None,
                   help="stop early (smoke tests)")
    args = p.parse_args()

    import glob

    cfg = load_config(args.config)
    cam_names = [c["name"] for c in cfg["cameras"]]
    image_paths = cfg["images"]
    missing = [n for n in cam_names if n not in image_paths]
    if missing:
        raise KeyError(
            f"{args.config} has no image path for {missing}. Every camera in the "
            "inherited `cameras` list needs one -- the order is positional.")

    files = []
    for d in args.input_dir:
        found = sorted(glob.glob(os.path.join(os.path.expanduser(d), args.pattern)))
        if not found:
            raise FileNotFoundError(f"no files matching {args.pattern} in {d}")
        files.extend(found)
    if args.max_episodes:
        files = files[:args.max_episodes]
    cprint(f"{len(files)} episodes from {len(args.input_dir)} directories", "cyan")

    writer = ZarrWriter(args.output, cam_names)
    for k, path in enumerate(files):
        images, state, action, joints = convert_episode(
            path, cfg, cam_names, image_paths)
        writer.add_episode(images, state, action, joints)
        cprint(f"  [{k+1}/{len(files)}] {os.path.basename(path)}: "
               f"{len(state)} frames  (total {writer.total})", "green")
        sys.stdout.flush()

    states, actions, joints = writer.finalize(cfg)

    hw = writer.meta.attrs["image_size"]
    cprint(f"\nwrote {args.output}", "green")
    cprint(f"  {len(writer.ends)} episodes, {writer.total} frames, "
           f"{hw[0]}x{hw[1]} per camera", "green")
    cprint(f"  state {states.shape}, action {actions.shape}, "
           f"joint_states {joints.shape}", "green")


if __name__ == "__main__":
    main()
