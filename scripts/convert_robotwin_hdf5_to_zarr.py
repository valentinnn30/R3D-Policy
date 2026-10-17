#!/usr/bin/env python
"""RoboTwin2 collection (episode*.hdf5) -> R3D zarr ReplayBuffer.  Sim baseline.

    python scripts/convert_robotwin_hdf5_to_zarr.py \\
        --input R3D/r3d/env/robotwin2/data/place_empty_cup_right/sim_head_p1024 \\
        --output R3D/data/place_empty_cup_right-sim_head_p1024-50.zarr

Per episode (T recorded frames), following RoboTwin's own DP3 conversion:

    point_cloud[t] = pointcloud[t]              t = 0 .. T-2   (T-1, 1024, 6)
    state[t]       = joint_action/vector[t]                    (T-1, 14)
    action[t]      = joint_action/vector[t+1]                  (T-1, 14)
    point_label[t] = pointcloud_label[t]        if recorded    (T-1, 1024) int16

The action is the NEXT recorded joint state; the last frame has no successor
and is dropped. `meta.attrs` records the crop box (RobotwinSimDataset builds the
point-cloud normalizer from it), the label map, and the task config, so the
zarr carries what evaluation must reproduce.
"""

import argparse
import glob
import json
import os
import sys

import h5py
import numpy as np
import yaml
import zarr

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RT_DIR = os.path.join(ROOT, "R3D", "r3d", "env", "robotwin2")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, help="dir holding data/episode*.hdf5 (+ seed.txt)")
    ap.add_argument("--output", required=True)
    ap.add_argument("--task-config", default=None,
                    help="RoboTwin task config name; default = basename of --input")
    ap.add_argument("--max-episodes", type=int, default=None)
    a = ap.parse_args()

    files = sorted(glob.glob(os.path.join(a.input, "data", "episode*.hdf5")),
                   key=lambda p: int(os.path.basename(p)[7:-5]))
    if a.max_episodes:
        files = files[:a.max_episodes]
    if not files:
        sys.exit(f"no episode*.hdf5 under {a.input}/data")
    task_config = a.task_config or os.path.basename(os.path.normpath(a.input))
    with open(os.path.join(RT_DIR, "task_config", f"{task_config}.yml")) as f:
        cfg = yaml.safe_load(f)
    if cfg["domain_randomization"].get("random_table_height", 0):
        sys.exit("random_table_height != 0: the crop box moves per episode, a fixed normalizer is wrong")
    bbox = cfg.get("bbox", [[-0.6, -0.35, 0.7401], [0.6, 0.35, 2.0]])  # camera.py default

    pcs, states, actions, labels, ends = [], [], [], [], []
    for p in files:
        with h5py.File(p, "r") as f:
            pc = f["pointcloud"][()].astype(np.float32)
            vec = f["joint_action/vector"][()].astype(np.float32)
            lab = f["pointcloud_label"][()].astype(np.int16) if "pointcloud_label" in f else None
        if len(pc) != len(vec) or len(pc) < 2:
            sys.exit(f"{p}: {len(pc)} clouds vs {len(vec)} joint vectors")
        pcs.append(pc[:-1]); states.append(vec[:-1]); actions.append(vec[1:])
        if lab is not None:
            labels.append(lab[:-1])
        ends.append((ends[-1] if ends else 0) + len(pc) - 1)
    if labels and len(labels) != len(files):
        sys.exit("pointcloud_label present in some episodes only")

    data = {"point_cloud": np.concatenate(pcs), "state": np.concatenate(states),
            "action": np.concatenate(actions)}
    if labels:
        data["point_label"] = np.concatenate(labels)
    rgb = data["point_cloud"][..., 3:6]
    if rgb.min() < 0 or rgb.max() > 1:
        sys.exit(f"rgb outside [0, 1]: [{rgb.min()}, {rgb.max()}]")

    if os.path.exists(a.output):
        sys.exit(f"{a.output} exists; remove it first")
    root = zarr.group(a.output)
    comp = zarr.Blosc(cname="zstd", clevel=3, shuffle=1)
    g = root.create_group("data")
    for k, v in data.items():
        g.create_dataset(k, data=v, chunks=(100,) + v.shape[1:], compressor=comp)
    meta = root.create_group("meta")
    meta.create_dataset("episode_ends", data=np.asarray(ends, np.int64), chunks=(100,))

    # by file path: importing through the `envs` package would import all of RoboTwin
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "point_labels", os.path.join(RT_DIR, "envs", "utils", "point_labels.py"))
    point_labels = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(point_labels)
    CLASS_NAMES = point_labels.CLASS_NAMES
    meta.attrs.update({
        "source": "robotwin2",
        "task_config": task_config,
        "task_config_yaml": cfg,
        "pcd_crop_bbox": bbox,
        "pcd_down_sample_num": cfg["pcd_down_sample_num"],
        "point_label_classes": {str(k): v for k, v in CLASS_NAMES.items()},
        "state_layout": "joint_action/vector[t]: left 6 joints + gripper, right 6 joints + gripper",
        "action_layout": "joint_action/vector[t+1]",
        "seeds": [int(s) for s in open(os.path.join(a.input, "seed.txt")).read().split()][:len(files)]
        if os.path.exists(os.path.join(a.input, "seed.txt")) else None,
    })
    print(f"{len(files)} episodes, {ends[-1]} frames -> {a.output}")
    for k, v in data.items():
        print(f"  data/{k:<12} {v.shape} {v.dtype}")


if __name__ == "__main__":
    main()
