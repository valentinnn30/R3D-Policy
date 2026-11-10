#!/usr/bin/env python
"""Frame-0 reference clouds for the frozen-robot-points run (round 7).  Run in r3d_robosuite.

    python scripts/sim/make_frame0_reference.py                    # all demos -> sidecar npz
    python scripts/sim/make_frame0_reference.py --eval-scenes 200  # check only: eval-scene reset frames

Re-renders frame 0 of every demo from its hdf5 state with the SAME CloudBuilder as the
converter, at FPS-M (default 8192) instead of N = 4096, and asserts BIT FOR BIT that the first
N points and labels equal the stored frame-0 cloud (FPS is greedy with a deterministic start).
Also asserts every reference has >= N robot points (r3d.common.robot_freeze). Any mismatch
aborts before anything is written.

Output: <zarr without .zarr>.frame0_p<M>.npz with cloud (E, M, 6) float32, label (E, M) int16,
n_valid (E,), demos (E,) -- episode order of the zarr.

--eval-scenes K: reset the first K eval scenes (seed block 1, as eval_mimicgen_ckpt.py), build
the M reference and the N cloud, report the robot-point minimum and the prefix check. Writes
nothing (the evaluation re-checks both at every reset).
"""

import argparse
import copy
import json
import multiprocessing as mp
import os
import sys
import time

os.environ.setdefault("MUJOCO_GL", "egl")
import h5py
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
R3D_DIR = os.path.join(ROOT, "R3D")
sys.path.insert(0, R3D_DIR)
sys.path.insert(0, os.path.dirname(__file__))
ROBOT_LABELS = (5, 7)


def hdf5_path(task):
    return os.path.join(R3D_DIR, "data", "mimicgen", "core", f"{task}.hdf5")


def demo_worker(args):
    a, items, env_args = args
    import torch  # noqa: F401
    import zarr
    from r3d.common import robosuite_pcd as RP
    from mimicgen_view_check import make_env, reset_to_demo
    env = make_env(env_args, a.camera, a.res)
    z = zarr.open(a.zarr, "r")
    out = []
    with h5py.File(hdf5_path(a.task), "r") as f:
        for i, ep, start in items:
            g = f[f"data/demo_{ep}"]
            s0 = g["states"][0]
            reset_to_demo(env, g.attrs["model_file"], s0)
            env.sim.set_state_from_flattened(s0)
            env.sim.forward()
            obs = env._get_observations(force_update=True)
            lut = RP.geom_label_lut(env.sim, a.env_name)
            ref, ref_lab, n = RP.CloudBuilder(env.sim, a.camera, a.res, a.res, a.bbox, a.m, lut).build(obs)
            pc, lab, _ = RP.CloudBuilder(env.sim, a.camera, a.res, a.res, a.bbox, a.n, lut).build(obs)
            stored, stored_lab = z["data/point_cloud"][start], z["data/point_label"][start]
            ok = (np.array_equal(pc, stored) and np.array_equal(lab, stored_lab)
                  and np.array_equal(ref[:a.n], stored) and np.array_equal(ref_lab[:a.n], stored_lab))
            n_robot = int(np.isin(ref_lab, ROBOT_LABELS).sum())
            out.append((i, ref, ref_lab, n, ok, n_robot))
            print(f"[{os.getpid()}] demo {ep}: in box {n}, robot pts in ref {n_robot}, prefix ok {ok}", flush=True)
    return out


def scene_worker(args):
    a, seeds, env_args = args
    import torch  # noqa: F401
    from r3d.common import robosuite_pcd as RP
    from mimicgen_view_check import make_env
    env = make_env(env_args, a.camera, a.res)
    out = []
    for seed in seeds:
        np.random.seed(seed)
        obs = env.reset()
        lut = RP.geom_label_lut(env.sim, a.env_name)
        ref, ref_lab, n = RP.CloudBuilder(env.sim, a.camera, a.res, a.res, a.bbox, a.m, lut).build(obs)
        pc, lab, _ = RP.CloudBuilder(env.sim, a.camera, a.res, a.res, a.bbox, a.n, lut).build(obs)
        ok = np.array_equal(ref[:a.n], pc) and np.array_equal(ref_lab[:a.n], lab)
        out.append((seed, n, ok, int(np.isin(ref_lab, ROBOT_LABELS).sum())))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--zarr", default=os.path.join(R3D_DIR, "data", "square_d2-agentview_p4096-200.zarr"))
    ap.add_argument("--m", type=int, default=8192, help="reference FPS size")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--eval-scenes", type=int, default=0)
    a = ap.parse_args()

    import zarr
    z = zarr.open(a.zarr, "r")
    attrs = dict(z["meta"].attrs)
    a.task, a.camera, a.res, a.env_name = attrs["task"], attrs["camera"], int(attrs["render_res"]), attrs["env_name"]
    a.bbox, a.n = attrs["pcd_crop_bbox"], int(attrs["pcd_down_sample_num"])
    if a.m <= a.n:
        raise ValueError(f"--m {a.m} must exceed the stored N {a.n}")
    with h5py.File(hdf5_path(a.task), "r") as f:
        env_args = json.loads(f["data"].attrs["env_args"])
    ctx = mp.get_context("spawn")
    t0 = time.time()

    if a.eval_scenes:
        kw_args = copy.deepcopy(env_args)
        kw_args["env_kwargs"]["controller_configs"]["control_delta"] = False   # as the evaluation
        seeds = [100000 + k for k in range(a.eval_scenes)]
        with ctx.Pool(a.workers) as pool:
            res = [r for part in pool.map(scene_worker, [(a, seeds[i::a.workers], kw_args)
                                                          for i in range(a.workers)]) for r in part]
        robot = np.array([r[3] for r in res])
        print(f"eval scenes {len(res)}: prefix ok {sum(r[2] for r in res)}/{len(res)}, robot pts in ref "
              f"min {robot.min()} median {int(np.median(robot))} (need >= {a.n}), in box min {min(r[1] for r in res)}  "
              f"{(time.time() - t0) / 60:.1f} min")
        sys.exit(0 if all(r[2] for r in res) and robot.min() >= a.n else 1)

    demos = list(attrs["demos"])
    starts = np.concatenate([[0], z["meta/episode_ends"][:-1]]).astype(int)
    items = [(i, ep, int(s)) for i, (ep, s) in enumerate(zip(demos, starts))]
    with ctx.Pool(a.workers) as pool:
        res = [r for part in pool.map(demo_worker, [(a, items[i::a.workers], env_args)
                                                     for i in range(a.workers)]) for r in part]
    res.sort(key=lambda r: r[0])
    bad = [demos[r[0]] for r in res if not r[4]]
    robot = np.array([r[5] for r in res])
    print(f"rendered {len(res)} demos in {(time.time() - t0) / 60:.1f} min; prefix mismatches {len(bad)}; "
          f"robot pts in ref min {robot.min()} median {int(np.median(robot))}")
    if bad:
        raise RuntimeError(f"re-rendered frame 0 differs from the stored cloud for demos {bad[:10]}")
    if robot.min() < a.n:
        raise RuntimeError(f"a reference has only {robot.min()} robot points < {a.n}")
    out = a.zarr.rstrip("/").removesuffix(".zarr") + f".frame0_p{a.m}.npz"
    np.savez(out + ".tmp.npz", cloud=np.stack([r[1] for r in res]).astype(np.float32),
             label=np.stack([r[2] for r in res]).astype(np.int16), n_valid=np.array([r[3] for r in res]),
             demos=np.array(demos), m=a.m, n=a.n)
    os.replace(out + ".tmp.npz", out)
    print("wrote", out)


if __name__ == "__main__":
    main()
