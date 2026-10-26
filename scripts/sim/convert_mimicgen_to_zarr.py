#!/usr/bin/env python
"""MimicGen hdf5 (robomimic format) -> R3D zarr ReplayBuffer.  Run in r3d_robosuite.

    python scripts/sim/convert_mimicgen_to_zarr.py --task square_d2 --demos 200 --workers 6
    python scripts/sim/convert_mimicgen_to_zarr.py --task square_d2 --verify 10   # check only

Every frame is re-rendered from the stored simulator state (no planning) with
`r3d.common.robosuite_pcd` -- the SAME cloud builder the evaluation uses:

    point_cloud[t]  (num_points, 6)  agentview, world frame, box crop, FPS (deterministic
                                     start: the first N points ARE the FPS-N cloud)
    point_label[t]  (num_points,)    point_labels.py classes (arm 5, gripper 7, nut 8, peg 9)
    state[t]        (11,)            eef pos | rot6d | gripper qpos      (robosuite_actions)
    action[t]       (10,)            ABSOLUTE OSC goal of the recorded delta action t:
                                     pos | rot6d | gripper (EquiDiff's conversion)
    meta/init_layout (n_ep, 14)      nut pos+quat(xyzw), peg pos+quat at t=0 (distance analysis)

`--verify K` replays the absolute actions of K demos under control_delta=False from
the recorded initial state and reports task success and end-effector tracking error
against the recording -- the check that the action conversion is right.
"""

import argparse
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

ROLE_BODIES = {"square": ("SquareNut_main", "peg1")}


def hdf5_path(task):
    return os.path.join(R3D_DIR, "data", "mimicgen", "core", f"{task}.hdf5")


def _layout(env, task):
    from scipy.spatial.transform import Rotation  # noqa: F401
    out = []
    for b in ROLE_BODIES[task.rsplit("_d", 1)[0]]:
        bid = env.sim.model.body_name2id(b)
        q_wxyz = env.sim.data.body_xquat[bid]
        out += list(env.sim.data.body_xpos[bid]) + [q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]]
    return np.array(out, dtype=np.float64)


def convert_demo(env, f, ep, a, cb_cache):
    from r3d.common import robosuite_actions as RA
    from r3d.common import robosuite_pcd as RP
    from mimicgen_view_check import reset_to_demo
    g = f[f"data/demo_{ep}"]
    S, A = g["states"][:], g["actions"][:]
    reset_to_demo(env, g.attrs["model_file"], S[0])
    if "bbox" not in cb_cache:
        cb_cache["bbox"] = RP.default_bbox(env.sim)
    lut = RP.geom_label_lut(env.sim, cb_cache["env_name"])
    cb = RP.CloudBuilder(env.sim, a.camera, a.res, a.res, cb_cache["bbox"], a.num_points, lut)
    layout = _layout(env, a.task)
    robot = env.robots[0]
    T = len(S)
    pcs = np.zeros((T, a.num_points, 6), np.float32)
    labs = np.zeros((T, a.num_points), np.int16)
    acts = np.zeros((T, RA.ACTION_DIM), np.float32)
    nval = np.zeros(T, np.int32)
    for t in range(T):
        env.sim.set_state_from_flattened(S[t])
        env.sim.forward()
        obs = env._get_observations(force_update=True)
        pcs[t], labs[t], nval[t] = cb.build(obs)
        robot.control(A[t], policy_step=True)            # OSC goal of the delta action (EquiDiff)
        c = robot.controller
        acts[t] = RA.abs_action_vector(c.goal_pos, c.goal_ori, A[t, -1])
    states = RA.state_vector(g["obs/robot0_eef_pos"][:], g["obs/robot0_eef_quat"][:], g["obs/robot0_gripper_qpos"][:])
    return dict(point_cloud=pcs, point_label=labs, state=states, action=acts, n_valid=nval, layout=layout)


def worker(args):
    a, eps, out_dir, env_args = args
    import torch  # noqa: F401  (cuda init per process)
    from mimicgen_view_check import make_env
    env = make_env(env_args, a.camera, a.res)
    f = h5py.File(hdf5_path(a.task), "r")
    cache = {"env_name": env_args["env_name"]}
    for ep in eps:
        p = os.path.join(out_dir, f"ep{ep:05d}.npz")
        if os.path.exists(p):
            continue
        d = convert_demo(env, f, ep, a, cache)
        np.savez(p + ".tmp.npz", bbox=np.array(cache["bbox"]), **d)
        os.replace(p + ".tmp.npz", p)
        print(f"[worker {os.getpid()}] demo {ep}: {len(d['state'])} frames", flush=True)
    return True


def verify(a, env_args):
    """replay absolute actions under control_delta=False; success + eef tracking error."""
    import copy
    from r3d.common import robosuite_actions as RA
    from mimicgen_view_check import reset_to_demo
    import robosuite as suite
    import mimicgen  # noqa: F401
    f = h5py.File(hdf5_path(a.task), "r")
    kw = copy.deepcopy(env_args["env_kwargs"])
    kw["controller_configs"]["control_delta"] = False
    kw.update(has_offscreen_renderer=False, use_camera_obs=False)
    env = suite.make(env_args["env_name"], **kw)
    cache = {"env_name": env_args["env_name"]}
    rows = []
    for ep in range(a.verify):
        g = f[f"data/demo_{ep}"]
        S = g["states"][:]
        # actions computed with the DELTA env (as in conversion), replayed in the ABS env
        denv = getattr(verify, "_denv", None)
        if denv is None:
            from mimicgen_view_check import make_env
            denv = verify._denv = make_env(env_args, a.camera, 64)
        d = convert_demo(denv, f, ep, argparse.Namespace(**{**vars(a), "res": 64, "num_points": 64}), cache)
        reset_to_demo(env, g.attrs["model_file"], S[0])
        eef_rec = g["obs/robot0_eef_pos"][:]
        err, success = [], False
        for t in range(len(S)):
            obs, _, _, _ = env.step(RA.env_action(d["action"][t]))
            if t + 1 < len(S):
                err.append(np.linalg.norm(obs["robot0_eef_pos"] - eef_rec[t + 1]))
            success = success or env._check_success()
        rows.append((ep, success, float(np.mean(err)), float(np.max(err))))
        print(f"demo {ep}: replay success {success}  eef err mean {1000 * np.mean(err):.1f} mm  max {1000 * np.max(err):.1f} mm", flush=True)
    print(f"replayed success {sum(r[1] for r in rows)}/{len(rows)}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", default="square_d2")
    ap.add_argument("--demos", type=int, default=200, help="first N demos (demo_0 .. demo_N-1)")
    ap.add_argument("--camera", default="agentview")
    ap.add_argument("--res", type=int, default=512)
    ap.add_argument("--num-points", type=int, default=4096, help="stored; the first N give FPS-N")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--output", default=None)
    ap.add_argument("--verify", type=int, default=0, help="only check the action conversion on K demos")
    a = ap.parse_args()

    with h5py.File(hdf5_path(a.task), "r") as f:
        env_args = json.loads(f["data"].attrs["env_args"])
    if a.verify:
        return verify(a, env_args)

    out = a.output or os.path.join(R3D_DIR, "data", f"{a.task}-{a.camera}_p{a.num_points}-{a.demos}.zarr")
    tmp = out + ".parts"
    os.makedirs(tmp, exist_ok=True)
    eps = list(range(a.demos))
    chunks = [eps[i::a.workers] for i in range(a.workers)]
    t0 = time.time()
    ctx = mp.get_context("spawn")
    with ctx.Pool(a.workers) as pool:
        pool.map(worker, [(a, c, tmp, env_args) for c in chunks])
    print(f"rendered {a.demos} demos in {(time.time() - t0) / 60:.1f} min", flush=True)

    import zarr
    from r3d.common import robosuite_pcd as RP
    from r3d.common import robosuite_actions as RA
    parts = [np.load(os.path.join(tmp, f"ep{ep:05d}.npz")) for ep in eps]
    bbox = parts[0]["bbox"].tolist()
    assert all(np.allclose(p["bbox"], bbox) for p in parts), "crop box differs between episodes"
    root = zarr.open(out, "w")
    N = a.num_points
    for k, chunks_ in (("point_cloud", (1, N, 6)), ("point_label", (1, N)), ("state", None), ("action", None)):
        arr = np.concatenate([p[k] for p in parts])
        root.create_dataset(f"data/{k}", data=arr, chunks=chunks_ or (min(len(arr), 10000), arr.shape[1]))
    root.create_dataset("meta/episode_ends", data=np.cumsum([len(p["state"]) for p in parts]).astype(np.int64))
    root.create_dataset("meta/init_layout", data=np.stack([p["layout"] for p in parts]))
    nval = np.concatenate([p["n_valid"] for p in parts])
    root["meta"].attrs.update({
        "source": "mimicgen", "task": a.task, "env_name": env_args["env_name"], "demos": eps,
        "camera": a.camera, "render_res": a.res, "pcd_crop_bbox": bbox, "pcd_down_sample_num": N,
        "point_label_classes": {str(k): v for k, v in RP.CLASS_NAMES.items()},
        "state_layout": "eef pos (3) | eef rot6d (6, first two rows) | gripper finger qpos (2)",
        "action_layout": "ABSOLUTE OSC goal: pos (3) | rot6d (6) | gripper (1); eval: control_delta=False",
        "init_layout": "nut pos(3) quat_xyzw(4) | peg pos(3) quat_xyzw(4)",
        "controller_configs": env_args["env_kwargs"]["controller_configs"],
        "points_in_box_min": int(nval.min()), "points_in_box_p10": float(np.percentile(nval, 10)),
        "state_dim": RA.STATE_DIM, "action_dim": RA.ACTION_DIM,
    })
    print(f"wrote {out}: {int(root['meta/episode_ends'][-1])} frames; points in box min {nval.min()}, "
          f"frames padded at {N}: {(nval < N).sum()}")


if __name__ == "__main__":
    main()
