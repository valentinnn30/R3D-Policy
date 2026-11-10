"""Narrowed MimicGen task variants for the sim token-allocation study (r3d_robosuite only).

Threading D1 with the TRIPOD (the target) restricted to the middle half of D1's x and yaw ranges
(docs/sim_token_allocation_learnings.md, "Next task candidate: MimicGen Threading"): the target
still moves, so the insertion cannot be memorized from proprio, but the range is small, so
failures should come from the observation rather than from generalization. The needle keeps D1's
range. ONE window definition serves both sides:
- training: `select_demos` keeps the stock D1 demos whose initial tripod pose lies inside it
  (convert_mimicgen_to_zarr.py --window threading_d1n);
- evaluation: `Threading_D1_Narrow` samples the tripod from exactly that window, and
  `check_in_window` asserts every eval scene against it.
x / y are relative to the table centre (the placement sampler's reference), yaw is the body's
z rotation (the sampler's z_rot), as measured by scripts/sim/threading_task_check.py.
"""

import json

import numpy as np

WINDOWS = {
    "threading_d1n": {
        "env_name": "Threading_D1_Narrow",
        "body": "tripod_obj_root",
        "x": [-0.0375, 0.0875],                         # D1: [-0.1, 0.15]
        "y": [-0.2, -0.1],                              # D1, unchanged
        "yaw": [float(np.pi / 3), float(2 * np.pi / 3)],  # D1: [pi/6, 5pi/6]
    },
}
_EPS = 1e-6


def _yaw(q_wxyz):
    w, x, y, z = q_wxyz
    return float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


def body_pose(sim, body, table_offset):
    """-> (x, y relative to the table centre, yaw) of a body."""
    bid = sim.model.body_name2id(body)
    p = sim.data.body_xpos[bid]
    return float(p[0] - table_offset[0]), float(p[1] - table_offset[1]), _yaw(sim.data.body_xquat[bid])


def in_window(pose, window):
    x, y, yaw = pose
    return (window["x"][0] - _EPS <= x <= window["x"][1] + _EPS and window["y"][0] - _EPS <= y <= window["y"][1] + _EPS
            and window["yaw"][0] - _EPS <= yaw <= window["yaw"][1] + _EPS)


def check_in_window(env, window):
    pose = body_pose(env.sim, window["body"], env.table_offset)
    if not in_window(pose, window):
        raise RuntimeError(f"{window['body']} at {pose} is outside the window {window}")
    return pose


def select_demos(hdf5, env_args, window, n):
    """The first n demo ids (file order) whose initial target pose lies in the window.
    -> (ids, n inside, n total)."""
    import h5py
    import robosuite as suite
    import mimicgen  # noqa: F401
    kw = dict(env_args["env_kwargs"])
    kw.update(has_renderer=False, has_offscreen_renderer=False, use_camera_obs=False, ignore_done=True)
    env = suite.make(env_args["env_name"], **kw)
    env.reset()
    with h5py.File(hdf5, "r") as f:
        ids = sorted(int(k.split("_")[1]) for k in f["data"].keys())
        inside = []
        for i in ids:   # both objects have free joints: the flattened state carries the pose
            env.sim.set_state_from_flattened(f[f"data/demo_{i}/states"][0])
            env.sim.forward()
            if in_window(body_pose(env.sim, window["body"], env.table_offset), window):
                inside.append(i)
    return inside[:n], len(inside), len(ids)


def _define_envs():
    from mimicgen.envs.robosuite.threading import Threading_D1

    class Threading_D1_Narrow(Threading_D1):
        """Threading D1, tripod restricted to WINDOWS['threading_d1n'] (registered with robosuite)."""

        def _get_initial_placement_bounds(self):
            b = super()._get_initial_placement_bounds()
            w = WINDOWS["threading_d1n"]
            b["tripod"].update(x=tuple(w["x"]), y=tuple(w["y"]), z_rot=tuple(w["yaw"]))
            return b

    return {"Threading_D1_Narrow": Threading_D1_Narrow}


ENVS = _define_envs()
