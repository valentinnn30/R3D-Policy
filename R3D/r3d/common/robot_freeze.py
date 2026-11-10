"""Frozen robot points (round 7, shortcut test; docs/sim_token_allocation_learnings.md).

Every frame's robot points are replaced by the robot points of the episode's FIRST frame;
everything else stays live. Proprioception (agent_pos) is untouched, so the robot points carry
no current information while the token allocation stays that of the uncut cloud.

Point count: frame t keeps its own scene points at their positions (bit for bit), and its R_t
robot positions get the first R_t robot points -- in FPS order -- of a DENSER reference cloud
of the first frame (FPS-M, M > N, same CloudBuilder). FPS is greedy with a deterministic start,
so the reference's first N points ARE the frame-0 N-point cloud: the robot keeps the uncut
point count in every frame, the cloud stays exactly N wide (no padding), and frame 0 maps to
itself bit for bit.

Shared by `RobotwinSimDataset` (training; reference from the sidecar written by
scripts/sim/make_frame0_reference.py) and scripts/sim/eval_mimicgen_ckpt.py (reference built
at env.reset()), so the two cannot drift.
"""

import numpy as np


def reference_robot_index(ref_labels, robot_labels, n_points):
    """Indices of the reference's robot points in FPS order. Raises if fewer than n_points:
    then a frame with (almost) only robot points could not be filled."""
    idx = np.flatnonzero(np.isin(np.asarray(ref_labels), list(robot_labels)))
    if len(idx) < n_points:
        raise RuntimeError(f"frame-0 reference has {len(idx)} robot points < {n_points}: "
                           "build it at a larger FPS size")
    return idx


def freeze_robot_points(cloud, labels, ref_cloud, ref_robot_idx, robot_labels):
    """cloud (N, C), labels (N,) of frame t; ref_cloud (M, C) the episode's first frame at FPS-M,
    ref_robot_idx from `reference_robot_index`. -> cloud (N, C), robot positions replaced."""
    robot = np.isin(np.asarray(labels), list(robot_labels))
    out = np.array(cloud, copy=True)
    out[robot] = ref_cloud[ref_robot_idx[:int(robot.sum())]]
    return out
