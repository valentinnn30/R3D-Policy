"""State / action vectors for robosuite (MimicGen) policies.  Sim token-allocation study.

Shared by the zarr conversion and the evaluation so both use one convention.

    state  (11) = eef pos (3) | eef rotation as rot6d (6) | gripper finger qpos (2)
    action (10) = ABSOLUTE OSC target: goal pos (3) | goal rotation as rot6d (6) | gripper (1)

The absolute target follows EquiDiff (`RobomimicAbsoluteActionConverter`): the
recorded DELTA action is fed to the OSC controller from the recorded state and the
controller's goal pose is read back; evaluation runs the same controller with
control_delta=False. rot6d (pytorch3d convention: the first two ROWS of the matrix)
instead of a quaternion -- no q / -q ambiguity in the regression target (the issue
the real-robot pipeline handles with `canonicalize_quaternions`). The state uses
rot6d for the same reason. robosuite quaternions are xyzw.
"""

import numpy as np
from scipy.spatial.transform import Rotation

STATE_DIM, ACTION_DIM = 11, 10


def mat_to_rot6d(R):
    """(..., 3, 3) -> (..., 6): first two rows, as pytorch3d.matrix_to_rotation_6d."""
    R = np.asarray(R)
    return R[..., :2, :].reshape(*R.shape[:-2], 6)


def rot6d_to_mat(d6):
    """(..., 6) -> (..., 3, 3) by Gram-Schmidt, as pytorch3d.rotation_6d_to_matrix."""
    d6 = np.asarray(d6, dtype=np.float64)
    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = a1 / np.linalg.norm(a1, axis=-1, keepdims=True)
    b2 = a2 - (b1 * a2).sum(-1, keepdims=True) * b1
    b2 = b2 / np.linalg.norm(b2, axis=-1, keepdims=True)
    b3 = np.cross(b1, b2)
    return np.stack([b1, b2, b3], axis=-2)


def state_vector(eef_pos, eef_quat_xyzw, gripper_qpos):
    """(..., 3), (..., 4), (..., 2) -> (..., 11) float32."""
    R = Rotation.from_quat(np.asarray(eef_quat_xyzw).reshape(-1, 4)).as_matrix()
    r6 = mat_to_rot6d(R).reshape(*np.shape(eef_quat_xyzw)[:-1], 6)
    return np.concatenate([eef_pos, r6, gripper_qpos], -1).astype(np.float32)


def abs_action_vector(goal_pos, goal_ori_mat, gripper):
    """controller goal -> (10,) float32."""
    return np.concatenate([goal_pos, mat_to_rot6d(goal_ori_mat), np.atleast_1d(gripper)]).astype(np.float32)


def env_action(a10):
    """(10,) policy action -> (7,) robosuite OSC_POSE action with control_delta=False:
    absolute pos | absolute rotation as axis-angle | gripper."""
    a10 = np.asarray(a10, dtype=np.float64)
    rotvec = Rotation.from_matrix(rot6d_to_mat(a10[3:9])).as_rotvec()
    return np.concatenate([a10[:3], rotvec, a10[9:10]])
