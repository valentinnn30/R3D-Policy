"""Prefix thinning for the claim sweep (R2 armsparse / R3 scenesparse; docs/sim_token_allocation_learnings.md).

The stored Square cloud is FPS output in selection order with a deterministic start, so its first `prefix`
points ARE the FPS-`prefix` cloud. With i = a point's index in that stored order and arm = cells
link0..link7 (== label 5, the set the arm cut removes; the gripper is never "arm"):

    point_set 'arm'      (armsparse):   keep i  iff  cell(i) not in arm  or  i < prefix
    point_set 'non_arm'  (scenesparse): keep i  iff  cell(i) in arm      or  i < prefix

so the thinned set is reduced to its points in the FPS-`prefix` cloud, and both rules together give exactly
the first `prefix` points. Removal goes through `cut_points_by_label` (kept points first in stored order,
padded with copies of the first kept point, valid count = `point_cloud_len`), as the arm cut. No random
draws. Shared by `RobotwinSimDataset` (training, flags computed before augmentation) and
`scripts/sim/eval_mimicgen_ckpt.py` (live), so train == eval by construction.
"""

import numpy as np

from r3d.common.label_cut import cut_points_by_label

ARM_CELL_NAMES = tuple(f"link{i}" for i in range(8))
POINT_SETS = ("arm", "non_arm")


def arm_cell_ids():
    from r3d.common.robosuite_pcd import CELL_NAMES
    return [CELL_NAMES.index(c) for c in ARM_CELL_NAMES]


def check_spec(point_set, prefix, num_points):
    if point_set not in POINT_SETS:
        raise ValueError(f"thin_point_set {point_set!r} not in {POINT_SETS}")
    if not 0 < int(prefix) < int(num_points):
        raise ValueError(f"thin_prefix {prefix} must be in (0, num_points={num_points})")


def thin_drop_labels(cells, point_set, prefix, arm_ids=None):
    """cells (..., N) link cell per point in STORED order -> (..., N) int8, 1 = remove."""
    cells = np.asarray(cells)
    arm = np.isin(cells, arm_cell_ids() if arm_ids is None else arm_ids)
    thinned = arm if point_set == "arm" else ~arm
    late = np.arange(cells.shape[-1]) >= int(prefix)
    return (thinned & late).astype(np.int8)


def thin_cloud(cloud, cells, point_set, prefix, arm_ids=None):
    """cloud (N, C), cells (N,) in stored order -> (cloud (N, C) padded, n_valid)."""
    return cut_points_by_label(cloud, thin_drop_labels(cells, point_set, prefix, arm_ids), [1])
