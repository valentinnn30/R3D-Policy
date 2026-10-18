"""Oracle label cut for the sim token-allocation runs (CLAUDE.md "Sim baseline").

Deletes the points whose semantic class (envs/utils/point_labels.py) is in
`drop`, AFTER preprocessing -- the stored 1024-point cloud is cut, nothing is
re-sampled -- and pads back to the fixed width with copies of the first kept
point, exactly like the real-data robot cut (`RealMultiCamDataset._cut_and_pad`).
The valid count travels as `point_cloud_len`; DP3 routes it around the
normalizer and the encoder masks the padding out of FPS and kNN, so the token
budget is spent on the kept points only.

Shared by `RobotwinSimDataset` (training) and `RoboTwin2Runner` (evaluation),
so the two cannot drift. In evaluation the labels come from the simulator's
segmentation: an oracle, i.e. the upper bound for a learned or geometric cut.
"""

import numpy as np


def cut_points_by_label(cloud, labels, drop):
    """cloud (N, C), labels (N,) -> (cloud (N, C) padded, n_valid)."""
    cloud = np.asarray(cloud)
    keep = ~np.isin(np.asarray(labels), list(drop))
    pts = cloud[keep]
    n = len(pts)
    if n == 0:
        raise RuntimeError(f"label cut {sorted(drop)} removed every point of a frame")
    out = np.empty_like(cloud)
    out[:n] = pts
    out[n:] = pts[0]
    return out, n
