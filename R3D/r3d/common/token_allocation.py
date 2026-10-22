"""Where do the encoder's tokens land? -- shared by the sim view check and analysis.

Reproduces the Uni3D encoder's grouping (`KNNGrouper`, pointnet_extractor.py):
FPS picks `num_group` centres on the NORMALIZED xyz (the encoder never sees
metres), each centre takes its `group_size` nearest points, and that patch is
one token. Given a per-point class label this reports, per class:

- `centres`  : how many token centres sit on a point of that class
- `slots`    : share of the num_group * group_size patch slots it fills
- `coverage` : fraction of its points that are in at least one patch
               (a point in no patch is never seen by the model)

Training draws the FPS start at random (`fps_random_config.random_start`), so
`random_start=True` matches the training distribution; pytorch3d's
`random_start_point` is the same distribution as the encoder's swap-to-front.
"""

import numpy as np
import torch
from pytorch3d.ops import knn_points, sample_farthest_points


def box_normalize(xyz, bbox):
    """Map xyz into [-1, 1] per axis using the crop box -- the same affine map as
    `RobotwinSimDataset.get_normalizer`. bbox = [[x0, y0, z0], [x1, y1, z1]]."""
    lo = np.asarray(bbox[0], dtype=np.float64)
    hi = np.asarray(bbox[1], dtype=np.float64)
    return (2.0 * (np.asarray(xyz, dtype=np.float64) - lo) / (hi - lo) - 1.0).astype(np.float32)


@torch.no_grad()
def token_allocation(xyz_norm, labels, n_classes, num_group=128, group_size=32,
                     random_start=True, device="cuda", return_idx=False):
    """xyz_norm: (B, N, 3) normalized clouds; labels: (B, N) int class per point.

    Returns dict of (B, n_classes) arrays: centres, slots, coverage, points.
    return_idx=True adds `centre_idx` (B, G) and `knn_idx` (B, G, K): which
    points are the token centres and which points each token's patch holds.
    """
    xyz = torch.as_tensor(np.asarray(xyz_norm), dtype=torch.float32, device=device)
    lab = torch.as_tensor(np.asarray(labels), dtype=torch.long, device=device)
    B, N, _ = xyz.shape

    _, c_idx = sample_farthest_points(xyz, K=num_group, random_start_point=random_start)
    centres_xyz = torch.gather(xyz, 1, c_idx.unsqueeze(-1).expand(-1, -1, 3))
    _, knn_idx, _ = knn_points(centres_xyz, xyz, K=group_size)          # (B, G, K)

    onehot = torch.nn.functional.one_hot(lab, n_classes).float()        # (B, N, C)
    points = onehot.sum(1)
    centres = torch.gather(onehot, 1, c_idx.unsqueeze(-1).expand(-1, -1, n_classes)).sum(1)
    flat = knn_idx.reshape(B, -1)
    slots = torch.gather(onehot, 1, flat.unsqueeze(-1).expand(-1, -1, n_classes)).sum(1)
    slots = slots / flat.shape[1]

    seen = torch.zeros(B, N, device=device)
    seen.scatter_(1, flat, 1.0)
    covered = (onehot * seen.unsqueeze(-1)).sum(1)
    coverage = torch.where(points > 0, covered / points.clamp(min=1), torch.full_like(points, float("nan")))

    out = dict(centres=centres, slots=slots, coverage=coverage, points=points)
    if return_idx:
        out.update(centre_idx=c_idx, knn_idx=knn_idx)
    return {k: v.cpu().numpy() for k, v in out.items()}
