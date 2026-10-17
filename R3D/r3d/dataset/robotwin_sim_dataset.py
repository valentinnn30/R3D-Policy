"""RoboTwin2 sim baseline dataset (see CLAUDE.md "Sim baseline").

`RobotwinDataset` with ONE change: the point_cloud normalizer comes from the crop
box, not from the data. The token-allocation study will crop points after
preprocessing; a data-fitted normalizer would move with every crop, so a crop
would change the encoder's input geometry twice (fewer points AND a different
scale) and the comparison would be confounded. The box is read from the zarr's
`meta.attrs['pcd_crop_bbox']`, written by `convert_robotwin_hdf5_to_zarr.py` from
the same task config the runner evaluates with.

Same reasoning as `RealMultiCamDataset.get_normalizer`. `point_label` is stored
in the zarr for analysis and is not read here.
"""

import numpy as np
import torch
import zarr

from r3d.dataset.robotwin_dataset import RobotwinDataset
from r3d.model.common.normalizer import LinearNormalizer, SingleFieldLinearNormalizer


class RobotwinSimDataset(RobotwinDataset):

    def __init__(self, zarr_path, **kwargs):
        super().__init__(zarr_path, **kwargs)
        attrs = zarr.open(zarr_path, mode="r")["meta"].attrs
        if "pcd_crop_bbox" not in attrs:
            raise KeyError(f"{zarr_path}: meta.attrs has no 'pcd_crop_bbox' -- convert with "
                           "scripts/convert_robotwin_hdf5_to_zarr.py")
        self.pcd_crop_bbox = np.asarray(attrs["pcd_crop_bbox"], dtype=np.float64)

    def get_normalizer(self, mode="limits", **kwargs):
        if self.use_target_ee:
            raise NotImplementedError("use_target_ee is not wired for the sim baseline")
        normalizer = LinearNormalizer()
        normalizer.fit(
            data={"action": self.replay_buffer["action"], "agent_pos": self.replay_buffer["state"]},
            last_n_dims=1, mode=mode, **kwargs)

        n_ch = self.replay_buffer["point_cloud"].shape[-1]
        if n_ch != 6:
            raise ValueError(f"point_cloud has {n_ch} channels, expected xyz+rgb = 6")
        lo = np.concatenate([self.pcd_crop_bbox[0], np.zeros(3)]).astype(np.float32)
        hi = np.concatenate([self.pcd_crop_bbox[1], np.ones(3)]).astype(np.float32)
        scale = (2.0 / np.maximum(hi - lo, 1e-8)).astype(np.float32)
        offset = (-1.0 - scale * lo).astype(np.float32)
        normalizer["point_cloud"] = SingleFieldLinearNormalizer.create_manual(
            scale=torch.from_numpy(scale),
            offset=torch.from_numpy(offset),
            input_stats_dict={
                "min": torch.from_numpy(lo),
                "max": torch.from_numpy(hi),
                "mean": torch.from_numpy(((lo + hi) / 2).astype(np.float32)),
                "std": torch.from_numpy(((hi - lo) / 4).astype(np.float32)),
            })
        return normalizer
