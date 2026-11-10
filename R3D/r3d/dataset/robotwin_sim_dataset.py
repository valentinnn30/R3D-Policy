"""RoboTwin2 sim baseline dataset (see CLAUDE.md "Sim baseline").

`RobotwinDataset` with ONE change: the point_cloud normalizer comes from the crop
box, not from the data. The token-allocation study will crop points after
preprocessing; a data-fitted normalizer would move with every crop, so a crop
would change the encoder's input geometry twice (fewer points AND a different
scale) and the comparison would be confounded. The box is read from the zarr's
`meta.attrs['pcd_crop_bbox']`, written by `convert_robotwin_hdf5_to_zarr.py` from
the same task config the runner evaluates with.

Same reasoning as `RealMultiCamDataset.get_normalizer`.

`drop_point_labels` (e.g. [5] = right arm) applies the oracle label cut
(`r3d/common/label_cut.py`) to every frame AFTER augmentation, so the kept
points are augmented exactly as in the uncut baseline (colour jitter's mean is
taken over the same full cloud). It adds `point_cloud_len` to the obs. Null =
no cut, and `point_label` is then not even loaded.

`drop_point_links` (e.g. [link0, ..., link6]) is the same oracle cut, by the per-point LINK CELL
(`data/point_link`, robosuite_pcd.CELL_NAMES) instead of the class: identical code path
(`cut_points_by_label` after augmentation, `point_cloud_len`), so link0..link7 reproduces
`drop_point_labels=[5]` bit for bit. `point_link` is consumed here and never reaches the policy.
Exclusive with `drop_point_labels`, `load_point_link` and `freeze_robot_labels`.

`dropout_point_labels` (e.g. [4, 5, 6, 7], with `dropout_keep` 0.5): robot-part DROPOUT in training only -- the
class-level counterpart of the remover's dropout-only run (DROP, Square): per window each listed class is removed
from all its frames with probability 1 - keep, independently, by the same `cut_points_by_label` after augmentation
(drawn from numpy's stream AFTER the augmentation, so the augmentation draws equal the uncut run's). Every training
sample then carries `point_cloud_len` (N when nothing is removed). The validation set and evaluation see the FULL
cloud. Exclusive with the other label options.

`load_point_link` adds `point_link` (T, N) int8 to the obs -- plus `point_label` (T, N) int8, read
only by the remover's analytics (nut vs peg coverage) -- `point_link` being the per-point link cell
(robosuite_pcd.CELL_NAMES, -1 = not robot) written by scripts/sim/add_link_labels_mimicgen.py,
windowed and prefix-sliced together with the clouds. The learned link remover
(DP3LinkRemover) consumes it before the normalizer. Off = not even loaded.

`num_points` keeps the FIRST num_points points of every stored cloud. The
converter stores RoboTwin's FPS output in selection order and FPS is greedy, so
the first N points of an FPS-4096 cloud ARE the FPS-N cloud (same start point)
-- one 4096-point collection serves 4096 / 2048 / 1024. At evaluation RoboTwin
produces the N-point cloud directly (task config with pcd_down_sample_num = N),
which is the same set.

`freeze_robot_labels` (e.g. [5, 7], round 7 shortcut test): every frame's robot points are
replaced by those of the episode's FIRST frame (`r3d/common/robot_freeze.py`), BEFORE
augmentation, so noise and colour jitter act on it as on the uncut cloud. The cloud stays
`num_points` wide with no padding; the reference is the frame-0 FPS-M sidecar
(`freeze_reference`, scripts/sim/make_frame0_reference.py). The window's episode rides along
as `episode_idx`, sliced by the sampler with the clouds. Null = off, nothing extra loaded.
"""

import numpy as np
import torch
import zarr

from r3d.common.label_cut import cut_points_by_label
from r3d.common.replay_buffer import ReplayBuffer
from r3d.common.robot_freeze import freeze_robot_points, reference_robot_index
from r3d.common.sampler import SequenceSampler
from r3d.dataset.robotwin_dataset import RobotwinDataset
from r3d.model.common.normalizer import LinearNormalizer, SingleFieldLinearNormalizer


class RobotwinSimDataset(RobotwinDataset):

    def __init__(self, zarr_path, drop_point_labels=None, num_points=None, load_point_link=False,
                 freeze_robot_labels=None, freeze_reference=None, drop_point_links=None,
                 dropout_point_labels=None, dropout_keep=0.5, **kwargs):
        super().__init__(zarr_path, **kwargs)
        attrs = zarr.open(zarr_path, mode="r")["meta"].attrs
        if "pcd_crop_bbox" not in attrs:
            raise KeyError(f"{zarr_path}: meta.attrs has no 'pcd_crop_bbox' -- convert with "
                           "scripts/convert_robotwin_hdf5_to_zarr.py")
        self.pcd_crop_bbox = np.asarray(attrs["pcd_crop_bbox"], dtype=np.float64)

        self.drop_point_labels = None if drop_point_labels is None else [int(c) for c in drop_point_labels]
        self.load_point_link = bool(load_point_link)
        if self.load_point_link:
            if self.drop_point_labels:
                raise ValueError("load_point_link (learned link remover) with drop_point_labels (oracle cut)")
            if self.use_target_ee:
                raise NotImplementedError("load_point_link with use_target_ee")
            if "point_link" not in zarr.open(zarr_path, mode="r")["data"]:
                raise KeyError(f"{zarr_path} has no data/point_link -- run scripts/sim/add_link_labels_mimicgen.py")
            from r3d.common.robosuite_pcd import CELL_NAMES
            if list(attrs.get("point_link_names", [])) != list(CELL_NAMES):
                raise ValueError(f"{zarr_path}: point_link_names differ from robosuite_pcd.CELL_NAMES")
        self.freeze_robot_labels = None if freeze_robot_labels is None else [int(c) for c in freeze_robot_labels]
        self.dropout_point_labels = None if not dropout_point_labels else [int(c) for c in dropout_point_labels]
        self.dropout_keep = float(dropout_keep)
        self.dropout_active = bool(self.dropout_point_labels)          # off in the validation copy
        if self.dropout_point_labels:
            if drop_point_labels or drop_point_links or load_point_link or freeze_robot_labels or self.use_target_ee:
                raise ValueError("dropout_point_labels with another label option / use_target_ee")
            if not 0.0 <= self.dropout_keep <= 1.0:
                raise ValueError(f"dropout_keep {self.dropout_keep} not in [0, 1]")
        self.drop_point_links = None
        if drop_point_links:
            from r3d.common.robosuite_pcd import CELL_NAMES
            if self.drop_point_labels or self.load_point_link or self.freeze_robot_labels:
                raise ValueError("drop_point_links with drop_point_labels / load_point_link / freeze_robot_labels")
            if self.use_target_ee:
                raise NotImplementedError("drop_point_links with use_target_ee")
            if "point_link" not in zarr.open(zarr_path, mode="r")["data"]:
                raise KeyError(f"{zarr_path} has no data/point_link -- run scripts/sim/add_link_labels_mimicgen.py")
            if list(attrs.get("point_link_names", [])) != list(CELL_NAMES):
                raise ValueError(f"{zarr_path}: point_link_names differ from robosuite_pcd.CELL_NAMES")
            self.drop_point_links = [CELL_NAMES.index(str(c)) for c in drop_point_links]  # unknown -> ValueError
        if self.freeze_robot_labels and (self.drop_point_labels or self.load_point_link or self.use_target_ee):
            raise ValueError("freeze_robot_labels with drop_point_labels / load_point_link / use_target_ee")
        # the remover also gets point_label: its analytics tell nut from peg (never a model input)
        extra = (["point_label"] if self.drop_point_labels or self.load_point_link or self.freeze_robot_labels
                 or self.dropout_point_labels
                 else []) \
            + (["point_link"] if self.load_point_link or self.drop_point_links else [])
        if extra:
            if self.drop_point_labels and self.use_target_ee:
                raise NotImplementedError("drop_point_labels with use_target_ee")
            # reload WITH the per-point labels and rebuild the sampler on the parent's own
            # episode masks, so the labels are windowed together with the clouds
            self.replay_buffer = ReplayBuffer.copy_from_path(
                zarr_path, keys=["state", "action", "point_cloud"] + extra)
            if self.freeze_robot_labels:
                ends = self.replay_buffer.episode_ends[:]
                self.replay_buffer.data["episode_idx"] = np.repeat(
                    np.arange(len(ends), dtype=np.int64), np.diff(np.concatenate([[0], ends])))
            self.sampler = SequenceSampler(
                replay_buffer=self.replay_buffer, sequence_length=self.horizon,
                pad_before=self.pad_before, pad_after=self.pad_after, episode_mask=self.train_mask)

        stored = self.replay_buffer["point_cloud"].shape[1]
        if num_points is not None and int(num_points) != stored:
            n = int(num_points)
            if n > stored:
                raise ValueError(f"num_points={n} but {zarr_path} stores {stored} points per cloud")
            # numpy-backed buffer: replace in place; the sampler reads it per sample
            data = self.replay_buffer.data
            data["point_cloud"] = np.ascontiguousarray(data["point_cloud"][:, :n])
            for key in ("point_label", "point_link"):
                if key in data:
                    data[key] = np.ascontiguousarray(data[key][:, :n])

        if self.freeze_robot_labels:
            self._load_freeze_reference(zarr_path, freeze_reference)

    def _load_freeze_reference(self, zarr_path, path):
        if path is None:
            raise ValueError("freeze_robot_labels needs freeze_reference (the frame-0 sidecar)")
        ref = np.load(path)
        attrs = zarr.open(zarr_path, mode="r")["meta"].attrs
        n_ep = len(self.replay_buffer.episode_ends)
        n = self.replay_buffer["point_cloud"].shape[1]
        if list(ref["demos"]) != list(attrs["demos"]) or ref["cloud"].shape[0] != n_ep:
            raise ValueError(f"{path}: episodes differ from {zarr_path}")
        if int(ref["n"]) != int(attrs["pcd_down_sample_num"]):
            raise ValueError(f"{path}: built for N={int(ref['n'])}, zarr stores {attrs['pcd_down_sample_num']}")
        starts = np.concatenate([[0], self.replay_buffer.episode_ends[:-1]]).astype(int)
        # the reference's first N points ARE the stored frame 0 (FPS prefix); checked per episode
        stored0 = self.replay_buffer["point_cloud"][starts]
        if not np.array_equal(ref["cloud"][:, :n], stored0[:, :n]):
            raise ValueError(f"{path}: reference prefix differs from the stored frame-0 clouds")
        self._freeze_ref = ref["cloud"]
        self._freeze_ref_idx = [reference_robot_index(lab, self.freeze_robot_labels, n) for lab in ref["label"]]

    def _sample_to_data(self, sample):
        data = super()._sample_to_data(sample)
        if self.drop_point_labels or self.dropout_point_labels:
            data["_point_label"] = sample["point_label"]
        if self.drop_point_links:
            data["_point_label"] = sample["point_link"]   # the same cut below, keyed on the link cell
        if self.freeze_robot_labels:
            ep = int(sample["episode_idx"][0])
            if not (sample["episode_idx"] == ep).all():
                raise RuntimeError("a window spans two episodes")
            ref, ref_idx = self._freeze_ref[ep], self._freeze_ref_idx[ep]
            data["obs"]["point_cloud"] = np.stack([
                freeze_robot_points(pc, lab, ref, ref_idx, self.freeze_robot_labels)
                for pc, lab in zip(data["obs"]["point_cloud"], sample["point_label"])])
        if self.load_point_link:
            data["obs"]["point_link"] = sample["point_link"].astype(np.int8)   # .long() in the policy
            data["obs"]["point_label"] = sample["point_label"].astype(np.int8)  # analytics only
        return data

    def __getitem__(self, idx):
        data = super().__getitem__(idx)  # sample + augmentation, as the baseline
        drop = self.drop_point_labels or self.drop_point_links
        if self.dropout_point_labels:
            if not self.dropout_active:          # validation: the full cloud, as at evaluation
                data.pop("_point_label")
                return data
            drop = [c for c in self.dropout_point_labels if np.random.random() >= self.dropout_keep]
        elif not drop:
            return data
        labels = data.pop("_point_label").numpy()
        pc = data["obs"]["point_cloud"].numpy()
        cut, lengths = zip(*(cut_points_by_label(pc[t], labels[t], drop)
                             for t in range(len(pc))))
        data["obs"]["point_cloud"] = torch.from_numpy(np.stack(cut))
        data["obs"]["point_cloud_len"] = torch.as_tensor(lengths, dtype=torch.int64)
        return data

    def get_validation_dataset(self):
        val = super().get_validation_dataset()
        val.dropout_active = False
        return val

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
