#!/usr/bin/env python
"""Checks for the frozen-robot-points option (round 7).  Run in r3d_sim from anywhere:

    python scripts/sim/check_frozen_robot.py            # all checks (needs the frame-0 sidecar)
    python scripts/sim/check_frozen_robot.py --off-only # only the default-off bit-identity checks

1. robot_freeze unit checks (synthetic): scene positions untouched, robot positions = the
   reference's first R_t robot points in order, width kept, identity on the reference's own frame,
   too-small reference raises.
2. DEFAULT OFF: the dataset at git HEAD and the working tree, same config (uncut, and the arm cut
   [5]), same numpy seed -> every sample bit-identical (train + val), normalizer identical.
   An old checkpoint's cfg has no freeze key (evaluation path unchanged).
3. FREEZE ON (sidecar needed): over train + val windows incl. episode starts / ends:
   - shapes unchanged (T, N, 6), no point_cloud_len;
   - scene points == the stored frame bit for bit; robot points == the episode's frame-0
     reference, first R_t robot points in FPS order (so the robot count is unchanged);
   - later frames' robot points actually differ from the live ones (the swap happened);
   - frames that ARE an episode's frame 0 come out bit-identical to the stored cloud;
   - with augmentation on, same numpy seed as the uncut dataset: scene xyz, agent_pos, action
     bit-identical (the augmentation draws are those of the uncut run).
"""

import argparse
import copy
import importlib.util
import os
import pathlib
import subprocess
import sys
import tempfile

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "R3D"))
FREEZE = [5, 7]
FAILS = []


def check(name, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name}{'  -- ' + detail if detail else ''}", flush=True)
    if not ok:
        FAILS.append(name)


def compose(overrides):
    from hydra import compose as hcompose, initialize_config_dir
    from omegaconf import OmegaConf
    OmegaConf.register_new_resolver("eval", eval, replace=True)
    with initialize_config_dir(config_dir=str(ROOT / "R3D" / "r3d" / "config"), version_base=None):
        return hcompose("r3d_robotwin2", overrides=["task=mimicgen_sim", "task_name=square_d2",
                                                    "setting=agentview_p4096-200", "task.num_points=4096",
                                                    "task.num_group=64"] + overrides)


def unit_checks():
    from r3d.common.robot_freeze import freeze_robot_points, reference_robot_index
    rng = np.random.default_rng(0)
    N, M = 50, 120
    cloud = rng.normal(size=(N, 6)).astype(np.float32)
    labels = rng.choice([3, 5, 7, 8, 9], N).astype(np.int16)
    ref = rng.normal(size=(M, 6)).astype(np.float32)
    ref_lab = rng.choice([5, 7, 8], M).astype(np.int16)
    ref_lab[:N] = np.where(rng.random(N) < 0.9, 5, 8)        # >= N robot points overall
    idx = reference_robot_index(ref_lab, FREEZE, N)
    out = freeze_robot_points(cloud, labels, ref, idx, FREEZE)
    robot = np.isin(labels, FREEZE)
    check("unit: width kept", out.shape == cloud.shape and out.dtype == cloud.dtype)
    check("unit: scene positions untouched", np.array_equal(out[~robot], cloud[~robot]))
    check("unit: robot positions = reference robot points, FPS order",
          np.array_equal(out[robot], ref[np.flatnonzero(np.isin(ref_lab, FREEZE))[:robot.sum()]]))
    check("unit: input not modified", not np.shares_memory(out, cloud))
    # identity: the reference's own first-N prefix maps to itself
    self_out = freeze_robot_points(ref[:N], ref_lab[:N], ref, idx, FREEZE)
    check("unit: frame 0 maps to itself", np.array_equal(self_out, ref[:N]))
    try:
        reference_robot_index(np.full(M, 8), FREEZE, N)
        check("unit: too-small reference raises", False)
    except RuntimeError:
        check("unit: too-small reference raises", True)


def head_dataset_class():
    src = subprocess.check_output(["git", "-C", str(ROOT), "show", "HEAD:R3D/r3d/dataset/robotwin_sim_dataset.py"])
    d = tempfile.mkdtemp()
    p = os.path.join(d, "robotwin_sim_dataset_head.py")
    open(p, "wb").write(src)
    spec = importlib.util.spec_from_file_location("robotwin_sim_dataset_head", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.RobotwinSimDataset


def same_sample(a, b):
    import torch
    def flat(d, pre=""):
        out = {}
        for k, v in d.items():
            out.update(flat(v, pre + k + "/") if isinstance(v, dict) else {pre + k: v})
        return out
    fa, fb = flat(a), flat(b)
    return fa.keys() == fb.keys() and all(torch.equal(fa[k], fb[k]) for k in fa)


def sample_indices(ds, n, rng):
    """random windows plus every episode-start window (padded) and the last window."""
    ind = ds.sampler.indices
    starts = np.flatnonzero(ind[:, 2] > 0)
    ends = np.flatnonzero(ind[:, 3] < ds.horizon)
    pick = np.concatenate([rng.choice(len(ind), n, replace=False), starts[:5], ends[-5:], [len(ind) - 1]])
    return np.unique(pick)


def off_checks(n):
    import hydra
    from omegaconf import OmegaConf
    os.chdir(ROOT)
    Head = head_dataset_class()
    rng = np.random.default_rng(1)
    for name, ov in (("uncut", []), ("arm cut [5]", ["task.drop_point_labels=[5]"])):
        cfg = compose(ov)
        check(f"off/{name}: config default freeze_robot_labels is null", cfg.task.freeze_robot_labels is None)
        new = hydra.utils.instantiate(cfg.task.dataset)
        kw = OmegaConf.to_container(cfg.task.dataset, resolve=True)
        for k in ("_target_", "freeze_robot_labels", "freeze_reference", "drop_point_links"):
            kw.pop(k)
        old = Head(**kw)
        for split, (a, b) in (("train", (new, old)), ("val", (new.get_validation_dataset(), old.get_validation_dataset()))):
            idx = sample_indices(a, n, rng)
            ok = len(a) == len(b)
            for i in idx:
                np.random.seed(int(i))
                sa = a[int(i)]
                np.random.seed(int(i))
                ok &= same_sample(sa, b[int(i)])
            check(f"off/{name}/{split}: {len(idx)} samples bit-identical to git HEAD", bool(ok))
        na, nb = new.get_normalizer(), old.get_normalizer()
        import torch
        sa, sb = na.state_dict(), nb.state_dict()
        check(f"off/{name}: normalizer identical", sa.keys() == sb.keys() and all(torch.equal(sa[k], sb[k]) for k in sa))
        del new, old
    import dill, torch
    ck = ROOT / "R3D/data/outputs/sim/mimicgen_sim-square_d2-p4096-g64-42/checkpoints/250.ckpt"
    if ck.exists():
        cfg = torch.load(open(ck, "rb"), pickle_module=dill, map_location="cpu", weights_only=False)["cfg"]
        check("off: uncut 250.ckpt cfg has no freeze key (eval path unchanged)",
              cfg.task.get("freeze_robot_labels", None) is None)


def on_checks(n):
    import hydra
    import torch
    os.chdir(ROOT)
    rng = np.random.default_rng(2)
    cfg = compose([f"task.freeze_robot_labels=[{FREEZE[0]},{FREEZE[1]}]"])
    if not (ROOT / cfg.task.freeze_reference).exists():
        check("on: sidecar exists", False, str(cfg.task.freeze_reference))
        return
    plain = dict(use_data_augmentation=False, use_color_jitter=False)
    frz = hydra.utils.instantiate(cfg.task.dataset, **plain)
    rb = frz.replay_buffer
    ends = rb.episode_ends[:]
    starts = np.concatenate([[0], ends[:-1]]).astype(int)
    ref = np.load(ROOT / cfg.task.freeze_reference)
    check("on: sidecar M > N and covers every episode", int(ref["m"]) > 4096 and len(ref["cloud"]) == len(ends))
    robot_ref = [np.flatnonzero(np.isin(l, FREEZE)) for l in ref["label"]]
    check("on: every reference has >= N robot points", min(len(r) for r in robot_ref) >= 4096,
          f"min {min(len(r) for r in robot_ref)}")
    for split, ds in (("train", frz), ("val", frz.get_validation_dataset())):
        idx = sample_indices(ds, n, rng)
        ok_shape = ok_scene = ok_robot = ok_f0 = True
        n_f0 = n_later = n_changed = 0
        for i in idx:
            s = ds[int(i)]
            pc = s["obs"]["point_cloud"].numpy()
            ok_shape &= pc.shape == (ds.horizon, 4096, 6) and "point_cloud_len" not in s["obs"]
            b0, b1, s0, s1 = ds.sampler.indices[int(i)]
            frames = np.concatenate([[b0] * s0, np.arange(b0, b1), [b1 - 1] * (ds.horizon - s1)]).astype(int)
            ep = int(np.searchsorted(ends, frames[0], side="right"))
            for t, f in enumerate(frames):
                stored, lab = rb["point_cloud"][f], rb["point_label"][f]
                robot = np.isin(lab, FREEZE)
                ok_scene &= np.array_equal(pc[t][~robot], stored[~robot])
                ok_robot &= np.array_equal(pc[t][robot], ref["cloud"][ep][robot_ref[ep][:robot.sum()]])
                if f == starts[ep]:
                    n_f0 += 1
                    ok_f0 &= np.array_equal(pc[t], stored)
                else:
                    n_later += 1
                    n_changed += int(not np.array_equal(pc[t][robot], stored[robot]))
        check(f"on/{split}: shapes (T, 4096, 6), no point_cloud_len ({len(idx)} windows)", bool(ok_shape))
        check(f"on/{split}: scene points == stored frame, bit for bit", bool(ok_scene))
        check(f"on/{split}: robot points == own episode's frame-0 reference, FPS order", bool(ok_robot))
        check(f"on/{split}: robot points replaced in later frames ({n_changed}/{n_later})",
              n_later > 0 and n_changed >= 0.9 * n_later)
        check(f"on/{split}: episode frame 0 unchanged ({n_f0} frames)", bool(ok_f0 and n_f0 > 0))
    # augmentation parity with the uncut dataset (same draws)
    frz_aug = copy.copy(frz)                       # shares the 3 GB buffer; augmentation back on
    frz_aug.use_data_augmentation = frz_aug.use_color_jitter = True
    unc = hydra.utils.instantiate(compose([]).task.dataset)
    ok = len(unc) == len(frz_aug)
    for i in sample_indices(unc, n, rng):
        np.random.seed(int(i))
        a = unc[int(i)]
        np.random.seed(int(i))
        b = frz_aug[int(i)]
        b0, b1, s0, s1 = frz.sampler.indices[int(i)]
        frames = np.concatenate([[b0] * s0, np.arange(b0, b1), [b1 - 1] * (frz.horizon - s1)]).astype(int)
        robot = np.isin(rb["point_label"][frames], FREEZE)
        ok &= torch.equal(a["obs"]["point_cloud"][..., :3][~torch.from_numpy(robot)],
                          b["obs"]["point_cloud"][..., :3][~torch.from_numpy(robot)])
        ok &= torch.equal(a["obs"]["agent_pos"], b["obs"]["agent_pos"]) and torch.equal(a["action"], b["action"])
    check("on: augmentation draws = uncut run's (scene xyz, agent_pos, action bit-identical)", bool(ok))
    nf, nu = frz.get_normalizer().state_dict(), unc.get_normalizer().state_dict()
    check("on: normalizer identical to uncut", nf.keys() == nu.keys() and all(torch.equal(nf[k], nu[k]) for k in nf))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--off-only", action="store_true")
    ap.add_argument("--n", type=int, default=150, help="random windows per split")
    a = ap.parse_args()
    unit_checks()
    off_checks(a.n)
    if not a.off_only:
        on_checks(a.n)
    print(f"\n{'ALL PASS' if not FAILS else f'{len(FAILS)} FAILED: {FAILS}'}")
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
