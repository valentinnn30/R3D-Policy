#!/usr/bin/env python
"""Checks for prefix thinning (claim sweep R2 armsparse / R3 scenesparse) and the E1 inference-only cut.

    python scripts/sim/check_prefix_thin.py           # r3d_sim: dataset side (1-5)
    python scripts/sim/check_prefix_thin.py --eval    # r3d_robosuite: the 200 eval resets (6)

1. OPTION OFF: uncut and arm cut [5] datasets, train + val windows (random + episode starts / ends), same
   numpy seed -> bit-identical to git HEAD's dataset; normalizer identical; config default null; the old
   uncut 250.ckpt cfg has no thin key (eval path unchanged).
2. "arm" (cells link0..link7) == label 5 (the arm cut's set) on every stored frame.
3. Both rules together == the first 1024 points, n_valid == 1024, on every frame of all 200 demos. The 1024
   runs (uncut / arm cut 1024) read the SAME zarr with task.num_points=1024 (= the first 1024 stored points;
   checked from their .hydra overrides), so the 2 x 2 corners are exact.
4. Dataset ON, per rule, train + val: (a) augmentation off: kept points == the stored points selected by
   the rule, in stored order, n == point_cloud_len, padding == first kept point; (b) augmentation on, same
   numpy seed as the uncut dataset: output == the rule applied to the uncut dataset's augmented sample, bit
   for bit (the flags come from the stored index; augmentation draws identical, nothing reordered).
5. Kept points per frame over all 30,646 stored frames, per rule: min / p1 / median (>= 64 required).
6. (--eval) 200 eval resets (seed block 1): first 1024 of the live 4096 cloud == the live FPS-1024 cloud;
   rules as in 4a on the live cloud; kept min / p1 / median.
"""

import argparse
import importlib.util
import json
import os
import pathlib
import subprocess
import sys
import tempfile

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "R3D"))
ZARR = ROOT / "R3D/data/square_d2-agentview_p4096-200.zarr"
OUT = ROOT / "R3D/data/outputs/sim/mimicgen/claim_sweep"
PREFIX = 1024
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


def head_dataset_class():
    src = subprocess.check_output(["git", "-C", str(ROOT), "show", "HEAD:R3D/r3d/dataset/robotwin_sim_dataset.py"])
    d = tempfile.mkdtemp()
    p = os.path.join(d, "robotwin_sim_dataset_head.py")
    open(p, "wb").write(src)
    spec = importlib.util.spec_from_file_location("robotwin_sim_dataset_head", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.RobotwinSimDataset


def flat(d, pre=""):
    out = {}
    for k, v in d.items():
        out.update(flat(v, pre + k + "/") if isinstance(v, dict) else {pre + k: v})
    return out


def same_sample(a, b):
    import torch
    fa, fb = flat(a), flat(b)
    return fa.keys() == fb.keys() and all(torch.equal(fa[k], fb[k]) for k in fa)


def sample_indices(ds, n, rng):
    ind = ds.sampler.indices
    starts = np.flatnonzero(ind[:, 2] > 0)
    ends = np.flatnonzero(ind[:, 3] < ds.horizon)
    pick = np.concatenate([rng.choice(len(ind), n, replace=False), starts[:5], ends[-5:], [len(ind) - 1]])
    return np.unique(pick)


def window_frames(ds, i):
    b0, b1, s0, s1 = ds.sampler.indices[int(i)]
    return np.concatenate([[b0] * s0, np.arange(b0, b1), [b1 - 1] * (ds.horizon - s1)]).astype(int)


def off_checks(n):
    import hydra
    import torch
    from omegaconf import OmegaConf
    Head = head_dataset_class()
    rng = np.random.default_rng(1)
    for name, ov in (("uncut", []), ("arm cut [5]", ["task.drop_point_labels=[5]"])):
        cfg = compose(ov)
        check(f"off/{name}: config default thin_point_set is null", cfg.task.thin_point_set is None)
        new = hydra.utils.instantiate(cfg.task.dataset)
        kw = OmegaConf.to_container(cfg.task.dataset, resolve=True)
        for k in ("_target_", "thin_point_set", "thin_prefix"):
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
        sa, sb = new.get_normalizer().state_dict(), old.get_normalizer().state_dict()
        check(f"off/{name}: normalizer identical", sa.keys() == sb.keys() and all(torch.equal(sa[k], sb[k]) for k in sa))
        del new, old
    import dill
    ck = ROOT / "R3D/data/outputs/sim/mimicgen_sim-square_d2-p4096-g64-42/checkpoints/250.ckpt"
    if ck.exists():
        cfg = torch.load(open(ck, "rb"), pickle_module=dill, map_location="cpu", weights_only=False)["cfg"]
        check("off: uncut 250.ckpt cfg has no thin key (eval path unchanged)", cfg.task.get("thin_point_set", None) is None)


def stored_checks():
    import zarr
    from r3d.common.label_cut import cut_points_by_label
    from r3d.common.prefix_thin import arm_cell_ids, thin_drop_labels
    z = zarr.open(str(ZARR), "r")
    arm = arm_cell_ids()
    F = z["data/point_cloud"].shape[0]
    ok_arm = ok_both = True
    kept = {"arm": [], "non_arm": []}
    for s in range(0, F, 2000):
        link = z["data/point_link"][s:s + 2000]
        lab = z["data/point_label"][s:s + 2000]
        ok_arm &= np.array_equal(np.isin(link, arm), lab == 5)
        d = {r: thin_drop_labels(link, r, PREFIX, arm) for r in kept}
        for r in kept:
            kept[r].append(4096 - d[r].sum(1))
        both = (d["arm"] | d["non_arm"]).astype(bool)
        ok_both &= bool((both.sum(1) == 4096 - PREFIX).all() and not both[:, :PREFIX].any())
        if s == 0:   # full cut path on a few frames: kept == the first 1024 points, n == 1024
            pc = z["data/point_cloud"][0:50]
            for t in range(50):
                out, nv = cut_points_by_label(pc[t], both[t].astype(np.int8), [1])
                ok_both &= nv == PREFIX and np.array_equal(out[:PREFIX], pc[t][:PREFIX])
    check(f"stored: arm cells link0..link7 == label 5 on all {F} frames", bool(ok_arm))
    check(f"stored: both rules together == the first {PREFIX} points on all {F} frames", bool(ok_both))
    for name in ("mimicgen_sim-square_d2-p1024-g64-42", "mimicgen_sim-square_d2-p1024-g64-43"):
        ov = (ROOT / f"R3D/data/outputs/sim/{name}/.hydra/overrides.yaml").read_text()
        check(f"stored: {name} read the 4096 zarr's first 1024 points",
              "setting=agentview_p4096-200" in ov and "task.num_points=1024" in ov)
    stats = {}
    for r in kept:
        k = np.concatenate(kept[r])
        stats[r] = dict(min=int(k.min()), p1=float(np.percentile(k, 1)), median=float(np.median(k)))
        check(f"stored: kept points per frame, {r}: min {stats[r]['min']} p1 {stats[r]['p1']:.0f} "
              f"median {stats[r]['median']:.0f} (>= 64)", stats[r]["min"] >= 64)
    return stats


def on_checks(n):
    import hydra
    import torch
    from r3d.common.prefix_thin import arm_cell_ids, thin_cloud, thin_drop_labels
    arm = arm_cell_ids()
    rng = np.random.default_rng(3)
    unc = hydra.utils.instantiate(compose([]).task.dataset)
    for rule in ("arm", "non_arm"):
        cfg = compose([f"task.thin_point_set={rule}"])
        ds = hydra.utils.instantiate(cfg.task.dataset)
        rb = ds.replay_buffer
        for split, d, u in (("train", ds, unc), ("val", ds.get_validation_dataset(), unc.get_validation_dataset())):
            idx = sample_indices(d, n, rng)
            ok_a = ok_b = True
            for i in idx:
                frames = window_frames(d, i)
                # (a) augmentation off
                d.use_data_augmentation = d.use_color_jitter = False
                s = d[int(i)]
                d.use_data_augmentation = d.use_color_jitter = True
                pc, ln = s["obs"]["point_cloud"].numpy(), s["obs"]["point_cloud_len"].numpy()
                for t, f in enumerate(frames):
                    stored, link = rb["point_cloud"][f], rb["point_link"][f]
                    keep = thin_drop_labels(link, rule, PREFIX, arm) == 0
                    nk = int(keep.sum())
                    ok_a &= (ln[t] == nk and np.array_equal(pc[t][:nk], stored[keep])
                             and (pc[t][nk:] == pc[t][0]).all())
                # (b) augmentation on, same numpy seed as the uncut dataset
                np.random.seed(int(i)); su = u[int(i)]
                np.random.seed(int(i)); st = d[int(i)]
                pu = su["obs"]["point_cloud"].numpy()
                for t, f in enumerate(frames):
                    exp, nk = thin_cloud(pu[t], rb["point_link"][f], rule, PREFIX, arm)
                    ok_b &= (np.array_equal(st["obs"]["point_cloud"][t].numpy(), exp)
                             and int(st["obs"]["point_cloud_len"][t]) == nk)
                ok_b &= torch.equal(su["obs"]["agent_pos"], st["obs"]["agent_pos"]) and torch.equal(su["action"], st["action"])
            check(f"on/{rule}/{split}: no aug -> kept == stored points selected by the rule, len, padding ({len(idx)} windows)", bool(ok_a))
            check(f"on/{rule}/{split}: aug on -> == the rule applied to the uncut augmented sample, bit for bit", bool(ok_b))
        del ds


def eval_checks(scenes):
    import copy
    import h5py
    import torch
    import zarr
    import robosuite as suite
    import mimicgen  # noqa: F401
    from r3d.common import robosuite_pcd as RP
    from r3d.common.prefix_thin import arm_cell_ids, thin_cloud, thin_drop_labels
    arm = arm_cell_ids()
    z = dict(zarr.open(str(ZARR), "r")["meta"].attrs)
    with h5py.File(ROOT / "R3D/data/mimicgen/core" / f"{z['task']}.hdf5", "r") as f:
        env_args = json.loads(f["data"].attrs["env_args"])
    kw = copy.deepcopy(env_args["env_kwargs"])
    kw["controller_configs"]["control_delta"] = False
    cam, res = z["camera"], int(z["render_res"])
    kw.update(camera_names=[cam], camera_heights=res, camera_widths=res, camera_depths=True,
              camera_segmentations="element", has_offscreen_renderer=True, use_camera_obs=True, ignore_done=True)
    env = suite.make(z.get("eval_env_name", env_args["env_name"]), **kw)
    ok_pre = ok_rule = ok_lab = True
    kept = {"arm": [], "non_arm": []}
    for k in range(scenes):
        seed = 100000 + k
        np.random.seed(seed); torch.manual_seed(seed)
        obs = env.reset()
        bbox = RP.default_bbox(env.sim)
        lut = RP.geom_label_lut(env.sim, env_args["env_name"])
        cells = RP.LinkCells(env.sim)
        full, lab, _, cell = RP.CloudBuilder(env.sim, cam, res, res, bbox, 4096, lut).build(obs, cells=cells)
        pre, pre_lab, _ = RP.CloudBuilder(env.sim, cam, res, res, bbox, PREFIX, lut).build(obs)
        ok_pre &= np.array_equal(full[:PREFIX], pre) and np.array_equal(lab[:PREFIX], pre_lab)
        ok_lab &= np.array_equal(np.isin(cell, arm), lab == 5)
        for r in kept:
            out, nk = thin_cloud(full, cell, r, PREFIX, arm)
            keep = thin_drop_labels(cell, r, PREFIX, arm) == 0
            ok_rule &= nk == keep.sum() and np.array_equal(out[:nk], full[keep])
            kept[r].append(nk)
    check(f"eval: first {PREFIX} of the live 4096 cloud == the live FPS-{PREFIX} cloud ({scenes} resets)", bool(ok_pre))
    check(f"eval: live arm cells == label 5 ({scenes} resets)", bool(ok_lab))
    check(f"eval: rules applied to the live cloud as in training ({scenes} resets)", bool(ok_rule))
    stats = {}
    for r in kept:
        k = np.array(kept[r])
        stats[r] = dict(min=int(k.min()), p1=float(np.percentile(k, 1)), median=float(np.median(k)))
        check(f"eval: kept points, {r}: min {stats[r]['min']} p1 {stats[r]['p1']:.0f} median {stats[r]['median']:.0f}",
              stats[r]["min"] >= 64)
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", action="store_true")
    ap.add_argument("--n", type=int, default=40, help="random windows per dataset split")
    ap.add_argument("--scenes", type=int, default=200)
    a = ap.parse_args()
    os.chdir(ROOT)
    OUT.mkdir(parents=True, exist_ok=True)
    if a.eval:
        stats = eval_checks(a.scenes)
        (OUT / "kept_points_eval.json").write_text(json.dumps(stats, indent=1))
    else:
        off_checks(a.n)
        stats = stored_checks()
        (OUT / "kept_points_train.json").write_text(json.dumps(stats, indent=1))
        on_checks(a.n)
    print("ALL PASS" if not FAILS else f"FAILED: {FAILS}")
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
