#!/usr/bin/env python
"""Checks for the arm-cut-style LINK cut (`task.drop_point_links`), e.g. the arm cut minus link7.

    python scripts/sim/check_link_cut.py          # r3d_sim: dataset checks
    python scripts/sim/check_link_cut.py --eval   # r3d_robosuite: live eval-side cells

Dataset (stored zarr, train + val windows incl. episode starts / ends):
1. config default drop_point_links is null; unknown cell names raise; combining with
   drop_point_labels raises.
2. label 5 == cells link0..link7 on EVERY stored point of the 4096 prefix.
3. drop_point_links=[link0..link7], same numpy seed -> every sample bit-identical to the arm cut
   (drop_point_labels=[5]), augmentation on.
4. drop_point_links=[link0..link6], same numpy seed -> == the UNCUT sample (same augmentation
   draws) cut by the window's stored point_link with cut_points_by_label; kept = arm-cut kept +
   exactly the link7 points; no point_link / point_label leaks into the obs; link7 present.
Eval (--eval): 10 reset scenes of the eval block, live CloudBuilder + LinkCells:
5. cut by live cells link0..7 == cut by live label 5 (cloud and count), and link0..6 keeps the
   label-5 cut plus exactly the live link7 points.
"""

import argparse
import copy
import json
import os
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "R3D"))
FAILS = []
L07 = [f"link{i}" for i in range(8)]
L06 = L07[:7]


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


def links(names):
    return "task.drop_point_links=[" + ",".join(names) + "]"


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
    ind = ds.sampler.indices
    starts = np.flatnonzero(ind[:, 2] > 0)
    ends = np.flatnonzero(ind[:, 3] < ds.horizon)
    pick = np.concatenate([rng.choice(len(ind), n, replace=False), starts[:5], ends[-5:], [len(ind) - 1]])
    return np.unique(pick)


def window_frames(ds, i):
    b0, b1, s0, s1 = ds.sampler.indices[int(i)]
    return np.concatenate([[b0] * s0, np.arange(b0, b1), [b1 - 1] * (ds.horizon - s1)]).astype(int)


def dataset_checks(n):
    import hydra
    import torch
    import zarr
    from r3d.common.label_cut import cut_points_by_label
    os.chdir(ROOT)
    rng = np.random.default_rng(3)

    check("config: drop_point_links defaults to null", compose([]).task.drop_point_links is None)
    for name, ov in (("unknown cell name raises", [links(["link9"])]),
                     ("with drop_point_labels raises", [links(L06), "task.drop_point_labels=[5]"])):
        try:
            hydra.utils.instantiate(compose(ov).task.dataset)
            check(f"config: {name}", False)
        except Exception as e:  # hydra wraps the ValueError
            check(f"config: {name}", "drop_point_links" in str(e) or "is not in" in str(e) or "link9" in str(e), str(e)[-80:])

    z = zarr.open(str(ROOT / compose([]).task.dataset.zarr_path), "r")["data"]
    eq, nf = True, 0
    for s in range(0, z["point_label"].shape[0], 5000):
        lab = z["point_label"][s:s + 5000, :4096]
        cell = z["point_link"][s:s + 5000, :4096]
        eq &= np.array_equal(lab == 5, (cell >= 0) & (cell <= 7))
        nf += len(lab)
    check(f"data: label 5 == cells link0..link7 on every stored point ({nf} frames)", bool(eq))

    lc07 = hydra.utils.instantiate(compose([links(L07)]).task.dataset)
    arm = hydra.utils.instantiate(compose(["task.drop_point_labels=[5]"]).task.dataset)
    for split, (a, b) in (("train", (lc07, arm)), ("val", (lc07.get_validation_dataset(), arm.get_validation_dataset()))):
        idx = sample_indices(a, n, rng)
        ok = len(a) == len(b)
        for i in idx:
            np.random.seed(int(i)); sa = a[int(i)]
            np.random.seed(int(i)); ok &= same_sample(sa, b[int(i)])
        check(f"link0..7 == arm cut [5]/{split}: {len(idx)} samples bit-identical (augmentation on)", bool(ok))
    del arm

    lc06 = copy.copy(lc07)                     # shares the buffer (point_link loaded)
    lc06.drop_point_links = [lc07.drop_point_links[i] for i in range(7)]
    unc = copy.copy(lc07)
    unc.drop_point_links = None                # the uncut sample, same draws
    cfg06 = hydra.utils.instantiate(compose([links(L06)]).task.dataset)
    check("link0..6: config ids == link0..6 cell ids", cfg06.drop_point_links == list(range(7)), str(cfg06.drop_point_links))
    del cfg06
    rb = lc07.replay_buffer
    for split, (a, u) in (("train", (lc06, unc)), ("val", (lc06.get_validation_dataset(), unc.get_validation_dataset()))):
        idx = sample_indices(a, n, rng)
        ok_ref = ok_keys = ok_split = True
        n7 = 0
        for i in idx:
            np.random.seed(int(i)); s6 = a[int(i)]
            np.random.seed(int(i)); su = u[int(i)]
            ok_keys &= set(s6["obs"]) == {"point_cloud", "agent_pos", "point_cloud_len"}
            pcu = su["obs"]["point_cloud"].numpy()
            for t, f in enumerate(window_frames(a, i)):
                cell = rb["point_link"][f]
                ref, ln = cut_points_by_label(pcu[t], cell, list(range(7)))
                ok_ref &= np.array_equal(s6["obs"]["point_cloud"][t].numpy(), ref) and int(s6["obs"]["point_cloud_len"][t]) == ln
                arm_n = int((~np.isin(cell, list(range(8)))).sum())
                k7 = int((cell == 7).sum())
                ok_split &= ln == arm_n + k7
                n7 += k7
            ok_ref &= torch.equal(s6["action"], su["action"]) and torch.equal(s6["obs"]["agent_pos"], su["obs"]["agent_pos"])
        check(f"link0..6/{split}: == uncut sample (same draws) cut by stored point_link ({len(idx)} windows)", bool(ok_ref))
        check(f"link0..6/{split}: obs keys point_cloud, agent_pos, point_cloud_len only", bool(ok_keys))
        check(f"link0..6/{split}: kept = arm-cut kept + exactly the link7 points", bool(ok_split))
        check(f"link0..6/{split}: link7 present (mean {n7 / (len(idx) * a.horizon):.0f} pts/frame)", n7 > 0)


def eval_checks(n_scenes):
    import copy as _c
    import h5py
    import zarr
    import robosuite as suite
    sys.path.insert(0, str(ROOT / "scripts" / "sim"))
    import mimicgen  # noqa: F401
    from r3d.common import robosuite_pcd as RP
    from r3d.common.label_cut import cut_points_by_label
    os.chdir(ROOT)
    zp = ROOT / "R3D/data/square_d2-agentview_p4096-200.zarr"
    za = dict(zarr.open(str(zp), "r")["meta"].attrs)
    with h5py.File(ROOT / "R3D/data/mimicgen/core" / f"{za['task']}.hdf5", "r") as f:
        env_args = json.loads(f["data"].attrs["env_args"])
    kw = _c.deepcopy(env_args["env_kwargs"])
    kw["controller_configs"]["control_delta"] = False
    cam, res = za["camera"], int(za["render_res"])
    kw.update(camera_names=[cam], camera_heights=res, camera_widths=res, camera_depths=True,
              camera_segmentations="element", has_offscreen_renderer=True, use_camera_obs=True, ignore_done=True)
    env = suite.make(env_args["env_name"], **kw)
    ok07 = ok06 = True
    n7 = narm = 0
    for seed in range(100000, 100000 + n_scenes):
        np.random.seed(seed)
        obs = env.reset()
        bbox = RP.default_bbox(env.sim)
        cb = RP.CloudBuilder(env.sim, cam, res, res, bbox, 4096, RP.geom_label_lut(env.sim, env_args["env_name"]))
        pc, lab, _, cell = cb.build(obs, cells=RP.LinkCells(env.sim))
        a_pc, a_n = cut_points_by_label(pc, lab, [5])
        c_pc, c_n = cut_points_by_label(pc, cell, list(range(8)))
        ok07 &= a_n == c_n and np.array_equal(a_pc, c_pc)
        s_pc, s_n = cut_points_by_label(pc, cell, list(range(7)))
        k7 = int((cell == 7).sum())
        ok06 &= s_n == a_n + k7 and np.array_equal(s_pc[:s_n], pc[~np.isin(cell, list(range(7)))])
        n7 += k7
        narm += int((lab == 5).sum())
    check(f"eval: live cells link0..7 cut == live label-5 cut ({n_scenes} scenes)", bool(ok07))
    check(f"eval: link0..6 cut = label-5 cut + exactly the live link7 points", bool(ok06))
    check(f"eval: arm and link7 in view (mean {narm / n_scenes:.0f} arm, {n7 / n_scenes:.0f} link7 pts)", n7 > 0 and narm > 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", action="store_true")
    ap.add_argument("--n", type=int, default=40)
    a = ap.parse_args()
    if a.eval:
        eval_checks(10)
    else:
        dataset_checks(a.n)
    print(f"\n{'ALL PASS' if not FAILS else f'{len(FAILS)} FAIL: ' + ', '.join(FAILS)}")
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
