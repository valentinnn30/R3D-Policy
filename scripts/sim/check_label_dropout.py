#!/usr/bin/env python
"""Checks for the class-level robot-part dropout (`task.dropout_point_labels`), hanging_mug 4096.  r3d_sim, CPU.

    python scripts/sim/check_label_dropout.py

1. DEFAULT OFF: the RoboTwin dataset (uncut, and the arm cut [4,5]) bit-identical to git HEAD, train + val.
2. ON ([4,5,6,7], keep 0.5), same numpy seed per window:
   - every training sample == the UNCUT sample (same augmentation draws) cut by ONE subset D of the classes, the same
     D in every frame of the window, with point_cloud_len;
   - each class is dropped in ~50% of the windows where it is present (binomial check);
   - validation samples == the uncut validation samples (full cloud, no point_cloud_len).
"""

import copy
import importlib.util
import itertools
import os
import pathlib
import subprocess
import sys
import tempfile

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "R3D"))
FAILS = []
CLASSES = [4, 5, 6, 7]


def check(name, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name}{'  -- ' + detail if detail else ''}", flush=True)
    if not ok:
        FAILS.append(name)


def compose(overrides):
    from hydra import compose as hcompose, initialize_config_dir
    from omegaconf import OmegaConf
    OmegaConf.register_new_resolver("eval", eval, replace=True)
    with initialize_config_dir(config_dir=str(ROOT / "R3D" / "r3d" / "config"), version_base=None):
        return hcompose("r3d_robotwin2", overrides=["task=robotwin2_sim", "task_name=hanging_mug_labeled",
                                                    "setting=sim_stock_p4096", "task.num_points=4096",
                                                    "task.num_group=64"] + overrides)


def head_class():
    src = subprocess.check_output(["git", "-C", str(ROOT), "show", "HEAD:R3D/r3d/dataset/robotwin_sim_dataset.py"])
    p = os.path.join(tempfile.mkdtemp(), "robotwin_sim_dataset_head.py")
    open(p, "wb").write(src)
    spec = importlib.util.spec_from_file_location("robotwin_sim_dataset_head", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.RobotwinSimDataset


def same(a, b):
    import torch
    def flat(d, pre=""):
        out = {}
        for k, v in d.items():
            out.update(flat(v, pre + k + "/") if isinstance(v, dict) else {pre + k: v})
        return out
    fa, fb = flat(a), flat(b)
    return fa.keys() == fb.keys() and all(torch.equal(fa[k], fb[k]) for k in fa)


def frames(ds, i):
    b0, b1, s0, s1 = ds.sampler.indices[int(i)]
    return np.concatenate([[b0] * s0, np.arange(b0, b1), [b1 - 1] * (ds.horizon - s1)]).astype(int)


def main():
    import hydra
    from omegaconf import OmegaConf
    from r3d.common.label_cut import cut_points_by_label
    os.chdir(ROOT)
    rng = np.random.default_rng(0)
    Head = head_class()
    for name, ov in (("uncut", []), ("arm cut [4,5]", ["task.drop_point_labels=[4,5]"])):
        cfg = compose(ov)
        check(f"off/{name}: dropout_point_labels defaults to null", cfg.task.dropout_point_labels is None)
        new = hydra.utils.instantiate(cfg.task.dataset)
        kw = OmegaConf.to_container(cfg.task.dataset, resolve=True)
        for k in ("_target_", "dropout_point_labels", "dropout_keep"):
            kw.pop(k)
        old = Head(**kw)
        for split, (a, b) in (("train", (new, old)), ("val", (new.get_validation_dataset(), old.get_validation_dataset()))):
            idx = rng.choice(len(a), min(60, len(a)), replace=False)
            ok = len(a) == len(b)
            for i in idx:
                np.random.seed(int(i)); sa = a[int(i)]
                np.random.seed(int(i)); ok &= same(sa, b[int(i)])
            check(f"off/{name}/{split}: {len(idx)} samples bit-identical to git HEAD", bool(ok))
        del new, old

    on = hydra.utils.instantiate(compose(["task.dropout_point_labels=[4,5,6,7]"]).task.dataset)
    unc = copy.copy(on)
    unc.dropout_point_labels = None              # the uncut sample, same draws (point_label loaded, unused)
    unc.dropout_active = False
    lab = on.replay_buffer["point_label"]
    subsets = [list(s) for r in range(5) for s in itertools.combinations(CLASSES, r)]
    idx = rng.choice(len(on), 400, replace=False)
    ok_match = ok_len = True
    dropped = {c: [] for c in CLASSES}
    for i in idx:
        np.random.seed(int(i)); so = on[int(i)]
        np.random.seed(int(i)); su = unc[int(i)]
        ok_len &= "point_cloud_len" in so["obs"]
        pcu = su["obs"]["point_cloud"].numpy()
        fr = frames(on, i)
        hit = None
        for D in subsets:
            if all(np.array_equal(so["obs"]["point_cloud"][t].numpy(), cut_points_by_label(pcu[t], lab[f], D)[0])
                   and int(so["obs"]["point_cloud_len"][t]) == cut_points_by_label(pcu[t], lab[f], D)[1]
                   for t, f in enumerate(fr)):
                hit = D
                break
        ok_match &= hit is not None
        if hit is None:
            continue
        present = {c for f in fr for c in np.unique(lab[f]).tolist()}
        for c in CLASSES:
            if c in present:
                dropped[c].append(c in hit)
    check(f"on: every sample == uncut sample cut by one class subset, same in all frames ({len(idx)} windows)", bool(ok_match))
    check("on: every training sample carries point_cloud_len", bool(ok_len))
    for c in CLASSES:
        d = np.array(dropped[c])
        if len(d) == 0:
            check(f"on: class {c} present in some window", False)
            continue
        z = (d.mean() - 0.5) / np.sqrt(0.25 / len(d))
        check(f"on: class {c} dropped in {d.mean():.1%} of {len(d)} windows where present (|z| < 3)", abs(z) < 3)
    val_on, val_unc = on.get_validation_dataset(), unc.get_validation_dataset()
    ok = True
    for i in rng.choice(len(val_on), min(60, len(val_on)), replace=False):
        np.random.seed(int(i)); a = val_on[int(i)]
        np.random.seed(int(i)); b = val_unc[int(i)]
        ok &= same(a, b) and "point_cloud_len" not in a["obs"]
    check("on/val: full cloud, == the uncut validation sample, no point_cloud_len", bool(ok))
    print(f"\n{'ALL PASS' if not FAILS else f'{len(FAILS)} FAIL: ' + ', '.join(FAILS)}")
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
