#!/usr/bin/env python
"""Initial scene layouts (object poses at t=0) for RoboTwin seeds -- training AND eval.

Used by the memorization check (`scene_distance_analysis.py`): is a test scene's
success explained by how close it is to a training layout?

    python scripts/sim/dump_scene_layouts.py --task hanging_mug_labeled \
        --zarr R3D/data/hanging_mug_labeled-sim_stock_p4096-50.zarr \
        --eval-groups 200000sim_stock_p1024 200000sim_stock_p2048 200000sim_stock_p4096

Each scene is rebuilt with the REAL `setup_demo(seed=...)` -- the same call
collection (`collect_data.py`) and evaluation (`RoboTwin2EnvManager.Init_task_env`)
make -- so placement, settling (`check_stable`) and every RNG draw before
`load_actors` are reproduced. No planning, no policy. Poses are read after
setup_demo returns, i.e. the scene the policy sees at step 0.

Run in r3d_sim. Output: R3D/data/outputs/sim/scene_layouts/<task>.json
"""

import argparse
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
R3D_DIR = os.path.join(ROOT, "R3D")
RT_DIR = os.path.join(R3D_DIR, "r3d", "env", "robotwin2")
OUT_DIR = os.path.join(R3D_DIR, "data", "outputs", "sim", "scene_layouts")


def _pose(actor):
    p = actor.get_pose()
    return list(map(float, p.p)), list(map(float, p.q))


def scene_record(task):
    """Object poses of a freshly set-up scene. Task-agnostic: every Actor attribute
    named in point_label_roles() plus its functional point 0 when it has one."""
    rec = {}
    for role, actors in task.point_label_roles().items():
        for i, a in enumerate(actors):
            p, q = _pose(a)
            r = {"p": p, "q": q}
            try:
                r["fp0"] = list(map(float, a.get_functional_point(0)[:3]))
            except Exception:
                pass
            rec[f"{role}{i}"] = r
    if hasattr(task, "mug_id"):
        rec["mug_id"] = int(task.mug_id)
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--zarr", required=True, help="training zarr; its attrs carry the seeds + task_config")
    ap.add_argument("--eval-groups", nargs="+", default=[],
                    help="keys of seeds_list/<task>.json whose seeds to dump")
    ap.add_argument("--eval-config", default="sim_stock_p4096",
                    help="task config used to rebuild eval scenes (placement does not depend on it)")
    ap.add_argument("--limit", type=int, default=None, help="debug: first N seeds of each kind")
    a = ap.parse_args()

    import zarr
    zattrs = dict(zarr.open(os.path.join(ROOT, a.zarr), "r")["meta"].attrs)
    train_seeds = [int(s) for s in zattrs["seeds"]]
    train_cfg = zattrs["task_config"]

    with open(os.path.join(RT_DIR, "seeds_list", f"{a.task}.json")) as f:
        lists = json.load(f)
    eval_seeds = sorted({int(s) for g in a.eval_groups for s in lists[g]["seeds"]})
    if a.limit:
        train_seeds, eval_seeds = train_seeds[:a.limit], eval_seeds[:a.limit]

    os.chdir(RT_DIR)
    sys.path.insert(0, R3D_DIR)
    from r3d.env.robotwin2 import RoboTwin2EnvManager

    out = {"task": a.task, "zarr": a.zarr, "train_config": train_cfg,
           "eval_groups": {g: lists[g]["seeds"] for g in a.eval_groups},
           "train": {}, "eval": {}}
    for kind, seeds, cfg, eval_mode in (("train", train_seeds, train_cfg, False),
                                        ("eval", eval_seeds, a.eval_config, True)):
        mgr = RoboTwin2EnvManager()
        # task_num=1: only builds args + task; the eval list is NOT used here
        mgr.Create_env(task_name=a.task, head_camera_type=None, seed=1, task_num=1,
                       instruction_type="unseen", task_config=cfg)
        args = dict(mgr.args)
        args["eval_mode"] = eval_mode
        t0 = time.time()
        for i, s in enumerate(seeds):
            mgr.task.setup_demo(now_ep_num=i, seed=s, is_test=eval_mode, **args)
            out[kind][str(s)] = scene_record(mgr.task)
            mgr.task.close_env(clear_cache=((i + 1) % args.get("clear_cache_freq", 5) == 0))
            print(f"[{kind}] {i + 1}/{len(seeds)} seed {s}  {time.time() - t0:.0f}s", flush=True)

    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, f"{a.task}{'_debug' if a.limit else ''}.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=1)
    print("wrote", path)


if __name__ == "__main__":
    main()
