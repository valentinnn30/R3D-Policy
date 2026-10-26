#!/usr/bin/env python
"""Evaluate R3D checkpoints on a MimicGen task over FIXED seeded scenes.  Run in r3d_robosuite.

    python scripts/sim/eval_mimicgen_ckpt.py <run>/checkpoints/250.ckpt --episodes 200 --workers 3

Scene k of seed block b is `np.random.seed(100000 * b + k)` before `env.reset()` (robosuite
places objects with the global numpy RNG -- checked: identical layouts on repeat), so every
checkpoint sees the SAME scenes and results pair across runs (McNemar). torch is re-seeded
with the scene seed too, so the policy's diffusion noise is deterministic per scene.

Observation = exactly the training data: the agentview cloud from `robosuite_pcd`
(same box, read from the training zarr's attrs, and asserted), first `num_points` of an FPS
with deterministic start, agent_pos / action from `robosuite_actions`, absolute OSC
(control_delta=False). With `task.drop_point_labels` the oracle label cut is applied as in
training. Episode cap 400 steps (EquiDiff's `max_steps['square_d2']`); success =
env._check_success() at any step.

Output per checkpoint: <run>/eval/<ckpt>_<episodes>ep_block<b>.json with per-scene seed,
success, steps and initial layout (nut / peg pos+quat) for the distance analysis.
"""

import argparse
import copy
import json
import multiprocessing as mp
import os
import pathlib
import sys
import time

os.environ.setdefault("MUJOCO_GL", "egl")
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
R3D_DIR = ROOT / "R3D"
MAX_STEPS = {"square_d2": 400}


def load_policy(ck):
    import dill
    import hydra
    import torch
    from omegaconf import OmegaConf
    OmegaConf.register_new_resolver("eval", eval, replace=True)
    payload = torch.load(open(ck, "rb"), pickle_module=dill, map_location="cpu", weights_only=False)
    cfg = copy.deepcopy(payload["cfg"])
    policy = hydra.utils.instantiate(cfg.policy)
    key = "ema_model" if cfg.training.use_ema else "model"
    policy.load_state_dict(payload["state_dicts"][key], strict=True)
    policy.eval().cuda()
    return policy, cfg, key


def run_scenes(args):
    ck, seeds, a = args
    os.chdir(ROOT)   # as train.py: relative cfg paths resolve against the repo root
    sys.path.insert(0, str(R3D_DIR))
    sys.path.insert(0, str(ROOT / "scripts" / "sim"))
    import torch
    import zarr
    import robosuite as suite
    import mimicgen  # noqa: F401  registers the D0/D1/D2 envs
    from r3d.common import robosuite_actions as RA
    from r3d.common import robosuite_pcd as RP
    from r3d.common.label_cut import cut_points_by_label
    from convert_mimicgen_to_zarr import _layout

    policy, cfg, _ = load_policy(ck)
    zattrs = dict(zarr.open(str(ROOT / cfg.task.dataset.zarr_path), "r")["meta"].attrs)
    task, N = zattrs["task"], int(cfg.task.num_points)
    drop = cfg.task.get("drop_point_labels", None)
    n_obs = int(cfg.n_obs_steps)

    import h5py
    with h5py.File(R3D_DIR / "data" / "mimicgen" / "core" / f"{task}.hdf5", "r") as f:
        env_args = json.loads(f["data"].attrs["env_args"])
    kw = copy.deepcopy(env_args["env_kwargs"])
    kw["controller_configs"]["control_delta"] = False
    cam, res = zattrs["camera"], int(zattrs["render_res"])
    kw.update(camera_names=[cam], camera_heights=res, camera_widths=res, camera_depths=True,
              camera_segmentations="element", has_offscreen_renderer=True, use_camera_obs=True,
              ignore_done=True)
    env = suite.make(env_args["env_name"], **kw)

    rows = []
    for seed in seeds:
        np.random.seed(seed)
        torch.manual_seed(seed)
        obs = env.reset()
        bbox = RP.default_bbox(env.sim)
        if not np.allclose(bbox, zattrs["pcd_crop_bbox"]):
            raise RuntimeError(f"crop box {bbox} != training {zattrs['pcd_crop_bbox']}")
        cb = RP.CloudBuilder(env.sim, cam, res, res, bbox, N, RP.geom_label_lut(env.sim, env_args["env_name"]))
        layout = _layout(env, task)

        def observe(o):
            pc, lab, _ = cb.build(o)
            item = {"agent_pos": torch.from_numpy(RA.state_vector(o["robot0_eef_pos"], o["robot0_eef_quat"],
                                                                  o["robot0_gripper_qpos"]))}
            if drop:
                pc, n = cut_points_by_label(pc, lab, list(drop))
                item["point_cloud_len"] = torch.tensor(n, dtype=torch.int64)
            item["point_cloud"] = torch.from_numpy(pc)
            return item

        hist = [observe(obs)] * n_obs
        steps, success = 0, False
        while steps < MAX_STEPS[task] and not success:
            batch = {k: torch.stack([h[k] for h in hist])[None].cuda() for k in hist[0]}
            with torch.no_grad():
                chunk = policy.predict_action(batch)["action"][0].cpu().numpy()
            for a10 in chunk:
                obs, _, _, _ = env.step(RA.env_action(a10))
                steps += 1
                hist = hist[1:] + [observe(obs)]
                if env._check_success():
                    success = True
                    break
                if steps >= MAX_STEPS[task]:
                    break
        rows.append({"seed": int(seed), "success": bool(success), "steps": steps, "layout": layout.tolist()})
        print(f"[{os.getpid()}] scene {seed}: {'SUCCESS' if success else 'FAIL'} ({steps} steps)", flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ckpts", nargs="+")
    ap.add_argument("--episodes", type=int, default=200)
    ap.add_argument("--block", type=int, default=1, help="scene seed block: seeds 100000*block + k")
    ap.add_argument("--workers", type=int, default=3)
    a = ap.parse_args()

    seeds = [100000 * a.block + k for k in range(a.episodes)]
    ctx = mp.get_context("spawn")
    for ck in [str(pathlib.Path(c).resolve()) for c in a.ckpts]:
        ckp = pathlib.Path(ck)
        out = ckp.parent.parent / "eval" / f"{ckp.stem}_{a.episodes}ep_block{a.block}.json"
        if out.exists():
            print("exists, skipping:", out)
            continue
        t0 = time.time()
        parts = [seeds[i::a.workers] for i in range(a.workers)]
        with ctx.Pool(a.workers) as pool:
            res = pool.map(run_scenes, [(ck, p, a) for p in parts])
        rows = sorted([r for part in res for r in part], key=lambda r: r["seed"])
        sr = float(np.mean([r["success"] for r in rows]))
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as fh:
            json.dump({"ckpt": ck, "episodes": a.episodes, "block": a.block, "success_rate": sr,
                       "minutes": (time.time() - t0) / 60, "scenes": rows}, fh, indent=1)
        print(f"ckpt: {ck}  success {sr:.3f} ({sum(r['success'] for r in rows)}/{len(rows)})  "
              f"{(time.time() - t0) / 60:.1f} min -> {out}", flush=True)


if __name__ == "__main__":
    main()
