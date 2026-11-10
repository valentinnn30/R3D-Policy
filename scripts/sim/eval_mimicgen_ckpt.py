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
training; `task.drop_point_links` cuts the same way by the live link cell (`LinkCells`).
A learned-link-remover checkpoint (DP3LinkRemover) gets `point_link` -- the link
cells from `robosuite_pcd.LinkCells`, the same code that labelled the training zarr -- and
thins with its deterministic, snapped rho; the rho table goes into the json. With
`task.freeze_robot_labels` (round 7) every observation's robot points are replaced by those of
the rollout's FIRST frame (`r3d/common/robot_freeze.py`, as in training): at env.reset() the
FPS-M reference is built (M from the training sidecar) and its first N points are asserted equal
to the normal N-point cloud. Episode cap 400 steps (EquiDiff's `max_steps['square_d2']`); success =
env._check_success() at any step.

`--remover-removed` (gates-only remover checkpoints): evaluate with ANOTHER mask than the learned
one -- the listed regions removed, every other region kept ("none" = keep everything, i.e. the
uncut cloud). A listed LINK CELL that is in no region of the checkpoint (link0..link4 for the
default 8 regions) is removed from the cloud before the policy sees it (compacted + padded, with
point_cloud_len -- the remover's padding contract). Written to <ckpt>_<episodes>ep_block<b>_mask-<removed>.json, so the learned-mask
result and the sweep analysis (which reads only *_block<b>.json) are untouched.

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
MAX_STEPS = {"square_d2": 400, "threading_d1": 500}   # threading demos: median 215, max 261 steps


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


def parse_removed(arg):
    return [] if arg == "none" else [r for r in arg.split(",") if r]


def override_mask(policy, arg):
    """inference mask override: listed regions removed, the rest kept whole. -> cell ids to remove
    from the cloud beforehand (listed link cells that are in no region)."""
    import torch
    rem = getattr(policy, "link_remover", None)
    if rem is None or rem.learn_density or not rem.learn_gates:
        raise ValueError("--remover-removed needs a gates-only link-remover checkpoint")
    in_region = {rem.cell_names[c] for c in range(len(rem.cell_names)) if int(rem.cell_to_region[c]) >= 0}
    extra = [r for r in parse_removed(arg) if r not in rem.region_names and r in rem.cell_names and r not in in_region]
    removed = [r for r in parse_removed(arg) if r not in extra]
    bad = [r for r in removed if r not in rem.region_names]
    if bad:
        raise KeyError(f"--remover-removed: unknown regions {bad} (known: {rem.region_names})")
    with torch.no_grad():
        rem.gate.copy_(torch.tensor([-rem.gate_max if r in removed else rem.gate_max for r in rem.region_names],
                                    device=rem.gate.device))
    want = torch.tensor([0.0 if r in removed else 1.0 for r in rem.region_names], device=rem.gate.device)
    if not torch.equal(rem.rho(), want):
        raise RuntimeError(f"mask override did not take: rho {rem.rho().tolist()}")
    return [rem.cell_names.index(r) for r in extra]


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
    from r3d.common.robot_freeze import freeze_robot_points, reference_robot_index
    from convert_mimicgen_to_zarr import _layout

    policy, cfg, _ = load_policy(ck)
    cut_cells = []
    if a.remover_removed is not None:
        cut_cells = override_mask(policy, a.remover_removed)
    zattrs = dict(zarr.open(str(ROOT / cfg.task.dataset.zarr_path), "r")["meta"].attrs)
    task, N = zattrs["task"], int(cfg.task.num_points)
    drop = cfg.task.get("drop_point_labels", None)
    drop_links = cfg.task.get("drop_point_links", None)
    if drop_links:   # the label cut keyed on the live link cell (as data/point_link in training)
        if drop:
            raise ValueError("drop_point_links with drop_point_labels")
        drop_links = [RP.CELL_NAMES.index(str(c)) for c in drop_links]
    n_obs = int(cfg.n_obs_steps)
    remover = getattr(policy, "link_remover", None) is not None
    freeze = cfg.task.get("freeze_robot_labels", None)
    if freeze:
        freeze = [int(c) for c in freeze]
        if drop or drop_links or remover:
            raise ValueError("freeze_robot_labels with drop_point_labels / drop_point_links / a link remover")
        ref_m = int(np.load(ROOT / cfg.task.freeze_reference)["m"])

    import h5py
    with h5py.File(R3D_DIR / "data" / "mimicgen" / "core" / f"{task}.hdf5", "r") as f:
        env_args = json.loads(f["data"].attrs["env_args"])
    kw = copy.deepcopy(env_args["env_kwargs"])
    kw["controller_configs"]["control_delta"] = False
    cam, res = zattrs["camera"], int(zattrs["render_res"])
    kw.update(camera_names=[cam], camera_heights=res, camera_widths=res, camera_depths=True,
              camera_segmentations="element", has_offscreen_renderer=True, use_camera_obs=True,
              ignore_done=True)
    window = zattrs.get("init_window")
    if window:   # a narrowed variant (r3d/common/robosuite_tasks.py): reset from exactly the training window
        from r3d.common import robosuite_tasks  # noqa: F401  registers the narrowed envs
    env_name = a.env_name or zattrs.get("eval_env_name", env_args["env_name"])
    check_window = window and env_name == zattrs.get("eval_env_name")   # --env-name: scenes may leave the window
    env = suite.make(env_name, **kw)

    rows = []
    for seed in seeds:
        np.random.seed(seed)
        torch.manual_seed(seed)
        obs = env.reset()
        if check_window:
            robosuite_tasks.check_in_window(env, window)
        inside = None if not window else robosuite_tasks.in_window(
            robosuite_tasks.body_pose(env.sim, window["body"], env.table_offset), window)
        bbox = RP.default_bbox(env.sim)
        if not np.allclose(bbox, zattrs["pcd_crop_bbox"]):
            raise RuntimeError(f"crop box {bbox} != training {zattrs['pcd_crop_bbox']}")
        lut = RP.geom_label_lut(env.sim, env_args["env_name"])
        cb = RP.CloudBuilder(env.sim, cam, res, res, bbox, N, lut)
        cells = RP.LinkCells(env.sim) if remover or drop_links else None
        layout = _layout(env, task)
        if freeze:   # the rollout's first frame, at FPS-M; its first N points are the N cloud
            ref, ref_lab, _ = RP.CloudBuilder(env.sim, cam, res, res, bbox, ref_m, lut).build(obs)
            pc0, lab0, _ = cb.build(obs)
            if not (np.array_equal(ref[:N], pc0) and np.array_equal(ref_lab[:N], lab0)):
                raise RuntimeError(f"scene {seed}: FPS-{ref_m} reference prefix != the {N}-point cloud")
            ref_idx = reference_robot_index(ref_lab, freeze, N)

        def observe(o):
            if remover or drop_links:
                pc, lab, _, cell = cb.build(o, cells=cells)
            else:
                pc, lab, _ = cb.build(o)
            if freeze:
                pc = freeze_robot_points(pc, lab, ref, ref_idx, freeze)
            item = {"agent_pos": torch.from_numpy(RA.state_vector(o["robot0_eef_pos"], o["robot0_eef_quat"],
                                                                  o["robot0_gripper_qpos"]))}
            if drop or drop_links:
                pc, n = cut_points_by_label(pc, cell if drop_links else lab, list(drop_links or drop))
                item["point_cloud_len"] = torch.tensor(n, dtype=torch.int64)
            item["point_cloud"] = torch.from_numpy(pc)
            if remover:
                item["point_link"] = torch.from_numpy(cell.astype(np.int64))
            return item

        cut_count = []

        def cut_extra_cells(item):
            """remove the override's non-region link cells: kept points first, padded with the
            first kept point, valid count in point_cloud_len (as the remover's own output)."""
            cell = item["point_link"].numpy()
            keep = ~np.isin(cell, cut_cells)
            n = int(keep.sum())
            cut_count.append(len(cell) - n)
            order = np.concatenate([np.flatnonzero(keep), np.flatnonzero(~keep)])
            pc = item["point_cloud"].numpy()[order]
            cl = cell[order]
            pc[n:], cl[n:] = pc[0], cl[0]
            item["point_cloud"], item["point_link"] = torch.from_numpy(pc), torch.from_numpy(cl)
            item["point_cloud_len"] = torch.tensor(n, dtype=torch.int64)
            return item

        if cut_cells:
            _observe = observe
            observe = lambda o: cut_extra_cells(_observe(o))
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
        row = {"seed": int(seed), "success": bool(success), "steps": steps, "layout": layout.tolist()}
        if inside is not None:
            row["in_train_window"] = bool(inside)
        if cut_cells:
            row["override_cells_cut_mean"] = float(np.mean(cut_count))
            row["override_cells_cut_max"] = int(np.max(cut_count))
        if remover:
            row["remover_rho"] = dict(zip(policy.link_remover.region_names, policy.link_remover.rho().tolist()))
        rows.append(row)
        print(f"[{os.getpid()}] scene {seed}: {'SUCCESS' if success else 'FAIL'} ({steps} steps)", flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ckpts", nargs="+")
    ap.add_argument("--episodes", type=int, default=200)
    ap.add_argument("--block", type=int, default=1, help="scene seed block: seeds 100000*block + k")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--remover-removed", default=None,
                    help='override the learned mask: comma-separated regions to remove, or "none"')
    ap.add_argument("--env-name", default=None,
                    help="evaluate in another env than the training one (e.g. Threading_D1 for a Threading_D1_Narrow "
                         "policy: out-of-window scenes); output gets _env-<name>, rows carry in_train_window")
    a = ap.parse_args()
    if a.remover_removed is not None:
        removed = parse_removed(a.remover_removed)
        mask_tag = "_mask-" + ("+".join(removed) if removed else "none")
    else:
        mask_tag = ""
    if a.env_name:
        mask_tag += f"_env-{a.env_name}"

    seeds = [100000 * a.block + k for k in range(a.episodes)]
    ctx = mp.get_context("spawn")
    for ck in [str(pathlib.Path(c).resolve()) for c in a.ckpts]:
        ckp = pathlib.Path(ck)
        out = ckp.parent.parent / "eval" / f"{ckp.stem}_{a.episodes}ep_block{a.block}{mask_tag}.json"
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
        extra = {"remover_rho": rows[0]["remover_rho"]} if "remover_rho" in rows[0] else {}
        if a.remover_removed is not None:
            extra["remover_override_removed"] = parse_removed(a.remover_removed)
        with open(out, "w") as fh:
            json.dump({"ckpt": ck, "episodes": a.episodes, "block": a.block, "success_rate": sr, **extra,
                       "minutes": (time.time() - t0) / 60, "scenes": rows}, fh, indent=1)
        print(f"ckpt: {ck}  success {sr:.3f} ({sum(r['success'] for r in rows)}/{len(rows)})  "
              f"{(time.time() - t0) / 60:.1f} min -> {out}", flush=True)


if __name__ == "__main__":
    main()
