#!/usr/bin/env python
"""Square D2: what the eval-only `--keep-table` cloud looks like, before evaluating with it. Run in r3d_robosuite:

    python scripts/sim/table_cloud_check.py [--scenes 10] [--keep-table 0.02]

On the first eval scenes (seed block 1, the reset frame), the training cloud (box + TABLE/BACKGROUND
filter) against the table cloud (floor `keep-table` m below the table top, TABLE kept), both at 4096
points: per class share of points and of 64 encoder-style FPS centres (random start, 20 draws), the
normalized table z (the training normalizer = the training box; DP3 clamps to [-1, 1]), and a PNG of
scene 0 (top + side view, class colours, token centres) for a look before any eval.
Writes R3D/data/outputs/sim/mimicgen/table_eval/table_cloud_check.{json,png}.
"""

import argparse
import copy
import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "R3D"))
OUT = ROOT / "R3D/data/outputs/sim/mimicgen/table_eval"
ZARR = ROOT / "R3D/data/square_d2-agentview_p4096-200.zarr"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", type=int, default=10)
    ap.add_argument("--keep-table", type=float, default=0.02)
    ap.add_argument("--draws", type=int, default=20)
    a = ap.parse_args()
    import h5py
    import torch
    import zarr
    import robosuite as suite
    import mimicgen  # noqa: F401
    from pytorch3d.ops import sample_farthest_points
    from r3d.common import robosuite_pcd as RP

    OUT.mkdir(parents=True, exist_ok=True)
    z = dict(zarr.open(str(ZARR), "r")["meta"].attrs)
    with h5py.File(ROOT / "R3D/data/mimicgen/core" / f"{z['task']}.hdf5", "r") as f:
        env_args = json.loads(f["data"].attrs["env_args"])
    kw = copy.deepcopy(env_args["env_kwargs"])
    kw["controller_configs"]["control_delta"] = False
    cam, res = z["camera"], int(z["render_res"])
    kw.update(camera_names=[cam], camera_heights=res, camera_widths=res, camera_depths=True,
              camera_segmentations="element", has_offscreen_renderer=True, use_camera_obs=True, ignore_done=True)
    env = suite.make(z.get("eval_env_name", env_args["env_name"]), **kw)
    box = np.asarray(z["pcd_crop_bbox"])
    groups = {"table": [RP.TABLE], "arm": [RP.RIGHT_ARM], "gripper": [RP.RIGHT_GRIPPER],
              "nut": [RP.OBJECT], "peg": [RP.TARGET], "other": [RP.ROBOT_BODY, RP.OTHER, RP.PAD]}
    rows, keep_scene0 = [], None
    for k in range(a.scenes):
        seed = 100000 + k
        np.random.seed(seed); torch.manual_seed(seed)
        obs = env.reset()
        lut = RP.geom_label_lut(env.sim, env_args["env_name"])
        bbox = RP.default_bbox(env.sim)
        tbox = [[bbox[0][0], bbox[0][1], bbox[0][2] - 0.003 - a.keep_table], list(bbox[1])]
        row = {"seed": seed}
        for name, bb, drop in (("train", bbox, (RP.TABLE, RP.BACKGROUND)), ("table", tbox, (RP.BACKGROUND,))):
            pc, lab, n = RP.CloudBuilder(env.sim, cam, res, res, bb, 4096, lut, drop_classes=drop).build(obs)
            x = torch.as_tensor(pc[None, :, :3], device="cuda")
            tok = np.zeros(len(groups))
            for _ in range(a.draws):
                _, idx = sample_farthest_points(x, K=64, random_start_point=True)
                tl = lab[idx[0].cpu().numpy()]
                tok += [np.isin(tl, c).sum() for c in groups.values()]
            row[name] = {"valid_points": int(n),
                         "points": {g: int(np.isin(lab, c).sum()) for g, c in groups.items()},
                         "tokens": {g: float(t / a.draws) for g, t in zip(groups, tok)}}
            if name == "table":
                tz = pc[lab == RP.TABLE, 2]
                row["table_z_norm"] = (float(((tz - box[0][2]) / (box[1][2] - box[0][2]) * 2 - 1).min())
                                       if len(tz) else None)
            if k == 0:
                keep_scene0 = keep_scene0 or {}
                keep_scene0[name] = (pc, lab, idx[0].cpu().numpy())
        rows.append(row)
        print(seed, {n: (row[n]["points"]["table"], round(row[n]["tokens"]["table"], 1),
                         round(row[n]["tokens"]["nut"], 1), round(row[n]["tokens"]["peg"], 1)) for n in ("train", "table")},
              flush=True)

    summ = {}
    for name in ("train", "table"):
        summ[name] = {"valid_points": float(np.mean([r[name]["valid_points"] for r in rows])),
                      "points": {g: float(np.mean([r[name]["points"][g] for r in rows])) for g in groups},
                      "tokens": {g: float(np.mean([r[name]["tokens"][g] for r in rows])) for g in groups}}
    summ["table_z_norm_min"] = float(min(r["table_z_norm"] for r in rows if r["table_z_norm"] is not None))
    json.dump({"keep_table_m": a.keep_table, "summary": summ, "scenes": rows}, open(OUT / "table_cloud_check.json", "w"), indent=1)
    print(json.dumps(summ, indent=1))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cols = {RP.TABLE: "#c8b48c", RP.RIGHT_ARM: "#4c72b0", RP.RIGHT_GRIPPER: "#55a868", RP.OBJECT: "#c44e52",
            RP.TARGET: "#8172b2", RP.ROBOT_BODY: "#999999", RP.OTHER: "#999999", RP.PAD: "#000000"}
    fig, ax = plt.subplots(2, 2, figsize=(12, 9))
    for j, name in enumerate(("train", "table")):
        pc, lab, idx = keep_scene0[name]
        c = [cols.get(int(l), "#999999") for l in lab]
        for i, (u, v, t) in enumerate(((0, 1, "top (x, y)"), (0, 2, "side (x, z)"))):
            ax[i, j].scatter(pc[:, u], pc[:, v], s=1, c=c)
            ax[i, j].scatter(pc[idx, u], pc[idx, v], s=40, facecolors="none", edgecolors="k", linewidths=1)
            ax[i, j].set_title(f"{name} cloud, {t}; circles = 64 FPS centres"); ax[i, j].set_aspect("equal")
    fig.tight_layout(); fig.savefig(OUT / "table_cloud_check.png", dpi=110)
    print("wrote", OUT / "table_cloud_check.png")


if __name__ == "__main__":
    main()
