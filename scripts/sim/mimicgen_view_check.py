#!/usr/bin/env python
"""View check for a MimicGen task before any conversion / training.  Run in r3d_robosuite.

    python scripts/sim/mimicgen_view_check.py --task square_d2 --episodes 0 250 500 750 999

Re-renders the chosen demos from their stored simulator states (no planning, no
policy) with `r3d.common.robosuite_pcd` -- the same code the conversion and the
evaluation will use -- and reports what the head camera sees:

- per class: pixels in the image / in the crop box, min over frames (is the nut /
  peg ever lost?), stored points at the 8192 / 4096 / 2048 / 1024 prefixes
- token allocation (`token_allocation.py`, random-start FPS as in training) at
  64 / 128 tokens on the 4096 and 1024 prefixes: centres, slot share, coverage
- a check that the unprojection is right: the nut points' centroid vs the nut body

and writes a preview zarr that `viz_clouds.py` reads unchanged:
    R3D/data/<task>-<camera>_p<N>-preview<k>.zarr
Report: R3D/data/outputs/sim_view/<task>/report.json
"""

import argparse
import json
import os
import sys
import time

os.environ.setdefault("MUJOCO_GL", "egl")
import h5py
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
R3D_DIR = os.path.join(ROOT, "R3D")
sys.path.insert(0, R3D_DIR)


def make_env(env_args, camera, res):
    import robosuite as suite
    import mimicgen  # noqa: F401  registers the D0/D1/D2 environments
    kw = dict(env_args["env_kwargs"])
    kw.update(camera_names=[camera], camera_heights=res, camera_widths=res,
              camera_depths=True, camera_segmentations="element", has_offscreen_renderer=True)
    return suite.make(env_args["env_name"], **kw)


def reset_to_demo(env, model_xml, state):
    env.reset()
    env.reset_from_xml_string(env.edit_model_xml(model_xml))
    env.sim.reset()
    env.sim.set_state_from_flattened(state)
    env.sim.forward()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", default="square_d2")
    ap.add_argument("--episodes", type=int, nargs="+", default=[0, 250, 500, 750, 999])
    ap.add_argument("--camera", default="agentview")
    ap.add_argument("--res", type=int, default=512)
    ap.add_argument("--num-points", type=int, default=8192, help="stored points (prefixes give smaller budgets)")
    a = ap.parse_args()

    from r3d.common import robosuite_pcd as RP
    from r3d.common.token_allocation import box_normalize, token_allocation

    f = h5py.File(os.path.join(R3D_DIR, "data", "mimicgen", "core", f"{a.task}.hdf5"), "r")
    env_args = json.loads(f["data"].attrs["env_args"])
    env = make_env(env_args, a.camera, a.res)

    pcs, labs, states, ends, rows = [], [], [], [], []
    px_img, px_box, nvalid, nut_err = [], [], [], []
    bbox = None
    t0 = time.time()
    for ep in a.episodes:
        g = f[f"data/demo_{ep}"]
        S = g["states"][:]
        reset_to_demo(env, g.attrs["model_file"], S[0])
        if bbox is None:
            bbox = RP.default_bbox(env.sim)
        lut = RP.geom_label_lut(env.sim, env_args["env_name"])
        cb = RP.CloudBuilder(env.sim, a.camera, a.res, a.res, bbox, a.num_points, lut)
        nut_bid = env.sim.model.body_name2id("SquareNut_main")
        for t in range(len(S)):
            env.sim.set_state_from_flattened(S[t])
            env.sim.forward()
            obs = env._get_observations(force_update=True)
            pc, lab, n = cb.build(obs)
            pi, pb = cb.class_pixels(obs)
            px_img.append(pi), px_box.append(pb), nvalid.append(n)
            if t == 0:   # unprojection check: nut points vs nut body (xy; the body origin sits above the base)
                nut = pc[lab == RP.OBJECT, :3]
                nut_err.append(float(np.linalg.norm(nut[:, :2].mean(0) - env.sim.data.body_xpos[nut_bid][:2])))
            pcs.append(pc), labs.append(lab)
            states.append(np.concatenate([g["obs/robot0_eef_pos"][t], g["obs/robot0_eef_quat"][t],
                                          g["obs/robot0_gripper_qpos"][t]]).astype(np.float32))
        ends.append(len(pcs))
        print(f"demo {ep}: {len(S)} frames, {time.time() - t0:.0f}s", flush=True)

    pcs, labs = np.stack(pcs), np.stack(labs)
    px_img, px_box, nvalid = np.stack(px_img), np.stack(px_box), np.array(nvalid)
    per_frame_s = (time.time() - t0) / len(pcs)

    # ---- zarr preview (viz_clouds.py layout)
    import zarr
    out = os.path.join(R3D_DIR, "data", f"{a.task}-{a.camera}_p{a.num_points}-preview{len(a.episodes)}.zarr")
    root = zarr.open(out, "w")
    root.create_dataset("data/point_cloud", data=pcs, chunks=(1, a.num_points, 6))
    root.create_dataset("data/point_label", data=labs, chunks=(1, a.num_points))
    root.create_dataset("data/state", data=np.stack(states))
    root.create_dataset("meta/episode_ends", data=np.array(ends, dtype=np.int64))
    root["meta"].attrs.update({
        "source": "mimicgen", "task": a.task, "env_name": env_args["env_name"], "camera": a.camera,
        "render_res": a.res, "pcd_crop_bbox": bbox, "pcd_down_sample_num": a.num_points,
        "point_label_classes": {str(k): v for k, v in RP.CLASS_NAMES.items()},
        "demos": a.episodes, "preview_only": "state = eef pos + quat(xyzw) + gripper qpos; no actions"})

    # ---- report
    names = RP.CLASS_NAMES
    show = [RP.TABLE, RP.ROBOT_BODY, RP.RIGHT_ARM, RP.RIGHT_GRIPPER, RP.OBJECT, RP.TARGET, RP.OTHER, RP.PAD]
    rep = {"task": a.task, "camera": a.camera, "res": a.res, "bbox": bbox, "frames": int(len(pcs)),
           "seconds_per_frame": per_frame_s, "nut_centroid_err_m": nut_err,
           "points_in_box_before_fps": {"min": int(nvalid.min()), "p10": float(np.percentile(nvalid, 10)),
                                        "median": float(np.median(nvalid))},
           "pixels": {}, "points": {}, "tokens": {}}
    for c in show:
        rep["pixels"][names[c]] = {"img_median": float(np.median(px_img[:, c])), "box_median": float(np.median(px_box[:, c])),
                                   "box_min": int(px_box[:, c].min())}
    for N in (8192, 4096, 2048, 1024):
        if N > a.num_points:
            continue
        cnt = np.stack([np.bincount(l[:N], minlength=RP.N_CLASSES) for l in labs])
        rep["points"][N] = {names[c]: {"median": float(np.median(cnt[:, c])), "min": int(cnt[:, c].min())} for c in show}
    sub = np.arange(0, len(pcs), 3)
    for N in (4096, 1024):
        for G in (128, 64):
            xyz = box_normalize(pcs[sub, :N, :3], bbox)
            r = token_allocation(xyz, labs[sub, :N].astype(np.int64), RP.N_CLASSES, num_group=G)
            rep["tokens"][f"{N}/{G}"] = {names[c]: {"centres": float(r["centres"][:, c].mean()),
                                                    "coverage": float(np.nanmean(r["coverage"][:, c]))
                                                    if np.isfinite(r["coverage"][:, c]).any() else None}
                                         for c in show}
    od = os.path.join(R3D_DIR, "data", "outputs", "sim_view", a.task)
    os.makedirs(od, exist_ok=True)
    with open(os.path.join(od, "report.json"), "w") as fh:
        json.dump(rep, fh, indent=1)

    print(f"\n{len(pcs)} frames, {per_frame_s * 1000:.0f} ms/frame; nut centroid error (m): {np.round(nut_err, 4)}")
    print("points in box before FPS:", rep["points_in_box_before_fps"])
    print(f"\n{'class':<14}{'img px':>9}{'box px':>9}{'box min':>9}" + "".join(f"{'pts@' + str(N):>10}" for N in rep["points"]))
    for c in show:
        p = rep["pixels"][names[c]]
        print(f"{names[c]:<14}{p['img_median']:>9.0f}{p['box_median']:>9.0f}{p['box_min']:>9}"
              + "".join(f"{rep['points'][N][names[c]]['median']:>10.0f}" for N in rep["points"]))
    print("\ntoken centres (coverage) per class")
    print(f"{'class':<14}" + "".join(f"{k:>16}" for k in rep["tokens"]))
    for c in show:
        print(f"{names[c]:<14}" + "".join(
            f"{v[names[c]]['centres']:>8.1f} ({v[names[c]]['coverage'] if v[names[c]]['coverage'] is None else round(v[names[c]]['coverage'], 2)})".rjust(16)
            for v in rep["tokens"].values()))
    print("wrote", out, "and", os.path.join(od, "report.json"))


if __name__ == "__main__":
    main()
