#!/usr/bin/env python
"""Is MimicGen Threading a vision task, or memorizable from proprio after the grasp?  (r3d_robosuite, CPU only)

    python scripts/sim/threading_task_check.py threading_d0 threading_d1 [--out <json>]

Per demo, from the stored simulator states (both objects have free joints, so the flattened state carries their
poses; checked against the per-demo xml reset on the first demos):
- tripod pose at t=0 (x, y relative to the table centre, yaw), needle pose at t=0;
- grasp frame t_g: first frame from which the gripper grasps the needle for >= 5 frames (robosuite _check_grasp);
- insertion frame: the first frame where _check_success() holds (the demo then ends);
- EE pose and the needle pose in the EE frame (in-hand offset) at insertion; ring centre and radius.

Reports
1. window counts: how many D1 demos fall in a narrowed tripod window (middle half of x and / or yaw);
2. memorization: spread of the EE position at insertion (world frame) vs the ring radius -- for all demos and for
   the window;
3. what determines the insertion pose: held-out (by demo, 5-fold) RMSE of the EE position at insertion, predicted
   from (a) nothing (mean), (b) proprio at the grasp frame, (c) (b) + tripod pose, (d) (c) + in-hand offset at the
   grasp frame. Gradient-boosted trees per output. If (a) / (b) are already below the ring radius the insertion is
   memorizable; if (c) is far below (b), the policy must see the tripod.
"""

import argparse
import json
import os
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "R3D"))
sys.path.insert(0, str(ROOT / "scripts" / "sim"))


def yaw_of(q_wxyz):
    w, x, y, z = q_wxyz
    return float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


def mat_of(q_wxyz):
    from robosuite.utils.transform_utils import quat2mat
    w, x, y, z = q_wxyz
    return quat2mat(np.array([x, y, z, w]))


def scan_task(task, max_demos=None):
    import h5py
    import robosuite as suite
    import mimicgen  # noqa: F401
    from mimicgen_view_check import reset_to_demo
    f = h5py.File(ROOT / "R3D/data/mimicgen/core" / f"{task}.hdf5", "r")
    env_args = json.loads(f["data"].attrs["env_args"])
    kw = dict(env_args["env_kwargs"])
    kw.update(has_renderer=False, has_offscreen_renderer=False, use_camera_obs=False, ignore_done=True)
    env = suite.make(env_args["env_name"], **kw)
    env.reset()
    nb, tb = env.sim.model.body_name2id(env.needle.root_body), env.sim.model.body_name2id(env.tripod.root_body)
    eef_site = env.robots[0].eef_site_id
    ring = [env.sim.model.geom_name2id(f"tripod_obj_ring_{i}") for i in range(env.tripod.num_ring_geoms)]
    radius = float(env.tripod.ring_size[1])
    table = np.asarray(env.table_offset, dtype=np.float64)
    gripper = env.robots[0].gripper

    def read():
        eef_p = env.sim.data.site_xpos[eef_site].copy()
        eef_R = env.sim.data.site_xmat[eef_site].reshape(3, 3).copy()
        n_p, n_q = env.sim.data.body_xpos[nb].copy(), env.sim.data.body_xquat[nb].copy()
        t_p, t_q = env.sim.data.body_xpos[tb].copy(), env.sim.data.body_xquat[tb].copy()
        return eef_p, eef_R, n_p, n_q, t_p, t_q

    demos = sorted(f["data"].keys(), key=lambda k: int(k.split("_")[1]))[:max_demos]
    rows, n_xml_checked = [], 0
    for di, dk in enumerate(demos):
        g = f[f"data/{dk}"]
        S = g["states"][:]
        if di < 3:   # the flattened state alone must reproduce the per-demo xml reset
            reset_to_demo(env, g.attrs["model_file"], S[0])
            ref = read()
            env.sim.set_state_from_flattened(S[0]); env.sim.forward()
            if not all(np.allclose(a, b, atol=1e-9) for a, b in zip(ref, read())):
                raise RuntimeError(f"{task} {dk}: flattened state differs from the xml reset")
            n_xml_checked += 1
        grasp, t_g, t_s, run = [], None, None, 0
        for t in range(len(S)):
            env.sim.set_state_from_flattened(S[t]); env.sim.forward()
            gr = bool(env._check_grasp(gripper=gripper, object_geoms=env.needle))
            run = run + 1 if gr else 0
            if t_g is None and run >= 5:
                t_g = t - 4
            if t_s is None and env._check_success():
                t_s = t
                break
        if t_s is None and t_g is not None:   # states are pre-action: success can come with the final action
            t_s, late = len(S) - 1, True
        else:
            late = False
        if t_g is None or t_s is None:
            rows.append({"demo": dk, "ok": False, "t_g": t_g, "t_s": t_s, "T": len(S)})
            continue
        env.sim.set_state_from_flattened(S[0]); env.sim.forward()
        _, _, n0p, n0q, t0p, t0q = read()
        feats = {}
        for name, t in (("g", t_g), ("s", t_s)):
            env.sim.set_state_from_flattened(S[t]); env.sim.forward()
            ep, eR, np_, nq, tp, tq = read()
            off = eR.T @ (np_ - ep)                           # needle position in the EE frame
            offR = eR.T @ mat_of(nq)
            feats[name] = dict(eef=ep, eef_R=eR, off=off, off_R=offR, grip=env.sim.data.qpos[env.robots[0]._ref_gripper_joint_pos_indexes].copy())
        env.sim.set_state_from_flattened(S[t_s]); env.sim.forward()
        ring_c = np.mean([env.sim.data.geom_xpos[i] for i in ring], 0)
        rows.append({
            "demo": dk, "ok": True, "T": len(S), "t_g": t_g, "t_s": t_s, "success_after_last_state": late,
            "tripod": [*(t0p[:2] - table[:2]), yaw_of(t0q)], "needle": [*(n0p[:2] - table[:2]), yaw_of(n0q)],
            "eef_g": feats["g"]["eef"].tolist(), "R_g": feats["g"]["eef_R"].ravel().tolist(),
            "grip_g": feats["g"]["grip"].tolist(), "off_g": feats["g"]["off"].tolist(),
            "offR_g": feats["g"]["off_R"].ravel().tolist(),
            "eef_s": feats["s"]["eef"].tolist(), "R_s": feats["s"]["eef_R"].ravel().tolist(),
            "off_s": feats["s"]["off"].tolist(), "ring": ring_c.tolist(),
        })
    return rows, radius, n_xml_checked


def cv_rmse(X, Y, k=5, seed=0):
    from sklearn.ensemble import HistGradientBoostingRegressor
    n = len(Y)
    if X is None:
        X = np.zeros((n, 1))
    idx = np.random.default_rng(seed).permutation(n)
    err = np.zeros_like(Y)
    for fold in range(k):
        te = idx[fold::k]
        tr = np.setdiff1d(idx, te)
        for j in range(Y.shape[1]):
            if X.shape[1] == 1 and not X.any():
                pred = np.full(len(te), Y[tr, j].mean())
            else:
                m = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, random_state=0).fit(X[tr], Y[tr, j])
                pred = m.predict(X[te])
            err[te, j] = pred - Y[te, j]
    return float(np.sqrt((err ** 2).sum(1).mean()))


def report(task, rows, radius, window=None):
    ok = [r for r in rows if r["ok"]]
    tri = np.array([r["tripod"] for r in ok])
    sel = np.ones(len(ok), bool) if window is None else (
        (tri[:, 0] >= window["x"][0]) & (tri[:, 0] <= window["x"][1])
        & (tri[:, 2] >= window["yaw"][0]) & (tri[:, 2] <= window["yaw"][1]))
    sub = [r for r, s in zip(ok, sel) if s]
    if len(sub) < 2:
        return {"task": task, "window": window, "n": len(sub)}
    E = np.array([r["eef_s"] for r in sub])
    Y = E
    P = np.array([r["eef_g"] + r["R_g"][:6] + r["grip_g"] for r in sub])       # proprio at the grasp
    T = np.array([[r["tripod"][0], r["tripod"][1], np.cos(r["tripod"][2]), np.sin(r["tripod"][2])] for r in sub])
    H = np.array([r["off_g"] + r["offR_g"][:6] for r in sub])
    out = {
        "task": task, "window": window, "n": len(sub), "n_failed_parse": len(rows) - len(ok),
        "n_success_after_last_state": int(sum(r["success_after_last_state"] for r in sub)),
        "ring_radius_mm": radius * 1000,
        "eef_at_insertion_std_mm": (E.std(0) * 1000).round(2).tolist(),
        "eef_at_insertion_max_dev_mm": float(np.linalg.norm(E - E.mean(0), axis=1).max() * 1000),
        "inhand_offset_std_mm_at_grasp": (np.array([r["off_g"] for r in sub]).std(0) * 1000).round(2).tolist(),
        "inhand_offset_change_grasp_to_insertion_mm": float(np.median(np.linalg.norm(
            np.array([r["off_s"] for r in sub]) - np.array([r["off_g"] for r in sub]), axis=1)) * 1000),
        "tripod_ranges": {"x": [float(tri[sel, 0].min()), float(tri[sel, 0].max())],
                          "y": [float(tri[sel, 1].min()), float(tri[sel, 1].max())],
                          "yaw_deg": [float(np.degrees(tri[sel, 2].min())), float(np.degrees(tri[sel, 2].max()))]},
        "frames_post_grasp_median": float(np.median([r["t_s"] - r["t_g"] for r in sub])),
        "frames_total_median": float(np.median([r["t_s"] for r in sub])),
    }
    if len(sub) >= 50:
        out["rmse_insertion_eef_mm"] = {
            "a_mean_only": cv_rmse(None, Y) * 1000,
            "b_proprio_at_grasp": cv_rmse(P, Y) * 1000,
            "c_+tripod_pose": cv_rmse(np.hstack([P, T]), Y) * 1000,
            "d_+inhand_offset": cv_rmse(np.hstack([P, T, H]), Y) * 1000,
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tasks", nargs="+")
    ap.add_argument("--max-demos", type=int, default=None)
    ap.add_argument("--out", default=str(ROOT / "R3D/data/outputs/sim/mimicgen/threading_check/threading_task_check.json"))
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    res = {}
    for task in a.tasks:
        rows, radius, nx = scan_task(task, a.max_demos)
        print(f"{task}: {len(rows)} demos scanned, {sum(r['ok'] for r in rows)} parsed, xml check on {nx}", flush=True)
        reps = [report(task, rows, radius)]
        if task.endswith("d1"):
            tri = np.array([r["tripod"] for r in rows if r["ok"]])      # the sampled ranges, as measured
            x, yaw = (tri[:, 0].min(), tri[:, 0].max()), (tri[:, 2].min(), tri[:, 2].max())
            mid = lambda lo, hi: (lo + (hi - lo) / 4, hi - (hi - lo) / 4)
            for name, w in (("x+yaw middle half", {"x": mid(*x), "yaw": mid(*yaw)}),
                            ("x middle half", {"x": mid(*x), "yaw": yaw}),
                            ("yaw middle half", {"x": x, "yaw": mid(*yaw)})):
                r = report(task, rows, radius, w)
                r["window_name"] = name
                reps.append(r)
        res[task] = {"reports": reps, "rows": rows}
        for r in reps:
            print(json.dumps({k: v for k, v in r.items()}, indent=1), flush=True)
    json.dump(res, open(a.out, "w"))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
