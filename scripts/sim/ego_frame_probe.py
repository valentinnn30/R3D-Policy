#!/usr/bin/env python
"""Is the action easier to predict in the end-effector (ego) frame than in the world frame?

Privileged low-dim probe on the Square D2 training demos (simulator object poses, no point
clouds, no policy): the SAME information is given to a generic regressor in three frames, and
the held-out error (5-fold by demo) of the action chunk is compared in mm.

    world : eef pose, nut pose, peg pose (world)            -> absolute goal position
    ego   : nut pose, peg pose in the eef frame (SE(3))     -> goal position in the eef frame
    yaw   : the same in the gravity-aligned eef frame       -> goal position in that frame
            (origin at the eef, z up, x along the eef heading)

The error norm is invariant to the frame, so the three numbers are comparable. Two generic
learners without built-in invariance: gradient-boosted trees and k-nearest neighbours.

    python scripts/sim/ego_frame_probe.py            # r3d_robosuite env (h5py, sklearn), CPU

Output: R3D/data/outputs/sim/mimicgen/ego_frame/probe.json
"""

import argparse
import json
import pathlib
import sys

import h5py
import numpy as np
import zarr
from scipy.spatial.transform import Rotation

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "R3D"))
from r3d.common import ego_frame as EF  # noqa: E402
from r3d.common import robosuite_actions as RA  # noqa: E402


def load(zarr_path, h5_path):
    z = zarr.open(str(zarr_path), "r")
    demos = list(z["meta"].attrs["demos"])
    ends = z["meta/episode_ends"][:]
    state, action = z["data/state"][:], z["data/action"][:]
    layout = z["meta/init_layout"][:]
    starts = np.concatenate([[0], ends[:-1]])
    rows = []
    with h5py.File(h5_path, "r") as f:
        for e, d in enumerate(demos):
            obj = f[f"data/demo_{d}/obs/object"][:]
            n = ends[e] - starts[e]
            if len(obj) != n:
                raise RuntimeError(f"demo {d}: {len(obj)} hdf5 frames vs {n} in the zarr")
            if not np.allclose(f[f"data/demo_{d}/obs/robot0_eef_pos"][:], state[starts[e]:ends[e], :3], atol=1e-5):
                raise RuntimeError(f"demo {d}: eef position differs between hdf5 and zarr")
            rows.append(dict(ep=e, state=state[starts[e]:ends[e]], action=action[starts[e]:ends[e]],
                             nut_pos=obj[:, :3], nut_quat=obj[:, 3:7],
                             peg_pos=np.repeat(layout[e, 7:10][None], n, 0),
                             peg_quat=np.repeat(layout[e, 10:14][None], n, 0)))
    return rows


def features(rows, steps):
    """-> dict frame -> (X, Y), plus groups (episode) and phase info. Y = goal positions of the
    listed future steps, in the frame."""
    out = {k: ([], []) for k in ("world", "ego", "yaw")}
    groups, d_nut, grip = [], [], []
    for r in rows:
        n = len(r["state"])
        pos, Ree = r["state"][:, :3].astype(np.float64), RA.rot6d_to_mat(r["state"][:, 3:9])
        Rn = Rotation.from_quat(r["nut_quat"]).as_matrix()
        Rp = Rotation.from_quat(r["peg_quat"]).as_matrix()
        g = r["state"][:, 9:11]
        idx = np.clip(np.arange(n)[:, None] + np.asarray(steps)[None] - 1, 0, n - 1)   # step k -> action[t + k - 1]
        goal = r["action"][:, :3].astype(np.float64)[idx]                                 # (n, K, 3) world
        for frame in ("world", "ego", "yaw"):
            if frame == "world":
                X = np.concatenate([pos, RA.mat_to_rot6d(Ree), g, r["nut_pos"], RA.mat_to_rot6d(Rn),
                                    r["peg_pos"], RA.mat_to_rot6d(Rp)], 1)
                Y = goal.reshape(n, -1)
            else:
                Rf = Ree if frame == "ego" else EF.yaw_frame(Ree)
                to = lambda p: np.einsum("nji,nj->ni", Rf, p - pos)             # R^T (p - t)
                rel = lambda R: RA.mat_to_rot6d(np.einsum("nji,njk->nik", Rf, R))
                X = np.concatenate([to(r["nut_pos"]), rel(Rn), to(r["peg_pos"]), rel(Rp), g], 1)
                if frame == "yaw":   # the eef's tilt and height are not removed by a yaw frame
                    X = np.concatenate([X, rel(Ree), pos[:, 2:3]], 1)
                Y = np.einsum("nji,nkj->nki", Rf, goal - pos[:, None]).reshape(n, -1)
            out[frame][0].append(X)
            out[frame][1].append(Y)
        groups.append(np.full(n, r["ep"]))
        d_nut.append(np.linalg.norm(r["nut_pos"] - pos, axis=1))
        grip.append(g[:, 0] - g[:, 1])
    cat = np.concatenate
    return {k: (cat(v[0]), cat(v[1])) for k, v in out.items()}, cat(groups), cat(d_nut), cat(grip)


def cv_error(X, Y, groups, model, folds=5, seed=0):
    """held-out per-sample error (mm): mean over the chunk's steps of the 3D position error."""
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.preprocessing import StandardScaler
    eps = np.unique(groups)
    rng = np.random.default_rng(seed)
    fold_of = dict(zip(rng.permutation(eps), np.arange(len(eps)) % folds))
    fold = np.array([fold_of[g] for g in groups])
    pred = np.zeros_like(Y)
    for k in range(folds):
        tr, te = fold != k, fold == k
        if model == "knn":
            sc = StandardScaler().fit(X[tr])
            pred[te] = KNeighborsRegressor(n_neighbors=5, weights="distance").fit(sc.transform(X[tr]), Y[tr]).predict(
                sc.transform(X[te]))
        else:
            for j in range(Y.shape[1]):
                pred[te, j] = HistGradientBoostingRegressor(max_iter=200, learning_rate=0.1, random_state=0).fit(
                    X[tr], Y[tr, j]).predict(X[te])
    err = np.linalg.norm((pred - Y).reshape(len(Y), -1, 3), axis=-1)     # (n, K)
    return 1000.0 * err.mean(1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--zarr", default=str(ROOT / "R3D/data/square_d2-agentview_p4096-200.zarr"))
    ap.add_argument("--hdf5", default=str(ROOT / "R3D/data/mimicgen/core/square_d2.hdf5"))
    ap.add_argument("--steps", type=int, nargs="+", default=[1, 4, 8])
    ap.add_argument("--out", default=str(ROOT / "R3D/data/outputs/sim/mimicgen/ego_frame/probe.json"))
    a = ap.parse_args()

    rows = load(a.zarr, a.hdf5)
    feats, groups, d_nut, grip = features(rows, a.steps)
    near = d_nut < 0.10
    res = {"steps": a.steps, "frames": len(groups), "near_nut_frames": int(near.sum())}
    print(f"{len(rows)} demos, {len(groups)} frames; within 10 cm of the nut: {near.mean():.0%}")
    print(f"{'learner':8s} {'frame':6s} {'all':>8s} {'near nut':>9s} {'far':>8s}   (held-out mm, mean over steps {a.steps})")
    for model in ("gbt", "knn"):
        for frame in ("world", "ego", "yaw"):
            e = cv_error(*feats[frame], groups, model)
            res[f"{model}_{frame}"] = dict(all=float(e.mean()), near=float(e[near].mean()), far=float(e[~near].mean()),
                                           median=float(np.median(e)))
            print(f"{model:8s} {frame:6s} {e.mean():8.1f} {e[near].mean():9.1f} {e[~near].mean():8.1f}")
    out = pathlib.Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1))
    print("wrote", out)


if __name__ == "__main__":
    main()
