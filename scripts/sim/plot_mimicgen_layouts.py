#!/usr/bin/env python
"""Training vs eval scene layouts for a MimicGen zarr: is the data spread over the task's
reset distribution, and how far is each eval scene from its nearest demo?  Run in r3d_robosuite.

    python scripts/sim/plot_mimicgen_layouts.py R3D/data/square_d2-agentview_p4096-200.zarr --episodes 200

Training layouts come from the zarr (`meta/init_layout`, written by the converter); eval layouts
are rebuilt with the evaluation's own seeding (np.random.seed(100000 * block + k); env.reset()).
Output: R3D/data/outputs/sim_view/<task>/layouts.png + layouts.json
"""
import argparse, copy, json, os, sys
os.environ.setdefault("MUJOCO_GL", "egl")
import numpy as np
from scipy.spatial.transform import Rotation
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "R3D")); sys.path.insert(0, os.path.dirname(__file__))


def yaw_deg(q_xyzw):
    return Rotation.from_quat(q_xyzw).as_euler("xyz", degrees=True)[:, 2]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("zarr"); ap.add_argument("--episodes", type=int, default=200)
    ap.add_argument("--block", type=int, default=1); a = ap.parse_args()
    import zarr, h5py, robosuite as suite, mimicgen  # noqa: F401
    from convert_mimicgen_to_zarr import _layout, hdf5_path
    z = zarr.open(os.path.join(ROOT, a.zarr), "r"); task = z["meta"].attrs["task"]
    tr = z["meta/init_layout"][:]
    with h5py.File(hdf5_path(task), "r") as f:
        ea = json.loads(f["data"].attrs["env_args"])
    kw = copy.deepcopy(ea["env_kwargs"]); kw.update(has_offscreen_renderer=False, use_camera_obs=False)
    env = suite.make(ea["env_name"], **kw)
    ev = []
    for k in range(a.episodes):
        np.random.seed(100000 * a.block + k); env.reset(); ev.append(_layout(env, task))
    ev = np.array(ev)
    # nut yaw has 90 deg symmetry only if the nut were symmetric -- the handle breaks it, so use the full angle
    def feats(L):  # nut xy, nut yaw, peg xy
        return np.c_[L[:, 0:2], yaw_deg(L[:, 3:7]), L[:, 7:9]]
    Ft, Fe = feats(tr), feats(ev)
    # distance to nearest demo: positions in cm, yaw difference wrapped, 10 deg ~ 1 cm weight is arbitrary -> report separately
    dpos = np.sqrt(((Fe[:, None, [0, 1, 3, 4]] - Ft[None, :, [0, 1, 3, 4]]) ** 2).sum(-1)) * 100
    dyaw = np.abs((Fe[:, None, 2] - Ft[None, :, 2] + 180) % 360 - 180)
    nn = dpos.argmin(1); d_nn, yaw_nn = dpos.min(1), dyaw[np.arange(len(Fe)), nn]
    nut_only = (np.sqrt(((Fe[:, None, :2] - Ft[None, :, :2]) ** 2).sum(-1)) * 100).min(1)
    peg_only = (np.sqrt(((Fe[:, None, 3:5] - Ft[None, :, 3:5]) ** 2).sum(-1)) * 100).min(1)
    rep = {"train_n": len(Ft), "eval_n": len(Fe),
           "ranges_train": {k: [float(Ft[:, i].min()), float(Ft[:, i].max())] for i, k in enumerate(["nut_x", "nut_y", "nut_yaw", "peg_x", "peg_y"])},
           "ranges_eval": {k: [float(Fe[:, i].min()), float(Fe[:, i].max())] for i, k in enumerate(["nut_x", "nut_y", "nut_yaw", "peg_x", "peg_y"])},
           "nearest_demo_cm_joint": np.percentile(d_nn, [10, 50, 90]).tolist(), "nearest_demo_yaw_deg": np.percentile(yaw_nn, [10, 50, 90]).tolist(),
           "nearest_demo_cm_nut_only": np.percentile(nut_only, [10, 50, 90]).tolist(), "nearest_demo_cm_peg_only": np.percentile(peg_only, [10, 50, 90]).tolist(),
           "eval_scenes_identical_to_a_demo": int((d_nn < 0.1).sum())}
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    ink, bg = "#0b0b0b", "#fcfcfb"; c_tr, c_ev = "#2a78d6", "#eb6834"
    fig, ax = plt.subplots(1, 4, figsize=(18, 4.4), facecolor=bg)
    for x in ax:
        x.set_facecolor(bg); [x.spines[s].set_visible(False) for s in ("top", "right")]; x.grid(color="#e4e3df", lw=0.6); x.set_axisbelow(True)
    for i, (xi, yi, t) in enumerate(((0, 1, "Nut position (m)"), (3, 4, "Peg position (m)"))):
        ax[i].scatter(Ft[:, xi], Ft[:, yi], s=22, color=c_tr, label=f"training demos ({len(Ft)})", edgecolor=bg, lw=0.5)
        ax[i].scatter(Fe[:, xi], Fe[:, yi], s=26, marker="x", color=c_ev, label=f"eval scenes ({len(Fe)})", lw=1.1)
        ax[i].set(title=t, xlabel="x", ylabel="y"); ax[i].set_aspect("equal"); ax[i].legend(frameon=False, fontsize=8, loc="upper left")
    b = np.linspace(-180, 180, 25)
    ax[2].hist(Ft[:, 2], bins=b, color=c_tr, alpha=0.85, label="training", rwidth=0.9)
    ax[2].hist(Fe[:, 2], bins=b, histtype="step", color=c_ev, lw=2, label="eval")
    ax[2].set(title="Nut yaw (deg)", xlabel="yaw", ylabel="scenes"); ax[2].legend(frameon=False, fontsize=8)
    ax[3].hist(d_nn, bins=20, color=c_tr, rwidth=0.9)
    ax[3].set(title="Eval scene -> nearest demo (nut + peg position)", xlabel="cm (joint, 4-D)", ylabel="eval scenes")
    ax[3].axvline(np.median(d_nn), color=ink, lw=1.5); ax[3].text(np.median(d_nn), ax[3].get_ylim()[1] * 0.92, f" median {np.median(d_nn):.1f} cm", fontsize=8)
    od = os.path.join(ROOT, "R3D", "data", "outputs", "sim_view", task); os.makedirs(od, exist_ok=True)
    fig.savefig(os.path.join(od, "layouts.png"), dpi=140, bbox_inches="tight", facecolor=bg)
    json.dump(rep, open(os.path.join(od, "layouts.json"), "w"), indent=1); print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
