#!/usr/bin/env python
"""Interactive look at the training clouds of a sim zarr: one self-contained HTML
per episode, opened in any browser (no server).  Run in r3d_sim.

    python scripts/sim/viz_clouds.py R3D/data/place_empty_cup_right-sim_head_p1024-50.zarr
    python scripts/sim/viz_clouds.py <zarr> --episodes 0 7 31 --num-group 32 --stride 3

What it shows is exactly what training reads: `data/point_cloud` (1024 points,
crop + FPS already applied) and `data/point_label`, plus the `num_group` token
centres the encoder's FPS would pick on the box-normalized cloud (random start,
as in training; `r3d/common/token_allocation.py`). Buttons switch the point
colouring between RGB and class; the slider steps through frames; the title
gives per-class point and token counts for the frame.

Output: R3D/data/outputs/sim_view/clouds/<zarr stem>_ep<k>_g<G>.html
"""

import argparse
import importlib.util
import json
import os
import sys

import numpy as np
import zarr

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "R3D"))
from r3d.common.token_allocation import box_normalize  # noqa: E402

PALETTE = {0: "#000000", 1: "#a07850", 2: "#888888", 3: "#555555", 4: "#1f77b4", 5: "#1f77b4",
           6: "#7f7f7f", 7: "#ff7f0e", 8: "#d62728", 9: "#2ca02c", 10: "#9467bd"}


def load_point_labels():
    spec = importlib.util.spec_from_file_location(
        "point_labels", os.path.join(ROOT, "R3D", "r3d", "env", "robotwin2", "envs", "utils", "point_labels.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def fps_centres(xyz_norm, k, rng):
    """Index of k FPS centres with a random start (numpy; same algorithm as pytorch3d)."""
    n = len(xyz_norm)
    idx = np.empty(k, dtype=np.int64)
    idx[0] = rng.integers(n)
    d = np.full(n, np.inf)
    for i in range(1, k):
        d = np.minimum(d, ((xyz_norm - xyz_norm[idx[i - 1]]) ** 2).sum(1))
        idx[i] = int(np.argmax(d))
    return idx


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("zarr")
    ap.add_argument("--episodes", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--num-group", type=int, default=128)
    ap.add_argument("--stride", type=int, default=4, help="every n-th frame")
    ap.add_argument("--num-points", type=int, default=None,
                    help="show only the first N points (FPS order = the FPS-N cloud the policy sees)")
    ap.add_argument("--drop-labels", type=int, nargs="*", default=None,
                    help="oracle cut: remove these classes BEFORE the token FPS, as training with "
                         "task.drop_point_labels does (e.g. 4 5 = arms, 4 5 6 7 = arms + grippers); "
                         "the removed points are shown faint grey (toggle)")
    ap.add_argument("--out", default=os.path.join(ROOT, "R3D", "data", "outputs", "sim_view", "clouds"))
    a = ap.parse_args()

    import plotly.graph_objects as go

    pl = load_point_labels()
    root = zarr.open(a.zarr, mode="r")
    ends = root["meta/episode_ends"][:]
    bbox = root["meta"].attrs["pcd_crop_bbox"]
    starts = np.concatenate([[0], ends[:-1]])
    os.makedirs(a.out, exist_ok=True)
    rng = np.random.default_rng(0)
    names = pl.CLASS_NAMES
    lo, hi = np.asarray(bbox[0]), np.asarray(bbox[1])

    for ep in a.episodes:
        s, e = int(starts[ep]), int(ends[ep])
        ts = list(range(s, e, a.stride))
        pc = root["data/point_cloud"][s:e, :a.num_points]
        lab = root["data/point_label"][s:e, :a.num_points]
        frames, titles = [], []
        for t in ts:
            p, l = pc[t - s], lab[t - s]
            if a.drop_labels:
                cut = np.isin(l, a.drop_labels)
                removed, p, l = p[cut], p[~cut], l[~cut]
            else:
                removed = p[:0]
            c_idx = fps_centres(box_normalize(p[:, :3], bbox), a.num_group, rng)
            rgb = [f"rgb({int(r*255)},{int(g*255)},{int(b*255)})" for r, g, b in p[:, 3:6]]
            cls = [PALETTE.get(int(x), "#000") for x in l]
            pts = dict(x=p[:, 0], y=p[:, 1], z=p[:, 2])
            cnt = {names[int(k)]: int(v) for k, v in zip(*np.unique(l, return_counts=True))}
            ccnt = {names[int(k)]: int(v) for k, v in zip(*np.unique(l[c_idx], return_counts=True))}
            title = (f"episode {ep}  frame {t - s}/{e - s - 1}  |  points {json.dumps(cnt)}  |  "
                     f"{a.num_group} token centres {json.dumps(ccnt)}")
            titles.append(title)
            frames.append(go.Frame(name=str(t - s), layout=dict(title=dict(text=title, font=dict(size=11))), data=[
                go.Scatter3d(**pts, mode="markers", marker=dict(size=2, color=rgb), name="points (rgb)"),
                go.Scatter3d(**pts, mode="markers", marker=dict(size=2, color=cls), name="points (class)"),
                go.Scatter3d(x=p[c_idx, 0], y=p[c_idx, 1], z=p[c_idx, 2], mode="markers",
                             marker=dict(size=5, color=[PALETTE.get(int(x), "#000") for x in l[c_idx]],
                                         symbol="diamond", line=dict(width=1, color="black")),
                             name=f"{a.num_group} token centres"),
                go.Scatter3d(x=removed[:, 0], y=removed[:, 1], z=removed[:, 2], mode="markers",
                             marker=dict(size=1.5, color="#d0d0d0", opacity=0.5),
                             name=f"cut ({len(removed)} pts, not seen by the policy)"),
            ]))
        fig = go.Figure(data=frames[0].data, frames=frames)
        fig.data[1].visible = False
        legend = "  ".join(f"<span style='color:{PALETTE[k]}'>&#9632;</span> {names[k]}"
                           for k in (pl.LEFT_ARM, pl.LEFT_GRIPPER, pl.RIGHT_GRIPPER, pl.OBJECT, pl.TARGET, pl.ROBOT_BODY))
        fig.update_layout(
            title=dict(text=titles[0], font=dict(size=11)), height=850,
            scene=dict(aspectmode="data",
                       xaxis=dict(range=[lo[0], hi[0]]), yaxis=dict(range=[lo[1], hi[1]]),
                       zaxis=dict(range=[lo[2], hi[2]]),
                       camera=dict(eye=dict(x=0.0, y=1.6, z=1.1))),  # roughly the head camera's side
            annotations=[dict(text="class colours: " + legend, showarrow=False, x=0, y=1.02,
                              xref="paper", yref="paper", xanchor="left")],
            updatemenus=[
                dict(type="buttons", x=0, y=0.98, xanchor="left", direction="right", buttons=[
                    dict(label="colour: RGB", method="restyle", args=[{"visible": [True, False, True, True]}]),
                    dict(label="colour: class", method="restyle", args=[{"visible": [False, True, True, True]}]),
                    dict(label="tokens on/off", method="restyle", args=[{"visible": "legendonly"}, [2]]),
                    dict(label="hide cut points", method="restyle", args=[{"visible": "legendonly"}, [3]]),
                ]),
                dict(type="buttons", x=0.45, y=0.98, xanchor="left", direction="right", buttons=[
                    dict(label="play", method="animate",
                         args=[None, dict(frame=dict(duration=250, redraw=True), fromcurrent=True)]),
                    dict(label="pause", method="animate",
                         args=[[None], dict(mode="immediate", frame=dict(duration=0, redraw=False))]),
                ]),
            ],
            sliders=[dict(steps=[dict(method="animate", label=f.name,
                                      args=[[f.name], dict(mode="immediate", frame=dict(duration=0, redraw=True))])
                                 for f in frames], currentvalue=dict(prefix="frame "))],
        )
        npts = f"_p{pc.shape[1]}" + (f"_cut{''.join(map(str, a.drop_labels))}" if a.drop_labels else "")
        out = os.path.join(a.out, f"{os.path.basename(a.zarr.rstrip('/'))[:-5]}_ep{ep}{npts}_g{a.num_group}.html")
        fig.write_html(out, include_plotlyjs=True, auto_play=False)
        print(f"episode {ep}: {len(ts)} frames -> {out}")


if __name__ == "__main__":
    main()
