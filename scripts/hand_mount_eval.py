#!/usr/bin/env python3
"""Measure how far the posed orca hand MODEL is from the real hand in the clouds.

Evaluation only -- nothing is changed. The mount link8 -> hand_root in
orcahand_v1b_fk.yaml is analytic (Desk EE [130,0,70] mm taken as the palm
centre + init-pose orientation). If that assumption is off, the WHOLE hand
model is shifted by a constant in the link8 frame, which is what this script
measures:

  1. the full hand model (all visual meshes, area-sampled) is posed with the
     MEASURED hand angles through the full URDF tree, placed with the current
     mount and the recorded flange pose (per-frame wrist extrinsic @ inv(hand-eye));
  2. per camera, the real cloud points near the posed model are collected over
     many frames where the hands are free (early in the episode, before the
     grasp), each expressed in that hand's link8 frame;
  3. a translation is estimated robustly: gated nearest-point residuals
     (cloud point -> model surface), median, iterated with a shrinking gate.

Per camera, because the top camera is rigid while the wrist cameras flex up to
~14 mm: if the TOP camera shows the same offset, it is the mount, not flex.

    micromamba activate r3d && unset PYTHONPATH
    cd ~/R3D-Policy
    python scripts/hand_mount_eval.py                  # 12 episodes, phase < 0.15
"""
import argparse
import os
import re
import subprocess
import sys
import xml.etree.ElementTree as ET

import numpy as np
import trimesh
import yaml
import zarr
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "R3D"))
from r3d.common.real_preprocess import (  # noqa: E402
    HAND_STATE_SLICES, HandFK, _axis_angle_matrix, _rpy_matrix)

VIZ = os.path.expanduser("~/ros2_ws/src/faive_system/src/viz")
XACRO = os.path.join(VIZ, "models/orcahand_v1b/orcahand_v1b.urdf.xacro")
XACRO_BIN = os.path.expanduser("~/micromamba/envs/ros_env/bin/xacro")


def _origin(el):
    T = np.eye(4)
    o = el.find("origin") if el is not None else None
    if o is not None:
        T[:3, :3] = _rpy_matrix([float(v) for v in o.get("rpy", "0 0 0").split()])
        T[:3, 3] = [float(v) for v in o.get("xyz", "0 0 0").split()]
    return T


class HandModel:
    """Area-sampled visual surface of one hand + full-tree FK, from the xacro."""

    def __init__(self, side, n_points=8000, seed=0):
        env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
        urdf = subprocess.check_output([XACRO_BIN, XACRO, f"hand_chirality:={side}"],
                                       text=True, env=env)
        root = ET.fromstring(urdf)
        self.joints = {}
        for j in root.findall("joint"):
            ax = j.find("axis")
            self.joints[j.find("child").get("link")] = dict(
                name=j.get("name"), type=j.get("type"), parent=j.find("parent").get("link"),
                T=_origin(j),
                axis=None if ax is None else [float(v) for v in ax.get("xyz").split()])
        rng = np.random.default_rng(seed)
        meshes = []
        for link in root.findall("link"):
            for vis in link.findall("visual"):
                m = vis.find("geometry/mesh")
                if m is None:
                    continue
                path = re.sub(r"^package://viz/", VIZ + "/", m.get("filename"))
                tm = trimesh.load(path, force="mesh")
                scale = np.array([float(v) for v in m.get("scale", "1 1 1").split()])
                tm.apply_scale(1.0)  # keep; scale applied per vertex below
                tm.vertices = tm.vertices * scale
                tm.apply_transform(_origin(vis))
                meshes.append((link.get("name"), tm))
        areas = np.array([tm.area for _, tm in meshes])
        counts = np.maximum(1, np.round(n_points * areas / areas.sum()).astype(int))
        self.samples = [(ln, trimesh.sample.sample_surface(tm, c, seed=int(rng.integers(1e9)))[0])
                        for (ln, tm), c in zip(meshes, counts)]

    def link_T(self, link, q_by_name, cache):
        if link in cache:
            return cache[link]
        if link not in self.joints:           # root (hand_root)
            cache[link] = np.eye(4)
            return cache[link]
        j = self.joints[link]
        T = self.link_T(j["parent"], q_by_name, cache) @ j["T"]
        if j["type"] == "revolute":
            R = np.eye(4)
            R[:3, :3] = _axis_angle_matrix(j["axis"], q_by_name[j["name"]])
            T = T @ R
        cache[link] = T
        return T

    def points(self, q_by_name):
        """(M, 3) model surface points in hand_root."""
        cache, out = {}, []
        for ln, p in self.samples:
            T = self.link_T(ln, q_by_name, cache)
            out.append(p @ T[:3, :3].T + T[:3, 3])
        return np.concatenate(out)


def robust_translation(pairs_fn, gates=(0.06, 0.04, 0.03, 0.02, 0.02, 0.02)):
    """Iterate t so that median(cloud - nearest model point) -> 0."""
    t = np.zeros(3)
    for g in gates:
        d = pairs_fn(t, g)
        if len(d) < 50:
            return t, d, False
        t = t + np.median(d, axis=0)
    return t, pairs_fn(t, 0.02), True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zarr", default="R3D/data/real_bimanual_mixed150.zarr")
    ap.add_argument("--preprocess", default="R3D/r3d/config/preprocess/real_bimanual.yaml")
    ap.add_argument("--fk", default="R3D/r3d/config/preprocess/orcahand_v1b_fk.yaml")
    ap.add_argument("--episodes", default="0,12,25,37,50,62,75,87,100,112,125,137")
    ap.add_argument("--max-phase", type=float, default=0.15)
    ap.add_argument("--frames-per-episode", type=int, default=6)
    args = ap.parse_args()

    raw = yaml.safe_load(open(args.preprocess))
    names = [c["name"] for c in raw["cameras"]]
    fk = HandFK(args.fk)
    root = zarr.open(args.zarr, mode="r")
    if list(root["meta"].attrs["camera_names"]) != names:
        raise ValueError("camera order mismatch")
    ends = np.asarray(root["meta/episode_ends"][:])
    starts = np.concatenate([[0], ends[:-1]])
    statics = {i: np.asarray(root[f"meta/extrinsics_cam{i}"][:], dtype=np.float64)
               for i in range(len(names)) if f"meta/extrinsics_cam{i}" in root}
    inv_he = {s: np.linalg.inv(np.asarray(raw["cameras"][names.index(f"cam_wrist_{s}")]
                                          ["extrinsics"], dtype=np.float64)) for s in HandFK.SIDES}
    models = {s: HandModel(s) for s in HandFK.SIDES}
    joint_names = list(fk.tips["left"] and __import__(
        "r3d.common.real_preprocess", fromlist=["HAND_JOINT_ORDER"]).HAND_JOINT_ORDER)

    # collect, per (side, camera, episode): cloud points and posed model points in link8
    buckets = {}
    for ep in [int(e) for e in args.episodes.split(",")]:
        s, e = int(starts[ep]), int(ends[ep])
        last = s + int(args.max_phase * (e - s - 1))
        for t in np.linspace(s, last, args.frames_per_episode).round().astype(int):
            Ts = [statics[i] if i in statics else
                  np.asarray(root[f"data/extrinsics_cam{i}"][t], dtype=np.float64)
                  for i in range(len(names))]
            state = np.asarray(root["data/state"][t], dtype=np.float64)
            for side in HandFK.SIDES:
                L8 = Ts[names.index(f"cam_wrist_{side}")] @ inv_he[side]
                q = dict(zip(joint_names, state[HAND_STATE_SLICES[side]]))
                M = fk.mount[side]
                model = models[side].points(q) @ M[:3, :3].T + M[:3, 3]     # link8
                inv_L8 = np.linalg.inv(L8)
                for ci, cname in enumerate(names):
                    pc = np.asarray(root[f"data/point_cloud_cam{ci}"][t])[:, :3].astype(np.float64)
                    w = pc @ Ts[ci][:3, :3].T + Ts[ci][:3, 3]                # world
                    p8 = w @ inv_L8[:3, :3].T + inv_L8[:3, 3]                  # link8
                    buckets.setdefault((side, cname), []).append((ep, p8, model))

    print(f"frames: episodes {args.episodes}, phase <= {args.max_phase}, "
          f"{args.frames_per_episode}/episode\n")
    print(f"{'hand':5s} {'camera':16s} {'offset in link8 [mm] (x, y, z)':34s} {'|t|':>6s} "
          f"{'pts':>6s} {'resid med':>9s} {'before':>7s}   per-episode |t| spread")
    for (side, cname), items in sorted(buckets.items()):
        trees = [cKDTree(m) for _, _, m in items]

        def pairs(tvec, gate, items=items, trees=trees):
            out = []
            for (ep, p8, model), tree in zip(items, trees):
                dist, idx = tree.query(p8 - tvec, distance_upper_bound=gate)
                ok = np.isfinite(dist)
                out.append(p8[ok] - tvec - model[idx[ok]])
            return np.concatenate(out) if out else np.zeros((0, 3))

        before = pairs(np.zeros(3), 0.02)
        t, resid, ok = robust_translation(pairs)
        per_ep = []
        for ep in sorted(set(i[0] for i in items)):
            sub = [(it, tr) for it, tr in zip(items, trees) if it[0] == ep]

            def pairs_ep(tvec, gate, sub=sub):
                out = []
                for (_, p8, model), tree in sub:
                    dist, idx = tree.query(p8 - tvec, distance_upper_bound=gate)
                    okk = np.isfinite(dist)
                    out.append(p8[okk] - tvec - model[idx[okk]])
                return np.concatenate(out) if out else np.zeros((0, 3))
            te, _, oke = robust_translation(pairs_ep)
            if oke:
                per_ep.append(te)
        per_ep = np.array(per_ep) * 1000
        spread = (f"n={len(per_ep)}  median {np.median(np.linalg.norm(per_ep, axis=1)):.0f} mm, "
                  f"std xyz {np.round(per_ep.std(0)).astype(int).tolist()}") if len(per_ep) else "-"
        print(f"{side:5s} {cname:16s} {str(np.round(t * 1000, 1).tolist()):34s} "
              f"{np.linalg.norm(t) * 1000:6.1f} {len(resid):6d} "
              f"{np.median(np.linalg.norm(resid, axis=1)) * 1000:7.1f}mm "
              f"{np.median(np.linalg.norm(before, axis=1)) * 1000 if len(before) else float('nan'):5.1f}mm"
              f"{'' if ok else ' (too few points)'}   {spread}")
    print("\noffset = where the real hand is relative to the model (add it to the mount "
          "translation to move the model onto the cloud).\nlink8 axes at the init pose: "
          "+x = world +x (finger direction), +y = world -y, +z = world -z (down).")


if __name__ == "__main__":
    main()
