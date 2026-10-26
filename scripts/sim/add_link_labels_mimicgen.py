#!/usr/bin/env python
"""Add per-point link cells (`data/point_link`) to a MimicGen zarr.  Run in r3d_robosuite.

    python scripts/sim/add_link_labels_mimicgen.py --check 2            # verify 2 demos, write nothing
    python scripts/sim/add_link_labels_mimicgen.py --workers 8          # all demos, then write

The learned link remover (docs/learned_remover_v0.md) needs one id per Panda link, but the
converter stored only classes (arm 5, gripper 7). Every frame is re-rendered from the hdf5
states with the SAME `CloudBuilder` and a `robosuite_pcd.LinkCells`, which only adds an
output. The re-rendered cloud and `point_label` must equal the stored ones BIT FOR BIT --
any mismatch aborts before anything is written. Then `data/point_link` (N_frames, N) int8
(CELL_NAMES index, -1 = not robot) is added; no existing array is touched.

FK labels (brief section 2), every `--fk-stride`-th frame: the robot's visual mesh vertices
placed by the simulator's geom poses (= forward kinematics), nearest vertex per point, cell
by the same gripper split. Reported against segmentation per distance threshold: agreement
on robot points, and non-robot (nut / peg) points FK would call robot -- the contact-region
risk for the real robot, where segmentation does not exist. Report only, not stored.
Report: R3D/data/outputs/sim/link_labels/<zarr name>.json
"""

import argparse
import json
import multiprocessing as mp
import os
import sys
import time

os.environ.setdefault("MUJOCO_GL", "egl")
import h5py
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
R3D_DIR = os.path.join(ROOT, "R3D")
sys.path.insert(0, R3D_DIR)
sys.path.insert(0, os.path.dirname(__file__))

FK_THRESH_M = (0.005, 0.010, 0.020)
NON_ROBOT_CLASSES = (8, 9, 10)          # object, target, other (robosuite_pcd)


def hdf5_path(task):
    return os.path.join(R3D_DIR, "data", "mimicgen", "core", f"{task}.hdf5")


def surface_samples(m, mid, spacing, rng):
    """Mesh vertices plus points sampled uniformly on its triangles (~`spacing` apart), in
    the geom frame. Vertices alone are a poor surface: large flat faces have none inside."""
    v = m.mesh_vert[m.mesh_vertadr[mid]:m.mesh_vertadr[mid] + m.mesh_vertnum[mid]].astype(np.float64)
    fa = m.mesh_face[m.mesh_faceadr[mid]:m.mesh_faceadr[mid] + m.mesh_facenum[mid]]
    a, b, c = v[fa[:, 0]], v[fa[:, 1]], v[fa[:, 2]]
    area = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
    k = np.ceil(area / spacing ** 2).astype(np.int64)
    tri = np.repeat(np.arange(len(fa)), k)
    r1, r2 = np.sqrt(rng.random(len(tri))), rng.random(len(tri))
    pts = (1 - r1)[:, None] * a[tri] + (r1 * (1 - r2))[:, None] * b[tri] + (r1 * r2)[:, None] * c[tri]
    return np.concatenate([v, pts])


class FKCells:
    """Nearest point of the robot's visual mesh SURFACES per cloud point, the meshes placed
    by the simulator's geom poses. -> (distance (P,), cell (P,)) by LinkCells' split."""

    def __init__(self, sim, link_cells, spacing=0.0015):
        m = sim.model
        self.sim, self.lc = sim, link_cells
        self.gids = [g for g in range(m.ngeom) if m.geom_group[g] == 1 and m.geom_type[g] == 7
                     and (link_cells.base[g] >= 0 or g == link_cells.hand or g in link_cells.fingers)]
        rng = np.random.default_rng(0)
        self.verts = [surface_samples(m, m.geom_dataid[g], spacing, rng) for g in self.gids]
        self.owner = np.concatenate([np.full(len(v), g, np.int64) for g, v in zip(self.gids, self.verts)])

    def __call__(self, pw):
        from scipy.spatial import cKDTree
        d = self.sim.data
        world = np.concatenate([d.geom_xpos[g] + v @ d.geom_xmat[g].reshape(3, 3).T
                                for g, v in zip(self.gids, self.verts)])
        dist, j = cKDTree(world).query(pw, k=1)
        geom = self.owner[j]
        return dist, self.lc(pw, geom)


def label_demo(env, f, ep, a, stored_pc, stored_lab):
    from r3d.common import robosuite_pcd as RP
    from mimicgen_view_check import reset_to_demo
    g = f[f"data/demo_{ep}"]
    S = g["states"][:]
    reset_to_demo(env, g.attrs["model_file"], S[0])
    lut = RP.geom_label_lut(env.sim, a.env_name)
    cb = RP.CloudBuilder(env.sim, a.camera, a.res, a.res, a.bbox, a.num_points, lut)
    lc = RP.LinkCells(env.sim)
    fk = FKCells(env.sim, lc)
    T = len(S)
    if T != len(stored_pc):
        raise RuntimeError(f"demo {ep}: {T} states but {len(stored_pc)} stored frames")
    cells = np.full((T, a.num_points), RP.NO_CELL, np.int8)
    n_pc_diff = n_lab_diff = 0
    fk_rows = []          # per FK frame: seg cell, fk dist, fk cell, class
    for t in range(T):
        env.sim.set_state_from_flattened(S[t])
        env.sim.forward()
        obs = env._get_observations(force_update=True)
        pc, lab, _, cell = cb.build(obs, cells=lc)
        n_pc_diff += int(not np.array_equal(pc, stored_pc[t]))
        n_lab_diff += int(not np.array_equal(lab, stored_lab[t]))
        robot = np.isin(lab, (RP.RIGHT_ARM, RP.RIGHT_GRIPPER))
        if ((cell >= 0) != robot).any():
            raise RuntimeError(f"demo {ep} frame {t}: link cells disagree with the robot classes")
        cells[t] = cell
        if t % a.fk_stride == 0:
            dist, fcell = fk(pc[:, :3].astype(np.float64))
            fk_rows.append(np.stack([cell.astype(np.float64), dist, fcell.astype(np.float64), lab.astype(np.float64)], 1))
    return cells, n_pc_diff, n_lab_diff, np.concatenate(fk_rows)


def worker(args):
    a, eps, out_dir, env_args, ep_ranges = args
    import torch  # noqa: F401  (cuda init per process)
    import zarr
    from mimicgen_view_check import make_env
    env = make_env(env_args, a.camera, a.res)
    f = h5py.File(hdf5_path(a.task), "r")
    z = zarr.open(a.zarr, "r")
    for ep in eps:
        p = os.path.join(out_dir, f"ep{ep:05d}.npz")
        if os.path.exists(p):
            continue
        s, e = ep_ranges[ep]
        cells, dpc, dlab, fk = label_demo(env, f, ep, a, z["data/point_cloud"][s:e], z["data/point_label"][s:e])
        np.savez(p + ".tmp.npz", cells=cells, n_pc_diff=dpc, n_lab_diff=dlab, fk=fk.astype(np.float32))
        os.replace(p + ".tmp.npz", p)
        print(f"[worker {os.getpid()}] demo {ep}: {e - s} frames, cloud mismatches {dpc}, label mismatches {dlab}", flush=True)
    return True


def fk_report(fk):
    from r3d.common.robosuite_pcd import CELL_NAMES
    seg_cell, dist, fk_cell, cls = fk[:, 0].astype(int), fk[:, 1], fk[:, 2].astype(int), fk[:, 3].astype(int)
    robot = seg_cell >= 0
    nonrobot = np.isin(cls, NON_ROBOT_CLASSES)
    link_of = lambda c: np.where(c < 8, c, np.where(c < 14, 8, np.where(c < 18, 9, 10)))  # link / hand / f1 / f2
    rep = {"fk_points": int(len(fk)), "robot_points": int(robot.sum()), "non_robot_points": int(nonrobot.sum()),
           "robot_dist_to_own_mesh_mm": {q: float(1000 * np.percentile(dist[robot], q)) for q in (50, 90, 99, 99.9)},
           "non_robot_dist_mm": {q: float(1000 * np.percentile(dist[nonrobot], q)) for q in (0.1, 1, 10, 50)},
           "per_threshold": {}}
    for th in FK_THRESH_M:
        fk_robot = dist <= th
        r = {"robot_recall": float(fk_robot[robot].mean()),
             "cell_agreement_on_robot": float((fk_robot & (fk_cell == seg_cell))[robot].mean()),
             "link_agreement_on_robot": float((fk_robot & (link_of(fk_cell) == link_of(seg_cell)))[robot].mean()),
             "non_robot_called_robot": float(fk_robot[nonrobot].mean()),
             "non_robot_called_robot_count": int(fk_robot[nonrobot].sum())}
        rep["per_threshold"][f"{int(1000 * th)}mm"] = r
    conf = {}
    for c in range(len(CELL_NAMES)):
        sel = robot & (seg_cell == c)
        if sel.sum():
            conf[CELL_NAMES[c]] = {"n": int(sel.sum()), "fk_same_cell_10mm": float(((dist <= 0.01) & (fk_cell == c))[sel].mean())}
    rep["per_cell_10mm"] = conf
    return rep


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--zarr", default=os.path.join(R3D_DIR, "data", "square_d2-agentview_p4096-200.zarr"))
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--fk-stride", type=int, default=5)
    ap.add_argument("--check", type=int, default=0, help="only verify the first K demos; write nothing")
    a = ap.parse_args()

    import zarr
    from r3d.common import robosuite_pcd as RP
    z = zarr.open(a.zarr, "r")
    attrs = dict(z["meta"].attrs)
    if attrs.get("source") != "mimicgen":
        raise ValueError(f"{a.zarr} is not a MimicGen zarr")
    if "point_link" in z["data"]:
        raise FileExistsError(f"{a.zarr} already has data/point_link")
    a.task, a.camera, a.res, a.env_name = attrs["task"], attrs["camera"], attrs["render_res"], attrs["env_name"]
    a.bbox, a.num_points = attrs["pcd_crop_bbox"], attrs["pcd_down_sample_num"]
    demos = list(attrs["demos"])
    ends = z["meta/episode_ends"][:]
    starts = np.concatenate([[0], ends[:-1]])
    ep_ranges = {ep: (int(s), int(e)) for ep, s, e in zip(demos, starts, ends)}
    eps = demos[:a.check] if a.check else demos
    with h5py.File(hdf5_path(a.task), "r") as f:
        env_args = json.loads(f["data"].attrs["env_args"])
    if env_args["env_name"] != a.env_name:
        raise ValueError("hdf5 env_name differs from the zarr's")

    name = os.path.basename(a.zarr.rstrip("/"))
    tmp = a.zarr.rstrip("/") + ".link_parts"
    os.makedirs(tmp, exist_ok=True)
    t0 = time.time()
    nw = max(1, min(a.workers, len(eps)))
    chunks = [eps[i::nw] for i in range(nw)]
    with mp.get_context("spawn").Pool(nw) as pool:
        pool.map(worker, [(a, c, tmp, env_args, ep_ranges) for c in chunks])
    print(f"rendered {len(eps)} demos in {(time.time() - t0) / 60:.1f} min", flush=True)

    parts = {ep: np.load(os.path.join(tmp, f"ep{ep:05d}.npz")) for ep in eps}
    bad = {ep: (int(p["n_pc_diff"]), int(p["n_lab_diff"])) for ep, p in parts.items()
           if int(p["n_pc_diff"]) or int(p["n_lab_diff"])}
    rep = {"zarr": a.zarr, "demos": len(eps), "frames": int(sum(len(p["cells"]) for p in parts.values())),
           "cell_names": RP.CELL_NAMES, "mismatched_demos": {str(k): v for k, v in bad.items()},
           "fk": fk_report(np.concatenate([p["fk"] for p in parts.values()]))}
    out_dir = os.path.join(R3D_DIR, "data", "outputs", "sim", "link_labels")
    os.makedirs(out_dir, exist_ok=True)
    rpath = os.path.join(out_dir, name.replace(".zarr", "") + ("_check.json" if a.check else ".json"))
    with open(rpath, "w") as fh:
        json.dump(rep, fh, indent=1)
    print(json.dumps(rep["fk"]["per_threshold"], indent=1))
    print(f"report: {rpath}")
    if bad:
        raise SystemExit(f"re-rendered clouds differ from the stored ones in demos {sorted(bad)} -- nothing written")
    if a.check:
        print(f"check OK: {len(eps)} demos bit-identical; nothing written (--check)")
        return

    cells = np.concatenate([parts[ep]["cells"] for ep in demos])
    if len(cells) != int(ends[-1]):
        raise RuntimeError("frame count mismatch")
    zw = zarr.open(a.zarr, "r+")
    zw.create_dataset("data/point_link", data=cells, chunks=(1, a.num_points),
                      compressor=zw["data/point_label"].compressor)
    zw["meta"].attrs["point_link_names"] = list(RP.CELL_NAMES)
    zw["meta"].attrs["point_link_note"] = ("robosuite_pcd.LinkCells: arm one cell per body, gripper split in "
                                           "its geom frames; -1 = not robot")
    print(f"wrote {a.zarr}/data/point_link {cells.shape} int8")


if __name__ == "__main__":
    main()
