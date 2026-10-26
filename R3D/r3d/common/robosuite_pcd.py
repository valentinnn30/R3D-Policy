"""Head-camera point cloud for robosuite / MimicGen scenes.  Sim token-allocation study.

ONE function builds the cloud for both the zarr conversion and evaluation, so the
two cannot drift (the same rule as `real_preprocess.py` for the real rig):

    render depth + per-geom segmentation from one camera
    -> unproject to the WORLD frame -> crop to the box (the z floor cuts the table)
    -> FPS to `num_points` (deterministic start, so the first N of a stored cloud
       ARE the FPS-N cloud: one zarr serves every point budget)

Every point carries a class from `point_labels.py`'s scheme, so `token_allocation`,
`label_cut` and `viz_clouds.py` work unchanged. The single Panda arm is labelled
RIGHT_ARM / RIGHT_GRIPPER -- the same ids the RoboTwin right-arm tasks used, so
`drop_point_labels=[5]` is the arm cut here too.

robosuite renders in OpenGL convention (row 0 = image BOTTOM); the depth and
segmentation images are flipped to top-left origin before unprojection with the
OpenCV-convention intrinsics/extrinsics of `camera_utils`.
"""

import numpy as np

# class ids of R3D/r3d/env/robotwin2/envs/utils/point_labels.py (kept literal: importing
# that module pulls in RoboTwin, which is not installed in the robosuite env)
PAD, TABLE, BACKGROUND, ROBOT_BODY = 0, 1, 2, 3
RIGHT_ARM, RIGHT_GRIPPER, OBJECT, TARGET, OTHER = 5, 7, 8, 9, 10
CLASS_NAMES = {0: "pad", 1: "table", 2: "background", 3: "robot_body", 4: "left_arm", 5: "right_arm",
               6: "left_gripper", 7: "right_gripper", 8: "object", 9: "target", 10: "other"}
N_CLASSES = 11

# body-name prefix -> class. Task roles (object / target) are per task.
ROBOT_PREFIXES = {"robot0_": RIGHT_ARM, "gripper0_": RIGHT_GRIPPER, "mount0_": ROBOT_BODY}
TASK_ROLES = {
    # Square: square nut onto the square peg; the round nut / round peg are unused
    "Square": {"object": ["SquareNut"], "target": ["peg1"], "other": ["RoundNut", "peg2"]},
}


def task_family(env_name):
    return env_name.split("_D")[0]


def geom_label_lut(sim, env_name):
    """geom id -> class (int16), plus one extra entry at index -1 for 'no geom'."""
    roles = TASK_ROLES[task_family(env_name)]
    m = sim.model
    lut = np.full(m.ngeom + 1, BACKGROUND, dtype=np.int16)
    for g in range(m.ngeom):
        body = m.body_id2name(m.geom_bodyid[g]) or ""
        cls = BACKGROUND
        for pre, c in ROBOT_PREFIXES.items():
            if body.startswith(pre):
                cls = c
        if body == "table" or body.startswith("table"):
            cls = TABLE
        for role, c in (("object", OBJECT), ("target", TARGET), ("other", OTHER)):
            if any(body.startswith(p) for p in roles[role]):
                cls = c
        lut[g] = cls
    return lut


# ---- per-link cells for the learned link remover (docs/learned_remover_v0.md) ----------
# A FINE partition of the Panda's visible surface is stored per point (`point_link`); the
# remover config groups cells into regions, so a coarser or finer grouping needs no
# re-render. The arm is one cell per body. The gripper renders as three visual meshes only
# (hand, two fingers; the pads are invisible collision boxes), so it is split
# GEOMETRICALLY in each mesh's own geom frame: p_local = R_geom^T (p - x_geom), with equal
# bins of the mesh's local bounding box -- analytic, identical at evaluation.
# Axes (measured on Square D2, asserted by `check_gripper_axes` whenever cells are built):
#   hand   geom frame: z = finger-opening axis (finger1 at z < 0), y = approach axis
#          (fingers at y ~ +0.05, link7 at y ~ -0.13)
#   finger geom frame: +z = towards the fingertip, -x = towards the other finger (inner face)
NO_CELL = -1
CELL_NAMES = (
    [f"link{i}" for i in range(8)]
    + [f"hand_{s}_{a}" for s in ("f1side", "centre", "f2side") for a in ("wrist", "front")]
    + [f"finger{f}_{p}_{side}" for f in (1, 2) for p in ("prox", "dist") for side in ("in", "out")])
N_CELLS = len(CELL_NAMES)   # 22
HAND_GEOM, FINGER_GEOMS = "gripper0_hand_visual", ("gripper0_finger1_visual", "gripper0_finger2_visual")


def default_region_groups():
    """v0 grouping (user decision 2026-10-26), 8 LEARNED regions: link5, link6, link7, hand
    f1side / centre / f2side (wrist + front), finger1, finger2 (prox + dist, in + out).
    link0-4 are NOT in any region (never thinned, always kept): on Square D2 they average
    0-5 points per frame at N = 1024 (link3/4 in view in ~5% of frames, link0's points
    mostly silhouette mislabels), so a learned rho for them would only drift on noise."""
    g = [{"name": f"link{i}", "cells": [f"link{i}"]} for i in (5, 6, 7)]
    g += [{"name": f"hand_{s}", "cells": [f"hand_{s}_wrist", f"hand_{s}_front"]} for s in ("f1side", "centre", "f2side")]
    g += [{"name": f"finger{f}", "cells": [f"finger{f}_{p}_{side}" for p in ("prox", "dist") for side in ("in", "out")]}
          for f in (1, 2)]
    return g


def per_link_region_groups():
    """The fine 15-region table (every link, hand x3, fingers prox / dist) -- for stats only."""
    g = [{"name": f"link{i}", "cells": [f"link{i}"]} for i in range(8)]
    g += [{"name": f"hand_{s}", "cells": [f"hand_{s}_wrist", f"hand_{s}_front"]} for s in ("f1side", "centre", "f2side")]
    g += [{"name": f"finger{f}_{p}", "cells": [f"finger{f}_{p}_in", f"finger{f}_{p}_out"]}
          for f in (1, 2) for p in ("prox", "dist")]
    return g


def _mesh_bounds(m, gid):
    mid = m.geom_dataid[gid]
    v = m.mesh_vert[m.mesh_vertadr[mid]:m.mesh_vertadr[mid] + m.mesh_vertnum[mid]]
    return v.min(0).astype(np.float64), v.max(0).astype(np.float64)


def _local(d, gid, p):
    return (p - d.geom_xpos[gid]) @ d.geom_xmat[gid].reshape(3, 3)


def check_gripper_axes(sim):
    """The axis convention above is a contract of the gripper split; fail loudly if a
    different gripper model / mesh frame breaks it."""
    m, d = sim.model, sim.data
    h = m.geom_name2id(HAND_GEOM)
    f1, f2 = (m.geom_name2id(n) for n in FINGER_GEOMS)
    l7 = d.body_xpos[m.body_name2id("robot0_link7")]
    p1, p2, pl = (_local(d, h, x) for x in (d.geom_xpos[f1], d.geom_xpos[f2], l7))
    ok = p1[1] > 0.02 and p2[1] > 0.02 and pl[1] < -0.05 and p1[2] < 0 < p2[2]
    for fa, fb, tip in ((f1, f2, "gripper0_finger_joint1_tip"), (f2, f1, "gripper0_finger_joint2_tip")):
        other = _local(d, fa, d.geom_xpos[fb])
        t = _local(d, fa, d.body_xpos[m.body_name2id(tip)])
        ok &= other[0] < -0.02 and t[2] > 0.01
    if not ok:
        raise RuntimeError("gripper geom frames do not match the axis convention of robosuite_pcd "
                           "(hand: z opening, y approach; finger: +z tip, -x inner)")


class LinkCells:
    """geom id + world point -> cell id (CELL_NAMES index, NO_CELL for non-robot)."""

    def __init__(self, sim):
        m = sim.model
        self.sim = sim
        self.base = np.full(m.ngeom + 1, NO_CELL, dtype=np.int16)   # +1: seg value -1 = no geom
        for g in range(m.ngeom):
            body = m.body_id2name(m.geom_bodyid[g]) or ""
            if m.geom_group[g] != 1:      # only visual geoms are ever rendered
                continue
            if body.startswith("robot0_link"):
                self.base[g] = int(body[len("robot0_link"):])
            elif body.startswith(("robot0_", "gripper0_")) and m.geom_id2name(g) not in (HAND_GEOM,) + FINGER_GEOMS:
                raise RuntimeError(f"visual robot geom {m.geom_id2name(g)} on body {body} has no cell")
        self.hand = m.geom_name2id(HAND_GEOM)
        self.fingers = [m.geom_name2id(n) for n in FINGER_GEOMS]
        self.hand_bounds = _mesh_bounds(m, self.hand)
        self.finger_bounds = _mesh_bounds(m, self.fingers[0])
        if not np.allclose(self.finger_bounds, _mesh_bounds(m, self.fingers[1])):
            raise RuntimeError("the two finger meshes differ")

    @staticmethod
    def _bin(x, lo, hi, n):
        return np.clip(((x - lo) / (hi - lo) * n).astype(np.int64), 0, n - 1)

    def __call__(self, pw, seg):
        """pw (P, 3) world points, seg (P,) geom ids of the same pixels -> (P,) int8."""
        check_gripper_axes(self.sim)
        d = self.sim.data
        out = self.base[seg].astype(np.int8)
        sel = seg == self.hand
        if sel.any():
            q = _local(d, self.hand, pw[sel])
            lo, hi = self.hand_bounds
            s = self._bin(q[:, 2], lo[2], hi[2], 3)                    # f1side / centre / f2side
            a = (q[:, 1] >= (lo[1] + hi[1]) / 2).astype(np.int64)      # wrist / front
            out[sel] = 8 + 2 * s + a
        lo, hi = self.finger_bounds
        for f, gid in enumerate(self.fingers):
            sel = seg == gid
            if sel.any():
                q = _local(d, gid, pw[sel])
                p = (q[:, 2] >= (lo[2] + hi[2]) / 2).astype(np.int64)  # prox / dist
                o = (q[:, 0] >= (lo[0] + hi[0]) / 2).astype(np.int64)  # in (-x) / out
                out[sel] = 14 + 4 * f + 2 * p + o
        return out


def default_bbox(sim, margin_y=0.15, margin_x_robot=0.0):
    """From the table top (+3 mm, so the table itself is cut) to 0.5 m above it.
    x/y: the table footprint, widened by `margin_y` on both sides and by
    `margin_x_robot` towards the robot -- on Square D2 the end effector reaches
    y = +-0.48 and x = -0.40 (0.05% of frames), and a carried nut would otherwise
    leave the cloud. Static clutter (pedestal at x = -0.56, walls, floor) stays cut."""
    m = sim.model
    tid = m.geom_name2id("table_collision")
    c = sim.data.geom_xpos[tid]
    hx, hy, hz = m.geom_size[tid]
    top = float(c[2] + hz)
    return [[float(c[0] - hx - margin_x_robot), float(c[1] - hy - margin_y), top + 0.003],
            [float(c[0] + hx), float(c[1] + hy + margin_y), top + 0.5]]


class CloudBuilder:
    """Holds the camera geometry; `build(obs)` turns one robosuite observation
    (rendered with camera_depths=True, camera_segmentations='element') into a cloud."""

    def __init__(self, sim, camera, height, width, bbox, num_points, lut):
        from robosuite.utils import camera_utils as CU
        self.sim, self.camera, self.H, self.W = sim, camera, height, width
        self.bbox = np.asarray(bbox, dtype=np.float64)
        self.num_points, self.lut = num_points, lut
        K = CU.get_camera_intrinsic_matrix(sim, camera, height, width)
        v, u = np.mgrid[0:height, 0:width]
        self._ray = np.stack([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], np.ones_like(u, float)], -1).reshape(-1, 3)
        self._CU = CU

    def build(self, obs, device="cuda", cells=None):
        """-> cloud (num_points, 6) float32 xyz + rgb[0,1], labels (num_points,) int16,
        n_valid (points in the box before FPS / padding).
        With `cells` (a LinkCells): also -> per-point link cell (num_points,) int8, carried
        through the same keep / FPS / pad indices (pad = NO_CELL); the cloud is unchanged."""
        import torch
        from pytorch3d.ops import sample_farthest_points
        CU, cam = self._CU, self.camera
        depth = CU.get_real_depth_map(self.sim, obs[f"{cam}_depth"])[..., 0]
        depth = np.flipud(depth).reshape(-1)
        seg = np.flipud(obs[f"{cam}_segmentation_element"][..., 0]).reshape(-1)
        rgb = np.flipud(obs[f"{cam}_image"]).reshape(-1, 3).astype(np.float32) / 255.0
        E = CU.get_camera_extrinsic_matrix(self.sim, cam)          # camera -> world, cv axes
        pw = (self._ray * depth[:, None]) @ E[:3, :3].T + E[:3, 3]
        keep = np.all((pw >= self.bbox[0]) & (pw <= self.bbox[1]), axis=1)
        # TABLE / BACKGROUND are what the box exists to cut; the few that survive it
        # are silhouette pixels with foreground depth and background label (measured
        # on Square D2: 97-100% of them sit on a > 2 cm depth jump, vs 5-14% for real
        # surfaces). A depth-edge filter would also cost the thin peg 14% of its points.
        keep &= ~np.isin(self.lut[seg], (TABLE, BACKGROUND))
        pts = np.concatenate([pw[keep], rgb[keep]], 1).astype(np.float32)
        lab = self.lut[seg[keep]]
        cell = None if cells is None else cells(pw[keep], seg[keep])
        n = len(pts)
        if n == 0:
            raise RuntimeError("crop box removed every point")
        if n > self.num_points:
            x = torch.as_tensor(pts[None, :, :3], device=device)
            _, idx = sample_farthest_points(x, K=self.num_points, random_start_point=False)
            idx = idx[0].cpu().numpy()
            pts, lab = pts[idx], lab[idx]
            cell = None if cell is None else cell[idx]
        elif n < self.num_points:   # pad with copies of the first point, labelled PAD
            pad = self.num_points - n
            pts = np.concatenate([pts, np.repeat(pts[:1], pad, 0)])
            lab = np.concatenate([lab, np.full(pad, PAD, np.int16)])
            cell = None if cell is None else np.concatenate([cell, np.full(pad, NO_CELL, np.int8)])
        if cells is None:
            return pts, lab, n
        return pts, lab, n, cell

    def class_pixels(self, obs):
        """valid pixels per class in the whole image and inside the box (view check)."""
        CU, cam = self._CU, self.camera
        depth = np.flipud(CU.get_real_depth_map(self.sim, obs[f"{cam}_depth"])[..., 0]).reshape(-1)
        seg = np.flipud(obs[f"{cam}_segmentation_element"][..., 0]).reshape(-1)
        E = CU.get_camera_extrinsic_matrix(self.sim, cam)
        pw = (self._ray * depth[:, None]) @ E[:3, :3].T + E[:3, 3]
        keep = np.all((pw >= self.bbox[0]) & (pw <= self.bbox[1]), axis=1)
        lab = self.lut[seg]
        return (np.bincount(lab, minlength=N_CLASSES), np.bincount(lab[keep], minlength=N_CLASSES))
