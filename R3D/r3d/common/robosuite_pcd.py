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

    def build(self, obs, device="cuda"):
        """-> cloud (num_points, 6) float32 xyz + rgb[0,1], labels (num_points,) int16,
        n_valid (points in the box before FPS / padding)."""
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
        n = len(pts)
        if n == 0:
            raise RuntimeError("crop box removed every point")
        if n > self.num_points:
            x = torch.as_tensor(pts[None, :, :3], device=device)
            _, idx = sample_farthest_points(x, K=self.num_points, random_start_point=False)
            idx = idx[0].cpu().numpy()
            pts, lab = pts[idx], lab[idx]
        elif n < self.num_points:   # pad with copies of the first point, labelled PAD
            pad = self.num_points - n
            pts = np.concatenate([pts, np.repeat(pts[:1], pad, 0)])
            lab = np.concatenate([lab, np.full(pad, PAD, np.int16)])
        return pts, lab, n

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
