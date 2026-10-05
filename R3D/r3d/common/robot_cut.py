"""Cut the robot out of the fused cloud: both Franka arms and everything behind
the orca hands' wrists (tower, 3D-printed mount, cables), keeping the hands
themselves and the object.

Hypothesis (2026-10-05, ~/ros2_ws/robot_cut_r1_plan.txt): run K beat every
policy with +56 FPS scene tokens; scene tokens are allocated by visible SURFACE,
not relevance, so removing the robot's surface should put more of a fixed
512-token budget on the relevant part without adding tokens.

R1 formulation ("K's exact input minus the robot"): stage 2 runs UNCHANGED
(fuse_cameras, uniform draw of 8192), THEN the points labelled cut are dropped.
The surviving points and their density are exactly what K saw.

v2 (2026-10-05): "cut generously, protect explicitly". v1 used tight mesh
distance fields with a 1 cm margin; it removed no wrong points but far too few
-- the large printed wrist mount is in no mesh, and arm points are noisy. Around
the arms and behind the hands there is nothing task-relevant, so the design is
the standard robot self-filter one (ROS robot_self_filter, MoveIt's
PointCloudOctomapUpdater: per-link padding + a sensor-ray shadow class), with an
explicit keep zone so generosity can never eat the hand:

  1 ARM     the URDF's `panda_link<i>_sc` self-collision capsules (already
            ~3 cm inflated) + `arm_padding`; links 6-7, which sit next to the
            hand, get `near_padding` instead
  2 MOUNT   in hand_root (rigid to link8): every point BEHIND the wrist joint
            (hand z < z_wrist - wrist_offset, fingers point +z), not further
            back than `forearm_back`, within `forearm_radius` of the forearm
            axis, and not further than `palm_side` towards the palm (hand +y;
            the flange and the printed mount are on the -y side, and a curled
            thumb reaches behind the wrist plane on the +y side). Removes
            tower + printed mount + cables whatever their shape.
  3 SHADOW  the straight ray from the point to ITS camera passes through the
            robot -> the point is physically occluded by it -> depth noise
            ("flying pixels" at the arm silhouette). Tested against the TIGHT
            mesh distance fields (+ `shadow_margin`), never the padded volumes:
            a cube the top camera really sees under the arm must survive. The
            last `shadow_skip` metres before the point are ignored (a point
            next to the robot is rule 1's business). Wrist cameras skip their
            own side (their rays start inside their mount).
  4 KEEP    override: within `keep_radius` of either hand -- capsules along
            every finger chain from the wrist joint to the tip (HandFK chains,
            measured hand angles) -- a point is never cut. A cut point saved
            this way is labelled SAVED (kept; reported for diagnostics).

Poses are pure FK from the 14 arm joint angles, exactly the route that places
the wrist clouds (real_bimanual.yaml `flange_pose: source: fk`):

  T_world<-link_i    = base[side] @ panda_fk_links(q)[i]
  T_world<-hand_root = base[side] @ panda_fk_links(q)[8] @ HandFK.mount[side]

numpy only -- the inference server imports this without ROS or trimesh.
"""
import os
from typing import Optional, Sequence

import numpy as np
import yaml

from r3d.common.real_preprocess import (HAND_STATE_SLICES, _PANDA_DH, _PANDA_FLANGE,
                                        _axis_angle_matrix, _dh_transform, _rpy_matrix,
                                        HAND_JOINT_ORDER, HandFK)

SIDES = ("left", "right")
# per-point labels returned by RobotCut.labels; CUT_LABELS are removed
KEEP, ARM, MOUNT, SHADOW, SAVED = 0, 1, 2, 3, 4
LABEL_NAMES = ("keep", "arm", "mount", "shadow", "saved")
CUT_LABELS = (ARM, MOUNT, SHADOW)
N_ARM_LINKS = 8          # link0..link7; link8 is the flange (no geometry)

DEFAULTS = dict(arm_padding=0.04, near_padding=0.015, forearm_radius=0.15,
                forearm_back=0.25, palm_side=0.05, wrist_offset=0.015, keep_radius=0.03,
                shadow=True, shadow_margin=0.01, shadow_skip=0.02)


def panda_fk_links(joints: Sequence[float]) -> np.ndarray:
    """(9, 4, 4): T_link0<-link_i for i = 0..7, then link8 (the flange).

    Same DH chain as `panda_fk_flange`; index 8 equals it exactly."""
    joints = np.asarray(joints, dtype=np.float64).reshape(-1)
    if joints.shape != (7,):
        raise ValueError(f"Panda FK needs 7 joint angles, got {joints.shape}")
    out = np.empty((9, 4, 4))
    T = np.eye(4)
    out[0] = T
    for i, ((a, d, alpha), theta) in enumerate(zip(_PANDA_DH, joints)):
        T = T @ _dh_transform(a, d, alpha, theta)
        out[i + 1] = T
    a, d, alpha = _PANDA_FLANGE
    out[8] = T @ _dh_transform(a, d, alpha, 0.0)
    return out


def _load_cameras(preprocess_yaml: str):
    """(bases per side, cameras) from the preprocess yaml -- the single source
    the fused clouds were placed with. cameras[i] = (mount, side, T) in zarr
    camera order: static -> T_world<-cam, wrist -> T_link8<-cam."""
    with open(preprocess_yaml, "r") as f:
        doc = yaml.safe_load(f)
    bases, cams = {}, []
    for cam in doc["cameras"]:
        T = np.asarray(cam["extrinsics"], dtype=np.float64)
        if cam.get("mount", "static") == "wrist":
            side = next(s for s in SIDES if cam["name"] == f"cam_wrist_{s}")
            fp = cam.get("flange_pose") or {}
            if fp.get("source") != "fk" or "base" not in fp:
                raise ValueError(f"{preprocess_yaml}: {cam['name']} has no FK base")
            bases[side] = np.asarray(fp["base"], dtype=np.float64)
            cams.append(("wrist", side, T))
        else:
            cams.append(("static", None, T))
    if set(bases) != set(SIDES):
        raise ValueError(f"{preprocess_yaml}: need cam_wrist_left/right FK bases")
    return bases, cams


def wrist_bases(preprocess_yaml: str) -> dict:
    """T_world<-link0 per side."""
    return _load_cameras(preprocess_yaml)[0]


def _seg_dist(p, a, b):
    """Distance of points p (N, 3) to segment a-b."""
    ab = b - a
    t = np.clip(((p - a) @ ab) / max(float(ab @ ab), 1e-12), 0.0, 1.0)
    return np.linalg.norm(p - (a + t[:, None] * ab), axis=1)


def _segs_dist(p, a, b):
    """Distance of points p (N, 3) to segments a-b (M, 3) -> (N, M) min over M."""
    ab = b - a                                                     # (M, 3)
    t = np.einsum("nmi,mi->nm", p[:, None, :] - a[None], ab) / \
        np.maximum((ab * ab).sum(1), 1e-12)
    t = np.clip(t, 0.0, 1.0)
    q = a[None] + t[..., None] * ab[None]
    return np.linalg.norm(p[:, None, :] - q, axis=2).min(axis=1)


class _Field:
    """Tight distance field on a regular grid in one part frame."""

    def __init__(self, dist, origin, pitch):
        self.dist = dist                      # (X, Y, Z) float, metres
        self.origin = origin                  # centre of voxel (0, 0, 0)
        self.pitch = float(pitch)
        self.shape = np.asarray(dist.shape)

    def lookup(self, p_local) -> np.ndarray:
        """Field value per point (inf outside the grid)."""
        ijk = np.rint((p_local - self.origin) / self.pitch).astype(np.int64)
        ok = np.all((ijk >= 0) & (ijk < self.shape), axis=1)
        out = np.full(len(p_local), np.inf)
        i = ijk[ok]
        out[ok] = self.dist[i[:, 0], i[:, 1], i[:, 2]]
        return out


class RobotCut:
    """Generous robot cut with a protected hand zone (see module docstring).

    `labels(xyz, joints14, hand_q, cam_ids)`: world xyz (N, 3); joints14 =
    [left q1..q7, right q1..q7] (rad, the h5's joint_states/position and the
    inference node's joints14); hand_q = the 48-vector state (hand angles at
    [14:48], as HandFK reads it) or None to skip the keep zone; cam_ids = (N,)
    camera index per point in preprocess-yaml order, or None to skip shadows.
    """

    def __init__(self, geometry_npz: str, preprocess_yaml: str, fk_yaml: str, **params):
        unknown = set(params) - set(DEFAULTS)
        if unknown:
            raise ValueError(f"unknown RobotCut params {sorted(unknown)}")
        self.p = dict(DEFAULTS, **params)
        g = np.load(geometry_npz, allow_pickle=False)
        self.bases, self.cams = _load_cameras(preprocess_yaml)
        fk = HandFK(fk_yaml)
        self.mount = fk.mount
        self.chains = fk.tips                                  # side -> 5 chains

        # rule 1: sc capsules in their link frames
        self.cap_link = g["cap_link"].astype(int)
        self.cap_p0 = g["cap_p0"].astype(np.float64)
        self.cap_p1 = g["cap_p1"].astype(np.float64)
        pad = np.where(self.cap_link >= 6, self.p["near_padding"], self.p["arm_padding"])
        self.cap_r = g["cap_r"].astype(np.float64) + pad
        # rule 2: wrist plane + forearm axis in hand_root, per side
        self.wrist_z = {s: float(g[f"wrist_{s}"][2]) for s in SIDES}
        self.axis_xy = {s: g[f"forearm_axis_{s}"].astype(np.float64)[:2] for s in SIDES}

        # rule 3: tight fields (links shared by both arms, tower per side)
        pitch = float(g["pitch"])
        self.field_pad = float(g["pad"])
        if self.p["shadow"] and not 0 <= self.p["shadow_margin"] < self.field_pad:
            raise ValueError(f"shadow_margin must be in [0, {self.field_pad})")
        self.fields = {f"link{i}": _Field(g[f"link{i}_dist"].astype(np.float32),
                                          g[f"link{i}_origin"], pitch)
                       for i in range(N_ARM_LINKS)}
        for s in SIDES:
            self.fields[f"tower_{s}"] = _Field(g[f"tower_{s}_dist"].astype(np.float32),
                                               g[f"tower_{s}_origin"], pitch)
        self._sphere = {}
        for key, f in self.fields.items():
            ijk = np.argwhere(f.dist <= self.p["shadow_margin"])
            pts = f.origin + ijk * f.pitch
            c = (pts.min(0) + pts.max(0)) / 2
            self._sphere[key] = (c, float(np.linalg.norm(pts - c, axis=1).max()) + pitch)

    # ------------------------------------------------------------- poses --
    def link_poses(self, joints14) -> dict:
        """side -> (9, 4, 4) T_world<-link_i (i = 0..7, 8 = flange)."""
        q = np.asarray(joints14, dtype=np.float64).reshape(-1)
        if q.shape != (14,):
            raise ValueError(f"need 14 arm joint angles, got {q.shape}")
        return {side: self.bases[side] @ panda_fk_links(q[7 * j:7 * j + 7])
                for j, side in enumerate(SIDES)}

    def hand_root(self, joints14, L=None) -> dict:
        L = L or self.link_poses(joints14)
        return {s: L[s][8] @ self.mount[s] for s in SIDES}

    def camera_origins(self, joints14, L=None) -> list:
        """(camera index -> (origin xyz, side or None)) for the ray test."""
        L = L or self.link_poses(joints14)
        out = []
        for mount, side, T in self.cams:
            Tw = T if mount == "static" else L[side][8] @ T
            out.append((Tw[:3, 3], side))
        return out

    def hand_polylines(self, side, q17, T_world_hand_root) -> np.ndarray:
        """(M, 2, 3) keep-zone segments: per finger chain, joint origin to
        joint origin, wrist -> tip (world)."""
        segs = []
        for chain in self.chains[side]:
            T = np.eye(4)
            pts = []
            for e in chain:
                step = np.eye(4)
                step[:3, :3] = _rpy_matrix(e["rpy"])
                step[:3, 3] = e["xyz"]
                T = T @ step
                if "joint" in e:
                    pts.append(T[:3, 3].copy())
                    rot = np.eye(4)
                    rot[:3, :3] = _axis_angle_matrix(
                        e["axis"], q17[HAND_JOINT_ORDER.index(e["joint"])])
                    T = T @ rot
            pts.append(T[:3, 3].copy())                        # the tip
            P = np.asarray(pts) @ T_world_hand_root[:3, :3].T + T_world_hand_root[:3, 3]
            segs += [(P[i], P[i + 1]) for i in range(len(P) - 1)]
        return np.asarray(segs)

    # ------------------------------------------------------------- rules --
    def _rule_arm(self, xyz, L):
        hit = np.zeros(len(xyz), bool)
        for s in SIDES:
            for c in range(len(self.cap_r)):
                T = L[s][self.cap_link[c]]
                a = T[:3, :3] @ self.cap_p0[c] + T[:3, 3]
                b = T[:3, :3] @ self.cap_p1[c] + T[:3, 3]
                hit |= _seg_dist(xyz, a, b) <= self.cap_r[c]
        return hit

    def _in_mount(self, p_hand, side):
        """Rule 2 membership for points already in `side`'s hand_root frame."""
        z = p_hand[..., 2]
        r = np.linalg.norm(p_hand[..., :2] - self.axis_xy[side], axis=-1)
        zw = self.wrist_z[side]
        y = p_hand[..., 1] - self.axis_xy[side][1]
        return ((z < zw - self.p["wrist_offset"]) & (z > zw - self.p["forearm_back"])
                & (r <= self.p["forearm_radius"]) & (y <= self.p["palm_side"]))

    def _rule_mount(self, xyz, H):
        hit = np.zeros(len(xyz), bool)
        for s in SIDES:
            T = H[s]
            hit |= self._in_mount((xyz - T[:3, 3]) @ T[:3, :3], s)
        return hit

    def _rule_shadow(self, xyz, cam_ids, L, H, n_samples=24):
        hit = np.zeros(len(xyz), bool)
        m, skip = self.p["shadow_margin"], self.p["shadow_skip"]
        parts = [(s, f"link{i}", L[s][i]) for s in SIDES for i in range(N_ARM_LINKS)] + \
                [(s, f"tower_{s}", H[s]) for s in SIDES]
        for ci, (origin, cam_side) in enumerate(self.camera_origins(None, L)):
            sel = np.flatnonzero(cam_ids == ci)
            if not len(sel):
                continue
            p = xyz[sel]
            d = p - origin                                      # ray: origin + s * d
            length = np.linalg.norm(d, axis=1)
            s_max = np.clip(1.0 - skip / np.maximum(length, 1e-9), 0.0, 1.0)
            for side, key, T in parts:
                if cam_side is not None and side == cam_side:
                    continue                                    # own side: ray starts inside it
                c, r = self._sphere[key]
                cw = T[:3, :3] @ c + T[:3, 3]
                # chord of the ray inside the part's bounding sphere
                oc = origin - cw
                A = (d * d).sum(1)
                B = 2 * (d @ oc)
                C = oc @ oc - r * r
                disc = B * B - 4 * A * C
                s0 = (-B - np.sqrt(np.maximum(disc, 0))) / (2 * A)
                s1 = (-B + np.sqrt(np.maximum(disc, 0))) / (2 * A)
                s0 = np.maximum(s0, 0.0)
                s1 = np.minimum(s1, s_max)
                live = np.flatnonzero((disc > 0) & (s1 > s0) & ~hit[sel])
                if not len(live):
                    continue
                t = np.linspace(0.0, 1.0, n_samples)
                ss = s0[live, None] + (s1 - s0)[live, None] * t[None]       # (n, S)
                q = origin + ss[..., None] * d[live, None, :]                # (n, S, 3)
                ql = (q.reshape(-1, 3) - T[:3, 3]) @ T[:3, :3]
                v = self.fields[key].lookup(ql).reshape(len(live), n_samples)
                hit[sel[live[(v <= m).any(axis=1)]]] = True
        return hit

    def _keep_zone(self, xyz, hand_q, H):
        q = np.asarray(hand_q, dtype=np.float64).reshape(-1)
        segs = np.concatenate([self.hand_polylines(s, q[HAND_STATE_SLICES[s]], H[s])
                               for s in SIDES])
        lo = segs.reshape(-1, 3).min(0) - self.p["keep_radius"]
        hi = segs.reshape(-1, 3).max(0) + self.p["keep_radius"]
        near = np.flatnonzero(np.all((xyz >= lo) & (xyz <= hi), axis=1))
        keep = np.zeros(len(xyz), bool)
        if len(near):
            keep[near] = _segs_dist(xyz[near], segs[:, 0], segs[:, 1]) <= self.p["keep_radius"]
        return keep

    # ------------------------------------------------------------- api --
    def labels(self, xyz, joints14, hand_q=None, cam_ids=None) -> np.ndarray:
        """(N,) int8 per point: KEEP / ARM / MOUNT / SHADOW / SAVED."""
        xyz = np.asarray(xyz, dtype=np.float64)[:, :3]
        out = np.zeros(len(xyz), np.int8)
        if not len(xyz):
            return out
        L = self.link_poses(joints14)
        H = self.hand_root(None, L)
        if self.p["shadow"] and cam_ids is not None:
            out[self._rule_shadow(xyz, np.asarray(cam_ids), L, H)] = SHADOW
        out[self._rule_mount(xyz, H)] = MOUNT
        out[self._rule_arm(xyz, L)] = ARM
        if hand_q is not None:
            out[(out != KEEP) & self._keep_zone(xyz, hand_q, H)] = SAVED
        return out

    def mask(self, xyz, joints14, hand_q=None, cam_ids=None) -> np.ndarray:
        """(N,) bool, True = cut."""
        return np.isin(self.labels(xyz, joints14, hand_q, cam_ids), CUT_LABELS)


def default_paths(root: Optional[str] = None) -> dict:
    """Defaults relative to the R3D-Policy checkout this module lives in."""
    root = root or os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")
    pre = os.path.join(root, "R3D", "r3d", "config", "preprocess")
    return {"geometry_npz": os.path.normpath(os.path.join(pre, "robot_cut_geometry.npz")),
            "preprocess_yaml": os.path.normpath(os.path.join(pre, "real_bimanual.yaml")),
            "fk_yaml": os.path.normpath(os.path.join(pre, "orcahand_v1b_fk.yaml"))}


def add_cli_args(ap):
    """--arm-padding ... --shadow-skip, defaults from DEFAULTS (shared by the tools)."""
    for k, v in DEFAULTS.items():
        flag = "--" + k.replace("_", "-")
        if isinstance(v, bool):
            ap.add_argument(flag, type=lambda x: str(x).lower() in ("1", "true", "yes"),
                            default=v, help=f"robot cut {k} (default {v})")
        else:
            ap.add_argument(flag, type=float, default=v, help=f"robot cut {k} [m] (default {v})")


def params_from_args(args) -> dict:
    return {k: getattr(args, k) for k in DEFAULTS}


def params_tag(params: dict) -> str:
    """Short stable tag of the non-default params, for output file names."""
    diff = [f"{k}{params[k]:g}" if not isinstance(params[k], bool) else f"{k}{int(params[k])}"
            for k in DEFAULTS if params[k] != DEFAULTS[k]]
    return "default" if not diff else "_".join(diff).replace("_padding", "pad").replace(".", "p")
