"""Estimate residual extrinsics corrections from inter-camera cloud agreement.

Registration error between the fused cameras is a direct, physical measurement of
how wrong the calibration is. This script turns that measurement into a
correction, by aligning each wrist camera's cloud onto the reference (static)
camera with point-to-plane ICP and then separating the result into the two
constant errors that produce it.

The separation
--------------
Writing A(t) for the nominal wrist-camera pose

    A(t) = T_world<-link0 . FK(joints(t)) . T_link8<-cam

a base error `d_base` (constant in WORLD frame) and a hand-eye error `d_cam`
(constant in CAMERA frame) together produce the world-frame correction

    C(t) = d_base . [ A(t) . d_cam . A(t)^-1 ]

Taking logs and linearising for small errors gives something linear in both
unknowns:

    log C(t)  ~  xi_base  +  Ad_{A(t)} . xi_cam

so stacking frames and solving least squares separates them -- but ONLY if the
arm visits genuinely different poses. With a static arm, Ad_{A(t)} is constant
and the two are perfectly confounded; the script reports the conditioning so
this is visible rather than silent.

Caveats worth reading before trusting the output
------------------------------------------------
* ICP slides freely on planar-dominated overlap (two in-plane translations and
  rotation about the normal). A tabletop is exactly that case. Point-to-plane
  helps, but a scene with real 3D structure in the overlap helps more.
* Depth-sensor bias is range-dependent, not rigid. No rigid transform can absorb
  it, so ICP will misattribute some of it into rotation.
* Only transforms RELATIVE to the reference camera are observable. The global
  frame is not -- which is fine, since a global offset is harmless.

Usage
-----
    python scripts/estimate_extrinsics_correction.py \
        --input-dir /path/to/episodes --config R3D/r3d/config/preprocess/real_bimanual.yaml
"""

import argparse
import glob
import os
import sys

import h5py
import numpy as np
import yaml
from scipy.spatial import cKDTree
from termcolor import cprint

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "R3D"))

from r3d.common.real_preprocess import panda_fk_flange  # noqa: E402


# ------------------------------------------------------------------ se(3) ----

def skew(w):
    return np.array([[0, -w[2], w[1]], [w[2], 0, -w[0]], [-w[1], w[0], 0]])


def log_se3(T):
    """Twist (omega, v) of a small rigid transform. Small-angle approximation."""
    R = T[:3, :3]
    cos_theta = np.clip((np.trace(R) - 1) / 2, -1.0, 1.0)
    theta = np.arccos(cos_theta)
    if theta < 1e-9:
        omega = np.zeros(3)
    else:
        omega = theta / (2 * np.sin(theta)) * np.array(
            [R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    return np.concatenate([omega, T[:3, 3]])


def exp_se3(xi):
    omega, v = xi[:3], xi[3:]
    theta = np.linalg.norm(omega)
    T = np.eye(4)
    if theta < 1e-9:
        T[:3, :3] = np.eye(3) + skew(omega)
    else:
        k = omega / theta
        K = skew(k)
        T[:3, :3] = np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)
    T[:3, 3] = v
    return T


def adjoint(T):
    """Ad_T for twists ordered (omega, v)."""
    R, p = T[:3, :3], T[:3, 3]
    Ad = np.zeros((6, 6))
    Ad[:3, :3] = R
    Ad[3:, :3] = skew(p) @ R
    Ad[3:, 3:] = R
    return Ad


# -------------------------------------------------------------------- ICP ----

def estimate_normals(points, k=20):
    """Per-point surface normal from local PCA. Needed for point-to-plane."""
    tree = cKDTree(points)
    _, idx = tree.query(points, k=k)
    nbrs = points[idx]
    centered = nbrs - nbrs.mean(axis=1, keepdims=True)
    cov = np.einsum("nki,nkj->nij", centered, centered) / k
    _, vecs = np.linalg.eigh(cov)
    return vecs[:, :, 0]          # eigenvector of the smallest eigenvalue


def icp_point_to_plane(src, dst, dst_normals, iters=30, reject=0.04):
    """Rigid transform aligning `src` onto `dst`. Returns (4x4, median residual).

    Point-to-plane rather than point-to-point: it lets points slide along the
    surface they were sampled from, which is the right model when two cameras
    sample the same surface at different spots, and converges far better on
    the near-planar overlap a tabletop produces.
    """
    tree = cKDTree(dst)
    T_total = np.eye(4)
    S = src.copy()
    residual = np.inf
    for _ in range(iters):
        d, i = tree.query(S, k=1, distance_upper_bound=reject)
        m = np.isfinite(d)
        if m.sum() < 200:
            break
        p, q, n = S[m], dst[i[m]], dst_normals[i[m]]
        # linearised point-to-plane: minimise sum(((R p + t) - q) . n)^2
        A = np.hstack([np.cross(p, n), n])
        b = -np.einsum("ij,ij->i", p - q, n)
        xi, *_ = np.linalg.lstsq(A, b, rcond=None)
        step = exp_se3(xi)
        S = S @ step[:3, :3].T + step[:3, 3]
        T_total = step @ T_total
        residual = np.median(np.abs(np.einsum("ij,ij->i", S[m] - q, n)))
    return T_total, residual


# -------------------------------------------------------------- data access --

def tf(pos, rot):
    T = np.eye(4)
    T[:3, :3] = np.asarray(rot, dtype=np.float64)
    T[:3, 3] = np.asarray(pos, dtype=np.float64)
    return T


def read_cloud(f, cam_cfg, t, T_world):
    g_xyz, g_n = cam_cfg["xyz"], cam_cfg["num_points"]
    n = int(np.asarray(f[g_n][t]))
    P = np.asarray(f[g_xyz][t][:n], dtype=np.float64)
    return P @ T_world[:3, :3].T + T_world[:3, 3]


def nominal_wrist_pose(f, cam_cfg, t):
    fp = cam_cfg["flange_pose"]
    base = np.asarray(fp["base"], dtype=np.float64)
    lo, hi = fp["joint_slice"]
    joints = np.asarray(f[fp["joints"]][t])[lo:hi]
    return base @ panda_fk_flange(joints) @ np.asarray(cam_cfg["extrinsics"], np.float64)


# ------------------------------------------------------------------- main ----

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input-dir", required=True)
    p.add_argument("--pattern", default="*.h5")
    p.add_argument("--config", default="R3D/r3d/config/preprocess/real_bimanual.yaml")
    p.add_argument("--reference", default="cam_top", help="static camera to align onto")
    p.add_argument("--frames-per-episode", type=int, default=8)
    p.add_argument("--max-src-points", type=int, default=40000,
                   help="subsample the moving cloud before ICP, for speed")
    args = p.parse_args()

    cfg = yaml.safe_load(open(args.config))
    cams = {c["name"]: c for c in cfg["cameras"]}
    if args.reference not in cams:
        p.error(f"--reference {args.reference} not in config cameras {list(cams)}")
    ref_cfg = cams[args.reference]
    ref_T = np.asarray(ref_cfg["extrinsics"], dtype=np.float64)
    wrists = [c for c in cfg["cameras"] if c.get("mount") == "wrist"]

    files = sorted(glob.glob(os.path.join(args.input_dir, args.pattern)))
    if not files:
        cprint(f"no files matching {args.pattern} in {args.input_dir}", "red")
        sys.exit(1)

    rng = np.random.default_rng(0)
    results = {c["name"]: {"C": [], "A": [], "res": [], "before": []} for c in wrists}

    for path in files:
        with h5py.File(path, "r") as f:
            n_frames = f[ref_cfg["xyz"]].shape[0]
            frames = np.linspace(0, n_frames - 1, args.frames_per_episode).astype(int)
            for t in frames:
                ref_pts = read_cloud(f, ref_cfg, t, ref_T)
                ref_normals = estimate_normals(ref_pts)
                tree_ref = cKDTree(ref_pts)
                for cam in wrists:
                    A = nominal_wrist_pose(f, cam, t)
                    pts = read_cloud(f, cam, t, A)
                    if len(pts) > args.max_src_points:
                        pts = pts[rng.choice(len(pts), args.max_src_points, replace=False)]
                    d, _ = tree_ref.query(pts, k=1, distance_upper_bound=0.05)
                    mm = np.isfinite(d)
                    if mm.mean() < 0.05:
                        continue
                    before = np.median(d[mm]) * 1000
                    C, res = icp_point_to_plane(pts, ref_pts, ref_normals)
                    r = results[cam["name"]]
                    r["C"].append(C)
                    r["A"].append(A)
                    r["res"].append(res * 1000)
                    r["before"].append(before)
        cprint(f"processed {path}", "green")

    print()
    for cam in wrists:
        name = cam["name"]
        r = results[name]
        if len(r["C"]) < 2:
            cprint(f"{name}: too few usable frames ({len(r['C'])})", "red")
            continue

        cprint(f"=== {name} ===", "cyan")
        print(f"  frames used            : {len(r['C'])}")
        print(f"  median NN before ICP   : {np.mean(r['before']):6.1f} mm")
        print(f"  point-to-plane residual: {np.mean(r['res']):6.1f} mm")

        # log C(t) = xi_base + Ad_A(t) xi_cam
        M = np.vstack([np.hstack([np.eye(6), adjoint(A)]) for A in r["A"]])
        y = np.concatenate([log_se3(C) for C in r["C"]])
        sol, *_ = np.linalg.lstsq(M, y, rcond=None)
        xi_b, xi_c = sol[:6], sol[6:]
        fit = np.linalg.norm(M @ sol - y) / np.sqrt(len(r["C"]))

        # conditioning: how separable are the two blocks given these poses?
        s = np.linalg.svd(M, compute_uv=False)
        cond = s[0] / max(s[-1], 1e-15)

        for label, xi in [("base  (world frame)", xi_b), ("handeye (cam frame)", xi_c)]:
            print(f"  {label}: rot {np.degrees(np.linalg.norm(xi[:3])):6.3f} deg   "
                  f"trans {np.linalg.norm(xi[3:]) * 1000:6.2f} mm")
        print(f"  model fit residual     : {fit:.4f}")
        print(f"  separation cond. number: {cond:.3e}", end="  ")
        if cond > 1e6:
            cprint("<- DEGENERATE: arm poses too similar to separate "
                   "base from hand-eye; treat the split as meaningless", "red")
        elif cond > 1e4:
            cprint("<- poorly conditioned; the split is unreliable", "yellow")
        else:
            cprint("ok", "green")

        # A correction far larger than any plausible mounting error means the
        # split has run away -- typically two huge, mutually cancelling terms.
        for label, xi in [("base", xi_b), ("handeye", xi_c)]:
            if np.degrees(np.linalg.norm(xi[:3])) > 10 or np.linalg.norm(xi[3:]) > 0.10:
                cprint(f"  IMPLAUSIBLE {label} correction -- larger than any real "
                       "mounting error. Do not use.", "red")

        # The decisive test: does applying the fitted correction actually improve
        # registration? A good per-frame ICP fit proves nothing if the two
        # constants it decomposes into do not generalise across poses.
        improved, base_line = [], []
        for A, C in zip(r["A"], r["C"]):
            A_corr = exp_se3(xi_b) @ A @ exp_se3(xi_c)
            # displacement the correction induces, vs what ICP asked for
            delta_fit = np.linalg.inv(C @ A) @ A_corr
            improved.append(np.linalg.norm(delta_fit[:3, 3]) * 1000)
            base_line.append(np.linalg.norm(log_se3(C)[3:]) * 1000)
        cprint(f"  fitted-vs-ICP mismatch : {np.mean(improved):6.1f} mm "
               f"(ICP asked for {np.mean(base_line):6.1f} mm)",
               "green" if np.mean(improved) < 0.3 * np.mean(base_line) else "yellow")

        # what the config would become, applying both corrections
        base_new = exp_se3(xi_b) @ np.asarray(cam["flange_pose"]["base"], np.float64)
        cam_new = np.asarray(cam["extrinsics"], np.float64) @ exp_se3(xi_c)
        np.set_printoptions(precision=6, suppress=True, floatmode="fixed")
        print(f"\n  corrected flange_pose.base for {name}:")
        for row in base_new:
            print("      - [" + ", ".join(f"{v:9.6f}" for v in row) + "]")
        print(f"  corrected extrinsics for {name}:")
        for row in cam_new:
            print("      - [" + ", ".join(f"{v:9.6f}" for v in row) + "]")
        print()

    cprint("Reminder: ICP slides on planar overlap and cannot absorb "
           "range-dependent depth bias. Validate against the physical setup "
           "before adopting these numbers.", "yellow")


if __name__ == "__main__":
    main()
