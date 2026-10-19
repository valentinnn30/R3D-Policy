#!/usr/bin/env python
"""Head-camera view check for the sim baseline: is the arm in view, what does the
crop cut, where do the 128 tokens land?  Run in the r3d_sim env.

    python scripts/sim/check_head_view.py                      # collect + report
    python scripts/sim/check_head_view.py --tasks place_empty_cup_right --episodes 10
    python scripts/sim/check_head_view.py --report-only        # re-analyse existing runs
    python scripts/sim/check_head_view.py --tasks stack_blocks_two_fov --task-config sim_stock_p1024

Collection is the REAL RoboTwin path (`script/collect_data.py:run` with the
`sim_head_p1024` config), so the stored clouds and labels are exactly what data
collection writes. The only addition, in this process only, is a per-frame
`view_stats` record taken from the same head-camera picture:

    class_px     (C,)  valid pixels per semantic class, whole image
    class_inbox  (C,)  of those, inside the crop box
    link_px      (L,)  pixels per robot link, whole image
    link_inbox   (L,)  of those, inside the crop box
    link_inimg   (L,)  link origin projects inside the image (frustum check)
    obj_px       (O,)  pixels per task object (point_label_roles()["object"]), whole image
    obj_inbox    (O,)  of those, inside the crop box -- per object, so "every cube
                       visible in every frame" is checked per cube

so "lost to the box" and "lost to the frustum" can be told apart.

Outputs under R3D/data/outputs/sim_view/<task>/: the hdf5 episodes, report.json,
head rgb + label-coloured PLY at four phases.
"""

import argparse
import glob
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
R3D_DIR = os.path.join(ROOT, "R3D")
RT_DIR = os.path.join(R3D_DIR, "r3d", "env", "robotwin2")
OUT_DIR = os.path.join(R3D_DIR, "data", "outputs", "sim_view")
TASK_CONFIG = "sim_head_p1024"  # default; --task-config overrides


# --------------------------------------------------------------------------- collect
def _robot_links(task):
    seen, links = set(), []
    for art in (task.robot.left_entity, task.robot.right_entity):
        for l in art.get_links():
            if id(l) not in seen:
                seen.add(id(l))
                links.append(l)
    return links


def _view_stats(task, point_labels):
    cams = task.cameras
    cam = cams.static_camera_list[cams.head_camera_id]
    pos = cam.get_picture("Position")
    seg = cam.get_picture("Segmentation")[..., 1].astype(np.int64)
    M = cam.get_model_matrix()
    valid = pos[..., 3] < 1
    pts = pos[..., :3][valid] @ M[:3, :3].T + M[:3, 3]
    ids = seg[valid]
    lo, hi = np.asarray(cams.pcd_crop_bbox[0]), np.asarray(cams.pcd_crop_bbox[1])
    inbox = np.all((pts >= lo) & (pts <= hi), axis=1)
    lab = point_labels.ids_to_labels(ids, task._point_label_lut)
    C = len(point_labels.CLASS_NAMES)

    links = _robot_links(task)
    H, W = pos.shape[:2]
    K, E = cam.get_intrinsic_matrix(), cam.get_extrinsic_matrix()  # cv: world -> cam
    link_px, link_inbox, link_inimg = [], [], []
    for l in links:
        m = ids == l.entity.per_scene_id
        link_px.append(m.sum())
        link_inbox.append((m & inbox).sum())
        pc = E[:3, :3] @ l.entity.get_pose().p + E[:3, 3]
        u, v, z = (K @ pc)
        link_inimg.append(bool(pc[2] > 0 and 0 <= u / z < W and 0 <= v / z < H))
    objs = [e for o in task.point_label_roles().get("object", []) for e in point_labels._entities_of(o)]
    obj_px = [int((ids == e.per_scene_id).sum()) for e in objs]
    obj_inbox = [int(((ids == e.per_scene_id) & inbox).sum()) for e in objs]
    return {
        "obj_px": np.asarray(obj_px),
        "obj_inbox": np.asarray(obj_inbox),
        "class_px": np.bincount(lab, minlength=C),
        "class_inbox": np.bincount(lab[inbox], minlength=C),
        "link_px": np.asarray(link_px),
        "link_inbox": np.asarray(link_inbox),
        "link_inimg": np.asarray(link_inimg),
    }


def collect(task_name, n_episodes, task_config=TASK_CONFIG):
    os.chdir(RT_DIR)
    # same search path RoboTwin2EnvManager sets up: envs/ is imported both as a
    # package and flat (envs/utils/get_camera_config.py: `from _GLOBAL_CONFIGS import`)
    sys.path[:0] = [RT_DIR, os.path.join(RT_DIR, "envs"), os.path.join(RT_DIR, "description", "utils"),
                    R3D_DIR, os.path.join(RT_DIR, "script")]
    import yaml
    import collect_data as cd
    from envs import _base_task
    from envs.utils import point_labels
    from envs._GLOBAL_CONFIGS import CONFIGS_PATH

    orig_get_obs = _base_task.Base_Task.get_obs

    def get_obs(self):
        pkl = orig_get_obs(self)
        pkl["view_stats"] = _view_stats(self, point_labels)
        if not hasattr(self, "_view_links_written"):
            names = [l.get_name() for l in _robot_links(self)]
            with open(os.path.join(self.save_dir, "link_names.json"), "w") as f:
                json.dump(names, f)
            self._view_links_written = True
        return pkl

    _base_task.Base_Task.get_obs = get_obs

    with open(os.path.join(CONFIGS_PATH, f"{task_config}.yml")) as f:
        args = yaml.safe_load(f)
    args["task_name"] = task_name
    args["task_config"] = task_config
    args["episode_num"] = n_episodes
    args["language_num"] = 1
    with open(os.path.join(CONFIGS_PATH, "_embodiment_config.yml")) as f:
        emb_types = yaml.safe_load(f)
    (emb,) = args["embodiment"]
    robot_file = emb_types[emb]["file_path"]
    args["left_robot_file"] = args["right_robot_file"] = robot_file
    args["dual_arm_embodied"] = True
    args["left_embodiment_config"] = args["right_embodiment_config"] = cd.get_embodiment_config(robot_file)
    args["embodiment_name"] = emb
    args["save_path"] = os.path.join(OUT_DIR, task_name)
    cd.run(cd.class_decorator(task_name), args)


# --------------------------------------------------------------------------- report
def load_point_labels():
    """envs/utils/point_labels.py by file path: importing it through the `envs`
    package would import all of RoboTwin, which reads assets relative to cwd."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "point_labels", os.path.join(RT_DIR, "envs", "utils", "point_labels.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _load_episode(path):
    import h5py
    with h5py.File(path, "r") as f:
        ep = {
            "pc": f["pointcloud"][()].astype(np.float32),
            "label": f["pointcloud_label"][()].astype(np.int64),
            "rgb": f["observation/head_camera/rgb"][()],
        }
        for k in f["view_stats"]:
            ep[k] = f["view_stats"][k][()]
    return ep


def _save_ply(path, xyz, labels):
    import open3d as o3d
    palette = np.array([[0, 0, 0], [.6, .45, .3], [.5, .5, .5], [.3, .3, .3], [0, .4, 1], [1, .3, 0],
                        [0, .8, 1], [1, .8, 0], [1, 0, 0], [0, .9, 0], [.8, 0, .8]])
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz.astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector(palette[labels])
    o3d.io.write_point_cloud(path, pcd)


def report(task_name, num_group, group_size, n_draws, task_config=TASK_CONFIG):
    import cv2
    sys.path.insert(0, R3D_DIR)
    pl = load_point_labels()
    CLASS_NAMES, PAD = pl.CLASS_NAMES, pl.PAD
    RIGHT_ARM, RIGHT_GRIPPER, OBJECT, TARGET = pl.RIGHT_ARM, pl.RIGHT_GRIPPER, pl.OBJECT, pl.TARGET
    PHASE_CLASSES = (pl.LEFT_ARM, RIGHT_ARM, pl.LEFT_GRIPPER, RIGHT_GRIPPER, OBJECT, TARGET)
    ARM_CLASSES = [pl.ROBOT_BODY, pl.LEFT_ARM, pl.RIGHT_ARM]
    GRIPPER_CLASSES = [pl.LEFT_GRIPPER, pl.RIGHT_GRIPPER]
    from r3d.common.token_allocation import box_normalize, token_allocation
    import yaml

    tdir = os.path.join(OUT_DIR, task_name, task_config) if os.path.isdir(
        os.path.join(OUT_DIR, task_name, task_config)) else os.path.join(OUT_DIR, task_name)
    files = sorted(glob.glob(os.path.join(tdir, "data", "episode*.hdf5")),
                   key=lambda p: int(os.path.basename(p)[7:-5]))
    if not files:
        raise FileNotFoundError(f"no episodes under {tdir}/data")
    with open(os.path.join(RT_DIR, "task_config", f"{task_config}.yml")) as f:
        bbox = yaml.safe_load(f)["bbox"]
    with open(os.path.join(tdir, "link_names.json")) as f:
        link_names = json.load(f)
    seeds = [int(s) for s in open(os.path.join(tdir, "seed.txt")).read().split()]
    C = len(CLASS_NAMES)
    names = [CLASS_NAMES[i] for i in range(C)]

    acc = {k: [] for k in ("class_px", "class_inbox", "stored", "centres", "slots", "coverage",
                           "phase", "link_px", "link_inbox", "link_inimg", "obj_px", "obj_inbox")}
    pad_frames, zmax, n_frames = 0, [], 0
    img_dir = os.path.join(tdir, "frames")
    os.makedirs(img_dir, exist_ok=True)
    for ei, path in enumerate(files):
        ep = _load_episode(path)
        T = len(ep["pc"])
        n_frames += T
        pad_frames += int((ep["label"] == PAD).any(1).sum())
        zmax.append(ep["pc"][..., 2].max(1))
        xyz_n = box_normalize(ep["pc"][..., :3], bbox)
        tok = [token_allocation(xyz_n, ep["label"], C, num_group, group_size) for _ in range(n_draws)]
        for k in ("centres", "slots", "coverage"):
            acc[k].append(np.nanmean(np.stack([t[k] for t in tok]), 0))
        acc["stored"].append(tok[0]["points"])
        for k in ("class_px", "class_inbox", "link_px", "link_inbox", "link_inimg", "obj_px", "obj_inbox"):
            acc[k].append(ep[k])
        acc["phase"].append(np.arange(T) / max(T - 1, 1))
        if ei == 0:
            for q in (0.0, 0.33, 0.66, 1.0):
                t = int(round(q * (T - 1)))
                img = cv2.imdecode(np.frombuffer(ep["rgb"][t], np.uint8), cv2.IMREAD_COLOR)
                cv2.imwrite(os.path.join(img_dir, f"ep0_t{t:03d}_rgb.jpg"), img)
                _save_ply(os.path.join(img_dir, f"ep0_t{t:03d}_labels.ply"), ep["pc"][t, :, :3], ep["label"][t])
    A = {k: np.concatenate(v, 0) for k, v in acc.items()}
    zmax = np.concatenate(zmax)

    def by_class(x):
        return {names[i]: float(np.nanmean(x[:, i])) for i in range(C)}

    phase_bins = [(0, .25), (.25, .5), (.5, .75), (.75, 1.01)]
    per_phase = {}
    for a, b in phase_bins:
        m = (A["phase"] >= a) & (A["phase"] < b)
        per_phase[f"{a:.2f}-{min(b, 1):.2f}"] = {
            "centres": {names[i]: float(A["centres"][m, i].mean()) for i in PHASE_CLASSES},
            "stored_points": {names[i]: float(A["stored"][m, i].mean()) for i in PHASE_CLASSES},
        }
    links = {}
    for j, n in enumerate(link_names):
        links[n] = {
            "frames_in_image_frac": float(A["link_inimg"][:, j].mean()),
            "frames_visible_frac": float((A["link_px"][:, j] > 0).mean()),
            "frames_in_box_frac": float((A["link_inbox"][:, j] > 0).mean()),
            "mean_px": float(A["link_px"][:, j].mean()),
            "mean_px_in_box": float(A["link_inbox"][:, j].mean()),
        }
    rep = {
        "task": task_name,
        "episodes": len(files),
        "frames": n_frames,
        "expert_success": f"{len(seeds)}/{max(seeds) + 1} seeds",
        "num_group": num_group, "group_size": group_size, "fps_draws_per_frame": n_draws,
        "frames_with_padding": pad_frames,
        "stored_z_max_m": {"p50": float(np.median(zmax)), "p99": float(np.percentile(zmax, 99)),
                           "max": float(zmax.max())},
        "right_gripper_points_ge10_frac": float((A["stored"][:, RIGHT_GRIPPER] >= 10).mean()),
        "any_gripper_points_ge10_frac": float((A["stored"][:, [pl.LEFT_GRIPPER, RIGHT_GRIPPER]].max(1) >= 10).mean()),
        "frames_per_episode": {"mean": float(n_frames / len(files)),
                               "min": int(min(len(p) for p in acc["phase"])), "max": int(max(len(p) for p in acc["phase"]))},
        # per task object (e.g. each cube): pixels in the head image / inside the box
        "per_object_px_min": [int(v) for v in A["obj_px"].min(0)],
        "per_object_inbox_min": [int(v) for v in A["obj_inbox"].min(0)],
        "frames_any_object_inbox_lt30": float((A["obj_inbox"].min(1) < 30).mean()),
        # the object leaving the head camera's FRUSTUM (not the crop box) shows up here:
        # at x >~ 0.24 the stock D435 sees only the cup's rim (2026-10-17)
        "object_points_lt50_frac": float((A["stored"][:, OBJECT] < 50).mean()),
        "object_points_lt50_episodes": [int(i) for i, e in enumerate(acc["stored"]) if np.median(e[:, OBJECT]) < 50],
        # share of the num_group token centres on robot points, mean over frames:
        # what the token-allocation method could hand back to the object
        "robot_token_share": {
            "arm": float(A["centres"][:, ARM_CLASSES].sum(1).mean() / num_group),
            "arm+gripper": float(A["centres"][:, ARM_CLASSES + GRIPPER_CLASSES].sum(1).mean() / num_group),
            "arm+gripper_per_phase": {
                f"{a:.2f}-{min(b, 1):.2f}": float(A["centres"][(A["phase"] >= a) & (A["phase"] < b)][
                    :, ARM_CLASSES + GRIPPER_CLASSES].sum(1).mean() / num_group) for a, b in phase_bins},
        },
        "image_px_per_class": by_class(A["class_px"]),
        "in_box_px_per_class": by_class(A["class_inbox"]),
        "stored_points_per_class": by_class(A["stored"]),
        "token_centres_per_class": by_class(A["centres"]),
        "patch_slot_share_per_class": by_class(A["slots"]),
        "point_coverage_per_class": by_class(A["coverage"]),
        "per_phase": per_phase,
        "robot_links": links,
    }
    with open(os.path.join(tdir, f"report_g{num_group}.json"), "w") as f:
        json.dump(rep, f, indent=2)
    _print_report(rep, names)
    print(f"\nwritten: {tdir}/report_g{num_group}.json, frames/ (rgb + label PLY, ep0)")


def _print_report(r, names):
    print(f"\n==== {r['task']}: {r['episodes']} eps, {r['frames']} frames, expert {r['expert_success']}")
    print(f"frames with zero-padding: {r['frames_with_padding']}   "
          f"stored z max p50/p99/max: {r['stored_z_max_m']['p50']:.3f}/{r['stored_z_max_m']['p99']:.3f}/"
          f"{r['stored_z_max_m']['max']:.3f} m   right gripper >=10 pts: {r['right_gripper_points_ge10_frac']:.1%}")
    rs = r["robot_token_share"]
    print(f"robot share of the {r['num_group']} tokens: arm {rs['arm']:.1%}, arm+gripper {rs['arm+gripper']:.1%}"
          "  | per phase: " + "  ".join(f"{k} {v:.0%}" for k, v in rs["arm+gripper_per_phase"].items()))
    print(f"frames/episode {r['frames_per_episode']}  | per-object min px (image / in box): "
          f"{r['per_object_px_min']} / {r['per_object_inbox_min']}  | frames with an object < 30 px in box: "
          f"{r['frames_any_object_inbox_lt30']:.1%}  | any gripper >= 10 pts: {r['any_gripper_points_ge10_frac']:.1%}")
    print(f"object < 50 pts: {r['object_points_lt50_frac']:.1%} of frames; "
          f"episodes with median < 50: {r['object_points_lt50_episodes']}")
    hdr = f"{'class':<14}{'img px':>9}{'in box':>9}{'stored':>8}{'centres':>9}{'slots':>7}{'cover':>7}"
    print(hdr)
    for n in names[1:]:
        print(f"{n:<14}{r['image_px_per_class'][n]:>9.0f}{r['in_box_px_per_class'][n]:>9.0f}"
              f"{r['stored_points_per_class'][n]:>8.1f}{r['token_centres_per_class'][n]:>9.2f}"
              f"{r['patch_slot_share_per_class'][n]:>7.1%}{r['point_coverage_per_class'][n]:>7.0%}")
    print("\nper phase (token centres | stored points):")
    for ph, d in r["per_phase"].items():
        print(f"  {ph}: " + "  ".join(f"{k} {d['centres'][k]:.1f}|{d['stored_points'][k]:.0f}" for k in d["centres"]))
    print("\nrobot links (in image / visible / in box, frac of frames; mean px in box):")
    for n, d in r["robot_links"].items():
        print(f"  {n:<28}{d['frames_in_image_frac']:>6.0%}{d['frames_visible_frac']:>6.0%}"
              f"{d['frames_in_box_frac']:>6.0%}{d['mean_px_in_box']:>9.0f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tasks", nargs="+", default=["place_empty_cup_right"])
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--task-config", default=TASK_CONFIG, help="RoboTwin task config (task_config/<name>.yml)")
    ap.add_argument("--num-group", type=int, nargs="+", default=[128, 64, 32],
                    help="token counts to report (the planned sweep)")
    ap.add_argument("--group-size", type=int, default=32)
    ap.add_argument("--fps-draws", type=int, default=4)
    ap.add_argument("--report-only", action="store_true")
    ap.add_argument("--collect-only", action="store_true")
    a = ap.parse_args()
    for t in a.tasks:
        if not a.report_only:
            # one task per process: sapien scenes and the get_obs patch are process-global
            import subprocess
            t0 = time.time()
            subprocess.run([sys.executable, __file__, "--_collect", t, str(a.episodes), a.task_config], check=True)
            print(f"[{t}] collected in {time.time() - t0:.0f}s")
        if not a.collect_only:
            for g in a.num_group:
                report(t, g, a.group_size, a.fps_draws, a.task_config)


if __name__ == "__main__":
    if len(sys.argv) == 5 and sys.argv[1] == "--_collect":
        import torch.multiprocessing as mp
        mp.set_start_method("spawn", force=True)
        collect(sys.argv[2], int(sys.argv[3]), sys.argv[4])
    else:
        main()
