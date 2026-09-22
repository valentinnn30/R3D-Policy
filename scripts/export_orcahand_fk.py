#!/usr/bin/env python3
"""Export the orca hand's fingertip kinematic chains + the link8 mount to yaml.

Input: the two URDFs rendered from the LIVE hand model,

    source ~/ros2_ws/install/setup.bash
    X=~/ros2_ws/src/faive_system/src/viz/models/orcahand_v1b/orcahand_v1b.urdf.xacro
    xacro $X hand_chirality:=left  > /tmp/orca_left.urdf
    xacro $X hand_chirality:=right > /tmp/orca_right.urdf

Output: R3D/r3d/config/preprocess/orcahand_v1b_fk.yaml, read by
`real_preprocess.HandFK`. The yaml, not the xacro, is what training and
inference load, so neither needs ROS.

The mount T_link8<-hand_root is ANALYTIC, not fitted (decided 2026-09-21):

  * rotation -- at the init pose the EE orientation is diag(1,-1,-1) in world
    (INIT_ARM_POSE in r3d_inference_node.py) and link8->EE carries no rotation,
    so R_world<-link8(init) = diag(1,-1,-1). There the fingers point along world
    +x and the palm faces world -z (back of the hand up). In the hand model +z
    is the finger direction and +y the palm normal (thumb at +y), so
    R_world<-hand = [x_h, y_h, z_h] = [-y_w, -z_w, +x_w].
  * translation -- the Desk EE point link8->EE = [130, 0, 70] mm is placed on
    a reference point P of the hand model (wrist = 0), t = EE - R @ P.
    Default `--ee-reference knuckle_tower` (re-measured on the hand,
    2026-09-22):
      - along the fingers (hand z -> link8 x, the 130 mm): the KNUCKLE of the
        middle finger, i.e. the origin of joint `--knuckle-joint`
        (middle_mcp; middle_abd sits 20 mm closer to the palm);
      - palm normal (hand y -> link8 z, the 70 mm): the CENTRE of the big
        forearm block, i.e. the centre of the `tower` link's main visual
        mesh bounding box -- through its thickness, not its surface;
      - lateral (hand x -> link8 y, the 0): the middle finger's line (same
        joint), which is also where the palm sits.
    `--ee-reference palm` reproduces the first version (P = palm link origin),
    which put the model ~2-3 cm too far out along the fingers.

It is verified visually, not numerically: scripts/fingertip_occupancy.py
writes .ply frames with the FK fingertips on the real cloud.
"""

import argparse
import datetime
import re
import subprocess
import xml.etree.ElementTree as ET

import numpy as np
import yaml

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "R3D"))
from r3d.common.real_preprocess import (  # noqa: E402
    HAND_FINGERS, HAND_JOINT_ORDER, HandFK)

JOINT_ORDER = list(HAND_JOINT_ORDER)
FINGERS = list(HAND_FINGERS)
EE_IN_LINK8 = np.array([0.130, 0.0, 0.070])
R_WORLD_LINK8_INIT = np.diag([1.0, -1.0, -1.0])
# columns = hand x, y, z expressed in world at the init pose
R_WORLD_HAND_INIT = np.array([[0.0, 0.0, 1.0],
                              [-1.0, 0.0, 0.0],
                              [0.0, -1.0, 0.0]])


def _chain(root, link):
    """Joints from the URDF root to `link`, root first."""
    by_child = {j.find("child").get("link"): j for j in root.findall("joint")}
    out = []
    while link in by_child:
        j = by_child[link]
        out.append(j)
        link = j.find("parent").get("link")
    if link != "hand_root":
        raise ValueError(f"chain ends at {link!r}, expected hand_root")
    return out[::-1]


VIZ = os.path.expanduser("~/ros2_ws/src/faive_system/src/viz")
XACRO = os.path.join(VIZ, "models/orcahand_v1b/orcahand_v1b.urdf.xacro")
XACRO_BIN = os.path.expanduser("~/micromamba/envs/ros_env/bin/xacro")


def _render(side):
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    return subprocess.check_output([XACRO_BIN, XACRO, f"hand_chirality:={side}"],
                                   text=True, env=env)


def _joint_origin(root, joint_name):
    """Position of a joint's origin in hand_root at q = 0."""
    j = next(j for j in root.findall("joint") if j.get("name") == joint_name)
    chain = [_entry(x) for x in _chain(root, j.find("child").get("link"))]
    return HandFK._eval_chain(chain, np.zeros(len(JOINT_ORDER)))[:3, 3]


def _tower_block_centre(root, visual_name="visual_tower_main_mesh"):
    """Centre of the forearm block (tower main visual mesh bbox) in hand_root."""
    import trimesh
    link = next(l for l in root.findall("link") if l.get("name") == "tower")
    vis = next(v for v in link.findall("visual") if v.get("name") == visual_name)
    m = vis.find("geometry/mesh")
    tm = trimesh.load(re.sub(r"^package://viz/", VIZ + "/", m.get("filename")), force="mesh")
    tm.vertices = tm.vertices * np.array([float(v) for v in m.get("scale", "1 1 1").split()])
    o = vis.find("origin")
    V = np.eye(4)
    if o is not None:
        from r3d.common.real_preprocess import _rpy_matrix
        V[:3, :3] = _rpy_matrix([float(v) for v in o.get("rpy", "0 0 0").split()])
        V[:3, 3] = [float(v) for v in o.get("xyz", "0 0 0").split()]
    tower = HandFK._eval_chain([_entry(x) for x in _chain(root, "tower")],
                               np.zeros(len(JOINT_ORDER)))
    tm.apply_transform(tower @ V)
    lo, hi = tm.bounds
    return (lo + hi) / 2


def _entry(j):
    o = j.find("origin")
    xyz = [0.0, 0.0, 0.0] if o is None else [float(v) for v in o.get("xyz", "0 0 0").split()]
    rpy = [0.0, 0.0, 0.0] if o is None else [float(v) for v in o.get("rpy", "0 0 0").split()]
    jt = j.get("type")
    if jt not in ("fixed", "revolute"):
        raise ValueError(f"joint {j.get('name')}: unsupported type {jt}")
    if j.find("mimic") is not None:
        raise ValueError(f"joint {j.get('name')}: mimic joints unsupported")
    e = {"xyz": xyz, "rpy": rpy}
    if jt == "revolute":
        name = j.get("name")
        if name not in JOINT_ORDER:
            raise ValueError(f"revolute joint {name!r} not in the controller order")
        e["joint"] = name
        e["axis"] = [float(v) for v in j.find("axis").get("xyz").split()]
    return e


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--left-urdf", default=None,
                    help="pre-rendered URDF; default: render the xacro with ros_env's xacro")
    ap.add_argument("--right-urdf", default=None)
    ap.add_argument("--ee-reference", default="knuckle_tower",
                    choices=["knuckle_tower", "palm"])
    ap.add_argument("--knuckle-joint", default="middle_mcp",
                    help="joint whose origin is 'the knuckle' (middle_mcp or middle_abd)")
    ap.add_argument("--out", default="R3D/r3d/config/preprocess/orcahand_v1b_fk.yaml")
    args = ap.parse_args()

    R = R_WORLD_LINK8_INIT.T @ R_WORLD_HAND_INIT
    assert np.allclose(R @ R.T, np.eye(3)) and np.isclose(np.linalg.det(R), 1.0)

    hands = {}
    for side, path in (("left", args.left_urdf), ("right", args.right_urdf)):
        root = ET.fromstring(open(path).read() if path else _render(side))
        chains = {f: [_entry(j) for j in _chain(root, f"{f}_fingertip")]
                  for f in FINGERS}
        palm_chain = [_entry(j) for j in _chain(root, "palm")]
        palm = HandFK._eval_chain(palm_chain, np.zeros(len(JOINT_ORDER)))[:3, 3]
        if args.ee_reference == "palm":
            P = palm
            ref = {"name": "palm", "point_in_hand_root": palm.round(9).tolist()}
        else:
            knuckle = _joint_origin(root, args.knuckle_joint)
            block = _tower_block_centre(root)
            # hand x: lateral (-> link8 y), hand y: palm normal (-> link8 z),
            # hand z: finger direction (-> link8 x)
            P = np.array([knuckle[0], block[1], knuckle[2]])
            ref = {"name": "knuckle_tower", "knuckle_joint": args.knuckle_joint,
                   "knuckle_in_hand_root": knuckle.round(9).tolist(),
                   "tower_block_centre_in_hand_root": block.round(9).tolist(),
                   "point_in_hand_root": P.round(9).tolist()}
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = EE_IN_LINK8 - R @ P
        hands[side] = {"mount_link8": T.round(9).tolist(),
                       "ee_reference": ref,
                       "palm_in_hand_root": palm.round(9).tolist(),
                       "tips": chains}
        print(f"{side}: EE reference ({args.ee_reference}) in hand_root "
              f"{np.round(P * 1000, 1)} mm -> hand_root in link8 "
              f"{np.round(T[:3, 3] * 1000, 1)} mm")

    doc = {
        "generated": datetime.date.today().isoformat(),
        "source": "ros2_ws/src/faive_system/src/viz/models/orcahand_v1b/"
                  "orcahand_v1b.urdf.xacro, rendered per hand_chirality",
        "mount": f"analytic: init-pose orientation + Desk EE [130,0,70] mm placed on "
                 f"the '{args.ee_reference}' reference point; see scripts/export_orcahand_fk.py",
        "joint_order": JOINT_ORDER,
        "fingers": FINGERS,
        "hands": hands,
    }
    with open(args.out, "w") as f:
        yaml.safe_dump(doc, f, sort_keys=False, default_flow_style=None)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
