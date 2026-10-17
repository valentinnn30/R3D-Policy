"""Per-point semantic labels for the head-camera point cloud.

The analysis this feeds (where do the encoder's FPS tokens land?) needs to know,
for every stored point, what it is: table residue, arm, gripper, the object being
moved, the target, ... SAPIEN's actor-level segmentation gives a per-pixel entity
id (`per_scene_id`); `Camera.get_pcd(return_actor_ids=True)` carries that id
through the same valid / crop / FPS indexing as the colour, and this module maps
ids to the fixed classes below.

The lookup is built ONCE per episode, after `load_actors` -- entities do not change
during an episode. Task roles (which actor is "object", which is "target") come from
the task's `point_label_roles()`; anything else that is not robot / table / wall is
OTHER.
"""

import json

import numpy as np

PAD = 0          # zero-padding added by get_pcd when the crop leaves < N points
TABLE = 1
BACKGROUND = 2   # wall, ground, or a pixel with no entity
ROBOT_BODY = 3   # robot links shared by both arms (base, torso, ...)
LEFT_ARM = 4
RIGHT_ARM = 5
LEFT_GRIPPER = 6
RIGHT_GRIPPER = 7
OBJECT = 8       # the actor being picked
TARGET = 9       # where it goes
OTHER = 10       # any other actor (distractors, clutter)

CLASS_NAMES = {
    PAD: "pad",
    TABLE: "table",
    BACKGROUND: "background",
    ROBOT_BODY: "robot_body",
    LEFT_ARM: "left_arm",
    RIGHT_ARM: "right_arm",
    LEFT_GRIPPER: "left_gripper",
    RIGHT_GRIPPER: "right_gripper",
    OBJECT: "object",
    TARGET: "target",
    OTHER: "other",
}
CLASS_NAMES_JSON = json.dumps({str(k): v for k, v in CLASS_NAMES.items()})


def _entities_of(obj):
    """Every sapien Entity behind a RoboTwin Actor / ArticulationActor / Entity /
    PhysxArticulation."""
    obj = getattr(obj, "actor", obj)  # unwrap RoboTwin's Actor wrappers
    if hasattr(obj, "get_links"):  # articulation: one entity per link
        return [link.entity for link in obj.get_links()]
    return [obj]


def _subtree(link):
    out, stack = [], [link]
    while stack:
        l = stack.pop()
        out.append(l)
        stack.extend(l.get_children())
    return out


def _ancestors(link):
    out = []
    while link is not None:
        out.append(link)
        link = link.get_parent()
    return out


def _gripper_root(articulation, gripper_name):
    """The hand body: parent link of the gripper's base (driven) joint."""
    joint = articulation.find_joint_by_name(gripper_name["base"])
    if joint is None:
        raise KeyError(f"gripper base joint {gripper_name['base']!r} not in articulation")
    return joint.get_parent_link()


def build_label_lut(task):
    """{per_scene_id: class} for the current scene of `task` (a Base_Task)."""
    lut = {}

    def assign(objs, cls):
        for o in objs:
            for e in _entities_of(o):
                lut[int(e.per_scene_id)] = cls

    for e in task.scene.get_entities():
        lut[int(e.per_scene_id)] = OTHER

    robot = task.robot
    sides = {}
    for side, art, gname in (("left", robot.left_entity, robot.left_gripper_name),
                             ("right", robot.right_entity, robot.right_gripper_name)):
        hand = _gripper_root(art, gname)
        sides[side] = (art, _subtree(hand), _ancestors(hand))

    # Robot links default to ROBOT_BODY; each arm chain is the hand's ancestors that
    # the OTHER hand does not share (aloha-agilex is one articulation for both arms,
    # so the shared trunk would otherwise be claimed by whichever side came last).
    for art, _, _ in sides.values():
        assign([art], ROBOT_BODY)
    left_anc = {id(l) for l in sides["left"][2]}
    right_anc = {id(l) for l in sides["right"][2]}
    for side, arm_cls, grip_cls, other_anc in (("left", LEFT_ARM, LEFT_GRIPPER, right_anc),
                                               ("right", RIGHT_ARM, RIGHT_GRIPPER, left_anc)):
        _, grip_links, anc = sides[side]
        for l in anc:
            if id(l) not in other_anc:
                lut[int(l.entity.per_scene_id)] = arm_cls
        for l in grip_links:
            lut[int(l.entity.per_scene_id)] = grip_cls

    assign([task.table], TABLE)
    assign([task.wall], BACKGROUND)
    roles = task.point_label_roles() if hasattr(task, "point_label_roles") else {}
    assign(roles.get("object", []), OBJECT)
    assign(roles.get("target", []), TARGET)
    return lut


def ids_to_labels(actor_ids, lut):
    """Map per-point entity ids (int, -1 = padding) to classes (int16)."""
    actor_ids = np.asarray(actor_ids)
    labels = np.full(actor_ids.shape, BACKGROUND, dtype=np.int16)  # id 0 / unknown
    for pid, cls in lut.items():
        labels[actor_ids == pid] = cls
    labels[actor_ids < 0] = PAD
    return labels
