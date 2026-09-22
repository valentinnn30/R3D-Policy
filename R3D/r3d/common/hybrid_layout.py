"""Block layout and pose resolution for the hybrid 2D/3D policies.

Plan: ~/ros2_ws/hybrid_2d3d_ablations_plan.txt

A hybrid observation is a list of camera BLOCKS, each produced by exactly the
pipeline its modality uses on its own:

    kind "pc"        one unfused camera-frame cloud   (RealMultiCamUnfusedDataset)
    kind "pc_fused"  the listed cameras fused in world frame (RealMultiCamDataset)
    kind "image"     one DINOv2 image                 (RealMultiCamImageDataset)

This module is the ONE place that decides the block order and what each
block's pose tag carries. The dataset, the encoder, the policy and the
on-robot server all call it, so none of them can order or tag the blocks
differently from the others -- a mismatch there would not raise, it would just
attach a pose to the wrong tokens.

Pose resolution reuses the per-camera rule the single-modality pipelines
already apply (`dp3.py` / `dp3_image.py`), with one source PER MODALITY:

    extrinsic -> every unfused/image block has a pose (static cams: a constant)
    proprio   -> only blocks whose camera rides on an arm (camera_arm_slice)
    none      -> no block has a pose
    fused     -> never a pose: the cloud is already in the world frame

The existing single-modality guard ("proprio but no camera has an arm slice ->
that is the 'none' control") becomes POLICY-WIDE here. One modality may
legitimately hold only the static top camera, which under 'proprio' resolves to
its identity embedding exactly as it did in Ablation A'; only when NO block
anywhere ends up with a pose is the run secretly the all-'none' control.
"""

from typing import Dict, List, Optional, Sequence

POSE_SOURCES = ("extrinsic", "proprio", "none")


def hybrid_blocks(cameras_3d: Sequence[int], cameras_2d: Sequence[int],
                  fuse_3d: bool = False) -> List[Dict]:
    """Blocks in their canonical order: ascending lowest camera index.

    Each block: {kind, key, cameras, campose_key}. `campose_key` is None for a
    fused block, which carries no pose. Keys use GLOBAL camera indices.
    """
    c3 = [int(c) for c in cameras_3d]
    c2 = [int(c) for c in cameras_2d]
    if not c3 or not c2:
        raise ValueError(
            f"a hybrid needs both modalities, got cameras_3d={c3} cameras_2d={c2}")
    if set(c3) & set(c2):
        raise ValueError(f"camera(s) {sorted(set(c3) & set(c2))} are in both modalities")
    for name, c in (("cameras_3d", c3), ("cameras_2d", c2)):
        if c != sorted(set(c)):
            raise ValueError(f"{name} must be unique and ascending, got {c}")

    blocks = []
    if fuse_3d:
        if len(c3) < 2:
            raise ValueError(f"fuse_3d needs at least two 3D cameras, got {c3}")
        blocks.append({"kind": "pc_fused", "key": "point_cloud",
                       "cameras": c3, "campose_key": None})
    else:
        for c in c3:
            blocks.append({"kind": "pc", "key": f"point_cloud_cam{c}",
                           "cameras": [c], "campose_key": f"campose_cam{c}"})
    for c in c2:
        blocks.append({"kind": "image", "key": f"image_cam{c}",
                       "cameras": [c], "campose_key": f"campose_cam{c}"})
    blocks.sort(key=lambda b: min(b["cameras"]))
    return blocks


def blocks_from_obs_keys(obs_keys, fused_cameras: Optional[Sequence[int]] = None):
    """Recover the block list from a shape_meta's observation keys.

    `fused_cameras` must be given when the shape_meta holds a fused
    `point_cloud` -- the key alone does not say which cameras went into it.
    """
    keys = list(obs_keys)
    c2 = sorted(int(k[len("image_cam"):]) for k in keys if k.startswith("image_cam"))
    c3 = sorted(int(k[len("point_cloud_cam"):]) for k in keys
                if k.startswith("point_cloud_cam"))
    fused = "point_cloud" in keys
    if fused:
        if c3:
            raise ValueError("shape_meta has both a fused point_cloud and point_cloud_cam*")
        if not fused_cameras:
            raise ValueError("a fused point_cloud needs `fused_cameras` to place its block")
        c3 = [int(c) for c in fused_cameras]
    return hybrid_blocks(c3, c2, fuse_3d=fused)


def _modality(block):
    return "2d" if block["kind"] == "image" else "3d"


def resolve_block_poses(blocks: List[Dict], pose_source_3d: str,
                        pose_source_2d: str, camera_arm_slice=None) -> List[Dict]:
    """Per block: {modality, source, has_pose, arm_slice}. Raises on a degenerate run.

    `camera_arm_slice` is indexed by GLOBAL camera index ([null, [0,7], [7,14]]).
    """
    for name, s in (("pose_source_3d", pose_source_3d), ("pose_source_2d", pose_source_2d)):
        if s not in POSE_SOURCES:
            raise ValueError(f"{name} must be one of {POSE_SOURCES}, got {s!r}")
    n_global = 1 + max(c for b in blocks for c in b["cameras"])
    if camera_arm_slice is None:
        camera_arm_slice = [None] * n_global
    if len(camera_arm_slice) < n_global:
        raise ValueError(
            f"camera_arm_slice has {len(camera_arm_slice)} entries but camera "
            f"index {n_global - 1} is used; it is indexed by GLOBAL camera index")

    out = []
    for b in blocks:
        mod = _modality(b)
        src = pose_source_3d if mod == "3d" else pose_source_2d
        if b["kind"] == "pc_fused":
            if src != "none":
                raise ValueError(
                    f"pose_source_3d={src!r} with fused 3D cameras {b['cameras']}: a "
                    "fused cloud is already in the world frame and carries no pose "
                    "tag (the fused run had no pose source). Set it to 'none'.")
            out.append({"modality": mod, "source": src, "has_pose": False,
                        "arm_slice": None})
            continue
        (cam,) = b["cameras"]
        sl = camera_arm_slice[cam]
        sl = None if sl is None else (int(sl[0]), int(sl[1]))
        if src == "extrinsic":
            has = True
        elif src == "proprio":
            has = sl is not None
        else:
            has = False
        out.append({"modality": mod, "source": src, "has_pose": has, "arm_slice": sl})

    used = {r["source"] for r in out}
    if "proprio" in used and not any(r["has_pose"] for r in out):
        raise ValueError(
            "pose_source 'proprio' is set but NO block ends up with a pose: every "
            "proprio camera is static (no arm slice) and every other block is "
            "untagged. That is the all-'none' control -- run it as "
            "pose_source_3d=none pose_source_2d=none so the run says what it is.")
    return out


def modality_pose_config(blocks, resolved, modality: str):
    """(n_blocks, per_camera_mlp, has_pose list) for ONE modality's CameraPoseEmbedding.

    `per_camera_mlp` follows the existing derivation: one MLP per camera under
    'proprio' (the constant flange->camera mount has to be learned), shared
    otherwise.
    """
    idx = [i for i, b in enumerate(blocks) if _modality(b) == modality]
    src = {resolved[i]["source"] for i in idx}
    assert len(src) == 1, src
    return len(idx), src.pop() == "proprio", [resolved[i]["has_pose"] for i in idx]


def effective_dataset_source(resolved, modality: str) -> str:
    """The pose_source to hand the single-modality dataset for this modality.

    Identical output in every case. The only rewrite is 'proprio' on a modality
    none of whose cameras has an arm (H1's 3D top, H2's 2D top): the dataset
    would emit zeros(9) either way, but `RealMultiCamUnfusedDataset` raises on
    that configuration because on its own it IS the degenerate run. The
    policy-wide guard above has already ruled that out.
    """
    rows = [r for r in resolved if r["modality"] == modality]
    src = rows[0]["source"]
    if src == "proprio" and not any(r["has_pose"] for r in rows):
        return "none"
    return src


def describe(blocks, resolved) -> str:
    lines = []
    for i, (b, r) in enumerate(zip(blocks, resolved)):
        mlp = "none (identity only)" if not r["has_pose"] else (
            "own" if r["source"] == "proprio" else "shared")
        lines.append(
            f"  block {i}: {b['key']:<18} cams={b['cameras']} {r['modality']} "
            f"source={r['source']:<9} pose={'yes' if r['has_pose'] else 'no':<3} "
            f"mlp={mlp}")
    return "\n".join(lines)
