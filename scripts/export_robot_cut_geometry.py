#!/usr/bin/env python3
"""Export the robot-cut geometry read by `r3d.common.robot_cut.RobotCut`.

v2 (2026-10-05) contents:
  cap_*                 rule 1 (ARM): the URDF's panda_link<i>_sc capsules,
                        AS-IS (sphere-to-sphere axis + radius, through the fixed
                        *_sc_joint); padding is added at runtime
  wrist_<side>          rule 2 (MOUNT): wrist joint origin in hand_root
  forearm_axis_<side>   rule 2: forearm axis (x, y) in hand_root = centre of the
                        tower block (it runs along hand z)
  link<i>_*, tower_*    rule 3 (SHADOW): the TIGHT distance fields below

    micromamba activate r3d && unset PYTHONPATH
    cd ~/R3D-Policy
    python scripts/export_robot_cut_geometry.py

One distance field per part on a `--pitch` voxel grid in the part's frame,
extending `--pad` past the mesh: distance (m) from each voxel centre to the
solid, 0 inside.

  link0..link7  Franka collision meshes (viz/models/franka_panda), link frame.
                Shared by both arms.
  tower_<side>  every collision mesh of the orca `tower` link (the mounting
                block -- not palm, not fingers), in hand_root, from the xacro
                rendered for that chirality (scale -1 1 1 mirrors the left).

The solid is rasterised from a dense surface sample (spacing < pitch/2) and
then hole-filled, so the four tower meshes that are not watertight still give
a solid; cameras only ever see surfaces anyway.
"""
import argparse
import os
import re
import sys
import xml.etree.ElementTree as ET

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "R3D"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from r3d.common.real_preprocess import HAND_JOINT_ORDER, HandFK, _rpy_matrix  # noqa: E402
from export_orcahand_fk import (VIZ, _chain, _entry, _joint_origin, _render,  # noqa: E402
                                _tower_block_centre)

FRANKA_URDF = os.path.join(VIZ, "models/franka_panda/panda.urdf")


def _origin(el):
    T = np.eye(4)
    o = el.find("origin")
    if o is not None:
        T[:3, :3] = _rpy_matrix([float(v) for v in o.get("rpy", "0 0 0").split()])
        T[:3, 3] = [float(v) for v in o.get("xyz", "0 0 0").split()]
    return T


def _load(mesh_el, T):
    import trimesh
    tm = trimesh.load(re.sub(r"^package://viz/", VIZ + "/", mesh_el.get("filename")),
                      force="mesh")
    tm.vertices = tm.vertices * np.array([float(v) for v in mesh_el.get("scale", "1 1 1").split()])
    tm.apply_transform(T)
    return tm


def _inside(mesh, pts, chunk=20000):
    """Generalised winding number > 0.5 (watertight mesh). numpy only:
    trimesh's contains() needs rtree, which the r3d env does not have."""
    tri = mesh.triangles                                     # (F, 3, 3)
    out = np.zeros(len(pts), bool)
    for s in range(0, len(pts), chunk):
        p = pts[s:s + chunk]
        a, b, c = (tri[None, :, k, :] - p[:, None, :] for k in range(3))
        la, lb, lc = (np.linalg.norm(v, axis=-1) for v in (a, b, c))
        num = np.einsum("pfi,pfi->pf", a, np.cross(b, c))
        den = (la * lb * lc + np.einsum("pfi,pfi->pf", a, b) * lc
               + np.einsum("pfi,pfi->pf", b, c) * la + np.einsum("pfi,pfi->pf", c, a) * lb)
        w = np.arctan2(num, den).sum(axis=1) / (2 * np.pi)
        out[s:s + chunk] = np.abs(w) > 0.5
    return out


def distance_field(meshes, pitch, pad, seed=0):
    """(dist, origin) for the union of `meshes` (already in the part frame)."""
    import trimesh
    from scipy import ndimage
    lo = np.min([m.bounds[0] for m in meshes], axis=0) - pad
    hi = np.max([m.bounds[1] for m in meshes], axis=0) + pad
    shape = np.ceil((hi - lo) / pitch).astype(int) + 1
    occ = np.zeros(shape, bool)
    for k, m in enumerate(meshes):
        # enough samples for a spacing well under pitch/2
        n = int(np.ceil(m.area / (pitch / 3) ** 2))
        pts = np.concatenate([trimesh.sample.sample_surface(m, n, seed=seed + k)[0], m.vertices])
        ijk = np.rint((pts - lo) / pitch).astype(int)
        occ[ijk[:, 0], ijk[:, 1], ijk[:, 2]] = True
        if m.is_watertight:
            # exact interior; hole-filling a rasterised shell can leak
            # (it did on link1/link2)
            a = np.floor((m.bounds[0] - lo) / pitch).astype(int)
            b = np.ceil((m.bounds[1] - lo) / pitch).astype(int) + 1
            g = np.stack(np.meshgrid(*[np.arange(a[d], b[d]) for d in range(3)],
                                     indexing="ij"), -1).reshape(-1, 3)
            g = g[_inside(m, lo + g * pitch)]
            occ[g[:, 0], g[:, 1], g[:, 2]] = True
    occ = ndimage.binary_fill_holes(occ)
    dist = ndimage.distance_transform_edt(~occ) * pitch
    return dist.astype(np.float16), lo


def sc_capsules():
    """The URDF's self-collision capsules per link: (link, p0, p1, r)."""
    root = ET.parse(FRANKA_URDF).getroot()
    links = {l.get("name"): l for l in root.findall("link")}
    sc_joint = {j.find("child").get("link"): _origin(j) for j in root.findall("joint")
                if j.get("type") == "fixed"}
    out = []
    for i in range(8):
        S = sc_joint[f"panda_link{i}_sc"]
        sph = [c for c in links[f"panda_link{i}_sc"].findall("collision")
               if c.find("geometry")[0].tag == "sphere"]
        for a, b in zip(sph[0::2], sph[1::2]):
            ra, rb = (float(x.find("geometry")[0].get("radius")) for x in (a, b))
            if ra != rb:
                raise ValueError(f"link{i}: capsule end spheres differ")
            out.append((i, (S @ _origin(a))[:3, 3], (S @ _origin(b))[:3, 3], ra))
    for i, a, b, r in out:
        print(f"  sc capsule link{i}: len {np.linalg.norm(b - a) * 1000:4.0f} mm, r {r * 1000:3.0f} mm")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="R3D/r3d/config/preprocess/robot_cut_geometry.npz")
    ap.add_argument("--pitch", type=float, default=0.003, help="voxel size [m]")
    ap.add_argument("--pad", type=float, default=0.04,
                    help="field extent past the mesh [m]; the largest usable margin")
    args = ap.parse_args()
    out = {"pitch": np.float64(args.pitch), "pad": np.float64(args.pad)}
    caps = sc_capsules()
    out["cap_link"] = np.array([c[0] for c in caps], np.int32)
    out["cap_p0"] = np.array([c[1] for c in caps])
    out["cap_p1"] = np.array([c[2] for c in caps])
    out["cap_r"] = np.array([c[3] for c in caps])

    root = ET.parse(FRANKA_URDF).getroot()
    links = {l.get("name"): l for l in root.findall("link")}
    for i in range(8):
        coll = links[f"panda_link{i}"].find("collision")
        m = _load(coll.find("geometry/mesh"), _origin(coll))
        d, lo = distance_field([m], args.pitch, args.pad, seed=10 * i)
        out[f"link{i}_dist"], out[f"link{i}_origin"] = d, lo
        print(f"link{i}: grid {d.shape}, solid {int((d == 0).sum())} voxels, "
              f"mesh volume {m.volume * 1e6:.0f} cm3 vs voxel {(d == 0).sum() * args.pitch ** 3 * 1e6:.0f} cm3")

    for side in ("left", "right"):
        hroot = ET.fromstring(_render(side))
        tower = next(l for l in hroot.findall("link") if l.get("name") == "tower")
        T_tower = HandFK._eval_chain([_entry(x) for x in _chain(hroot, "tower")],
                                     np.zeros(len(HAND_JOINT_ORDER)))   # hand_root <- tower
        meshes = [_load(c.find("geometry/mesh"), T_tower @ _origin(c))
                  for c in tower.findall("collision")]
        d, lo = distance_field(meshes, args.pitch, args.pad, seed=100)
        out[f"wrist_{side}"] = _joint_origin(hroot, "wrist")
        out[f"forearm_axis_{side}"] = _tower_block_centre(hroot)
        print(f"  {side}: wrist joint {np.round(out[f'wrist_{side}'] * 1000, 1)} mm, "
              f"forearm axis (x, y) {np.round(out[f'forearm_axis_{side}'][:2] * 1000, 1)} mm "
              "(hand_root)")
        out[f"tower_{side}_dist"], out[f"tower_{side}_origin"] = d, lo
        print(f"tower_{side}: {len(meshes)} meshes, grid {d.shape}, solid "
              f"{(d == 0).sum() * args.pitch ** 3 * 1e6:.0f} cm3, extent "
              f"{np.round((np.max([m.bounds[1] for m in meshes], 0) - np.min([m.bounds[0] for m in meshes], 0)) * 1000).astype(int)} mm")

    out["source"] = np.array(f"{FRANKA_URDF} collision meshes + orcahand_v1b xacro tower")
    np.savez_compressed(args.out, **out)
    print(f"wrote {args.out} ({os.path.getsize(args.out) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
