#!/usr/bin/env python3
"""Undo the /paco clock skew baked into an existing zarr.

The recording machine and the Franka control PC ran unsynchronised clocks: every
/paco message carried a header stamp ~4.207 s in the FUTURE. A stamp that is
later than its own arrival can only mean a fast clock, never late data -- local
camera topics show -2 to -12 ms, which is what real transport latency looks
like. `syncronize_data.py` aligns streams by header stamp, so the /paco-sourced
fields were paired with point clouds from ~4.2 s away.

Only the /paco-sourced fields are wrong:

  state[:, 0:14]      arm EE poses     <- /paco/../end_effector_pose   BROKEN
  extrinsics_cam1/2   wrist cam poses  <- /paco/joint_states via FK    BROKEN
  action[:, 0:14]     arm commands     <- LOCAL teleop node            fine
  state/action[14:]   hands            <- LOCAL, headerless            fine
  point_cloud_cam*    cameras          <- LOCAL                        fine

This takes those two from `shift` frames later, so grid index i gets the value
actually measured at that instant, and drops each episode's tail.

KNOWN RESIDUAL, not fixed here: the per-camera crop and FPS at conversion time
also used the wrong extrinsics, so each wrist camera's stored 5120 points are
the subset visible from a pose displaced by a median ~10-13 cm (p99 ~30 cm).
`fuse_cameras` re-crops with the corrected extrinsics, so wrongly-INCLUDED
points are removed at training time; the irreversible part is legitimate points
dropped near one workspace boundary. Removing that needs a full re-conversion
from the h5, reading joint_states and qpos_arm at the same offset.

`shift` is the CLOCK OFFSET only, measured per file from the raw h5. Sweeping it
shows the command-vs-state gap bottoming out ~4 frames further on; that extra
~0.19 s is genuine servo lag and is deliberately KEPT.

    python scripts/fix_zarr_clock_offset.py --dry-run
    python scripts/fix_zarr_clock_offset.py
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys

import h5py
import numpy as np
import zarr

SRC = os.path.expanduser("~/R3D-Policy/R3D/data/real_bimanual.zarr")
DST = os.path.expanduser("~/R3D-Policy/R3D/data/real_bimanual_clockfixed.zarr")
H5DIR = os.path.expanduser("~/ros2_ws/tmp/50traj_v1")
ARM = slice(0, 14)
G, R, Y, N = "\033[32m", "\033[31m", "\033[33m", "\033[0m"


def measure_offset(h5_path):
    """median(header stamp - arrival) on a /paco topic, seconds."""
    with h5py.File(h5_path, "r") as f:
        g = f["left/arm/end_effector_pose"]
        ks = sorted(g.keys(), key=int)
        d = [(float(g[k].attrs["stamp"]) - float(int(k))) / 1e9 for k in ks]
    return float(np.median(d))


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", default=SRC)
    p.add_argument("--output", default=DST)
    p.add_argument("--h5-dir", default=H5DIR)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--chunk", type=int, default=200)
    args = p.parse_args()

    z = zarr.open(args.input, "r")
    ends = z["meta/episode_ends"][:]
    starts = np.concatenate([[0], ends[:-1]])
    hz = float(z["meta"].attrs["control_hz"])
    h5s = sorted(os.path.join(args.h5_dir, x)
                 for x in os.listdir(args.h5_dir) if x.endswith(".h5"))
    if len(h5s) != len(ends):
        print(f"{R}FAIL{N} {len(h5s)} h5 files vs {len(ends)} episodes")
        return 1

    print(f"{'ep':>3} {'frames':>7} {'offset_s':>9} {'shift':>6} {'kept':>7}")
    shifts, kept = [], []
    for i, (s, e, h5) in enumerate(zip(starts, ends, h5s)):
        off = measure_offset(h5)
        k = int(round(off * hz))
        shifts.append(k); kept.append((e - s) - k)
        print(f"{i:3d} {e-s:7d} {off:9.4f} {k:6d} {kept[-1]:7d}")

    tin, tout = int(ends[-1]), int(sum(kept))
    print(f"\n{tin} -> {tout} frames ({tin-tout} dropped, "
          f"{100*(tin-tout)/tin:.1f}%)")
    if args.dry_run:
        print(f"{G}dry run: nothing written{N}")
        return 0

    if os.path.exists(args.output):
        shutil.rmtree(args.output)
    out = zarr.open(args.output, "w")
    gd, gm = out.create_group("data"), out.create_group("meta")
    src = z["data"]
    new_ends = np.cumsum(kept).astype(np.int64)
    new_starts = np.concatenate([[0], new_ends[:-1]])

    for name in list(src.array_keys()):
        a = src[name]
        # preserve the source's codec/filters -- letting zarr fall back to its
        # default (lz4) made the output LARGER than the input despite having
        # fewer frames, because the source uses zstd
        d = gd.zeros(name, shape=(tout,) + a.shape[1:], chunks=a.chunks,
                     dtype=a.dtype, compressor=a.compressor,
                     filters=a.filters, order=a.order)
        shifted = name in ("extrinsics_cam1", "extrinsics_cam2")
        for s, e, k, ns, ne in zip(starts, ends, shifts, new_starts, new_ends):
            for b in range(0, ne - ns, args.chunk):
                n = min(args.chunk, (ne - ns) - b)
                if name == "state":
                    blk = np.asarray(a[s+b: s+b+n])                  # hands as-is
                    blk[:, ARM] = np.asarray(a[s+b+k: s+b+k+n, ARM])
                elif shifted:
                    blk = np.asarray(a[s+b+k: s+b+k+n])
                else:
                    blk = np.asarray(a[s+b: s+b+n])
                d[ns+b: ns+b+n] = blk
        print(f"  wrote {name:20s} {d.shape}")

    gm.array("episode_ends", new_ends)
    gm.array("extrinsics_cam0", z["meta/extrinsics_cam0"][:])
    for k, v in z["meta"].attrs.items():
        gm.attrs[k] = v
    gm.attrs["clock_fix_shift_frames"] = [int(x) for x in shifts]
    gm.attrs["clock_fix_note"] = (
        "state[:,0:14] and extrinsics_cam1/2 shifted forward to undo a /paco "
        "header-stamp clock offset. Servo lag retained. Per-camera crop/FPS "
        "still reflect the uncorrected extrinsics -- see the module docstring.")

    zz = zarr.open(args.output, "r")
    s2, a2 = zz["data/state"][:], zz["data/action"][:]
    gap = np.median(np.linalg.norm(a2[:, 0:3] - s2[:, 0:3], axis=1)) * 1000
    s0, a0 = z["data/state"][:], z["data/action"][:]
    gap0 = np.median(np.linalg.norm(a0[:, 0:3] - s0[:, 0:3], axis=1)) * 1000
    good = gap < 20
    print(f"\n{G if good else R}{'PASS' if good else 'FAIL'}{N}  "
          f"left |action-state| median {gap0:.1f} mm -> {gap:.1f} mm "
          f"(residual = real servo lag)")
    same = np.array_equal(zz["data/action"][:1000, 14:], z["data/action"][:1000, 14:])
    print(f"{G if same else R}{'PASS' if same else 'FAIL'}{N}  hand actions untouched")
    print(f"\nwrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
