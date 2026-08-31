#!/usr/bin/env python3
"""Concatenate two or more R3D zarrs into one training set.

Episodes are stored flat with cumulative `meta/episode_ends`, so merging is a
concatenation of every `data/` array plus an offset on the boundaries. The only
real work is refusing to merge datasets that are not interchangeable.

REFUSES unless the inputs agree on: the set of data arrays, every array's
per-frame shape and dtype, `meta/extrinsics_cam0`, and every SEMANTIC `meta`
attribute (camera_names, control_hz, quaternion_order/slices, state/action
layout). Those encode what an observation MEANS -- two zarrs built with
different workspaces, floor cuts or point budgets would concatenate happily and
train into nonsense, since the normalizer is derived from the config, not from
the data.

`patch_*` and `note_*` attributes are PROVENANCE, not semantics: they are not
required to match, and every input's are carried into the output tagged by
source, so a repair applied to one dataset stays discoverable after merging.

Copies frame-by-frame in blocks, never loading a whole array: a merged set here
is several GB.

    micromamba run -n r3d python scripts/merge_zarr.py \\
        --inputs R3D/data/real_bimanual_floorcut80.zarr \\
                 R3D/data/real_bimanual_recovery25.zarr \\
        --output R3D/data/real_bimanual_floorcut105.zarr
"""
import argparse
import numpy as np
import zarr
from termcolor import cprint

BLOCK = 200          # frames copied per read/write


def check(zs, paths):
    ref, rp = zs[0], paths[0]
    keys = set(ref["data"].keys())
    for z, p in zip(zs[1:], paths[1:]):
        k = set(z["data"].keys())
        if k != keys:
            raise SystemExit(f"data arrays differ:\n  {rp}: {sorted(keys)}\n  {p}: {sorted(k)}")
        for name in sorted(keys):
            a, b = ref["data"][name], z["data"][name]
            if a.shape[1:] != b.shape[1:] or a.dtype != b.dtype:
                raise SystemExit(
                    f"'{name}' is not interchangeable:\n"
                    f"  {rp}: {a.shape[1:]} {a.dtype}\n  {p}: {b.shape[1:]} {b.dtype}\n"
                    "Different point budgets or workspace -> different observation.")
        if "extrinsics_cam0" in ref["meta"]:
            if not np.allclose(np.array(ref["meta"]["extrinsics_cam0"]),
                               np.array(z["meta"]["extrinsics_cam0"])):
                raise SystemExit("meta/extrinsics_cam0 differs -- the static camera "
                                 "moved, or one set was built with other calibration.")
        # Only attributes that define what an OBSERVATION MEANS have to match.
        # `patch_*` / `note_*` are provenance -- e.g. a zarr whose wrist column
        # was repaired carries a patch note the others do not, and refusing to
        # merge over that would be wrong: it describes history, not semantics.
        for a in sorted(set(ref["meta"].attrs) | set(z["meta"].attrs)):
            if a.startswith(("patch_", "note_")):
                continue
            if ref["meta"].attrs.get(a) != z["meta"].attrs.get(a):
                raise SystemExit(f"meta attribute '{a}' differs:\n"
                                 f"  {rp}: {ref['meta'].attrs.get(a)}\n"
                                 f"  {p}: {z['meta'].attrs.get(a)}")
    return sorted(keys)


def provenance(zs, paths):
    """Collect every patch/note attribute across the inputs, tagged by source."""
    out = {}
    for z, p in zip(zs, paths):
        for k, v in z["meta"].attrs.items():
            if k.startswith(("patch_", "note_")):
                out[f"{k} [{p.split('/')[-1]}]"] = v
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--inputs", nargs="+", required=True)
    p.add_argument("--output", required=True)
    a = p.parse_args()

    zs = [zarr.open(x, "r") for x in a.inputs]
    keys = check(zs, a.inputs)

    ends = [np.array(z["meta"]["episode_ends"]) for z in zs]
    frames = [int(e[-1]) for e in ends]
    total = sum(frames)
    for x, e, n in zip(a.inputs, ends, frames):
        cprint(f"  {x}: {len(e)} episodes, {n} frames", "cyan")
    cprint(f"-> {sum(len(e) for e in ends)} episodes, {total} frames", "green")

    out = zarr.open(a.output, "w")
    g = out.create_group("data")
    for name in keys:
        src = zs[0]["data"][name]
        dst = g.create_dataset(name, shape=(total,) + src.shape[1:],
                               chunks=src.chunks, dtype=src.dtype,
                               compressor=src.compressor)
        off = 0
        for z in zs:
            s = z["data"][name]
            for i in range(0, s.shape[0], BLOCK):
                j = min(i + BLOCK, s.shape[0])
                dst[off + i: off + j] = s[i:j]
            off += s.shape[0]
        cprint(f"  {name:22s} {dst.shape}", "green")

    m = out.create_group("meta")
    merged, off = [], 0
    for e in ends:
        merged.append(e + off)
        off += e[-1]
    m.create_dataset("episode_ends", data=np.concatenate(merged).astype(np.int64))
    for k in zs[0]["meta"]:
        if k != "episode_ends":
            m.create_dataset(k, data=np.array(zs[0]["meta"][k]))
    for k, v in zs[0]["meta"].attrs.items():
        if not k.startswith(("patch_", "note_")):
            m.attrs[k] = v
    # provenance from EVERY input, tagged by source, so a repair applied to one
    # dataset stays discoverable after it is merged into a larger one
    for k, v in provenance(zs, a.inputs).items():
        m.attrs[k] = v

    e = np.array(m["episode_ends"])
    assert e[-1] == total and np.all(np.diff(e) > 0), "episode_ends not monotonic"
    cprint(f"\nwrote {a.output}", "green")
    cprint(f"  episodes {len(e)}  frames {total}  boundaries monotonic, last == total", "green")


if __name__ == "__main__":
    main()
