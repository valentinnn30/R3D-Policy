#!/usr/bin/env python3
"""Rewrite an R3D zarr so every point cloud is stored ONE CHUNK PER FRAME.

`RealMultiCamDataset` opens its zarr with `create_from_path(mode="r")` -- off
disk, through the OS page cache, so concurrent training runs share one cached
copy instead of each holding a private ~12.6 GiB one. That only works if a
2-frame observation window decompresses ~2 frames. The converter used to write
100-frame chunks, which makes every sample decompress ~23 MB of clouds to use
~0.45 MB.

Only `data/point_cloud_*` is rechunked. state/action/target_ee/extrinsics are
tiny per frame, read as horizon-long windows, and stay at 100 frames/chunk.
Everything else -- meta arrays, every attribute -- is copied verbatim, and each
rechunked array is verified bit-identical to its source before exiting.

    micromamba run -n r3d python scripts/rechunk_zarr_per_frame.py \\
        --input  R3D/data/real_bimanual_mixed150.zarr \\
        --output R3D/data/real_bimanual_mixed150_perframe.zarr
"""
import argparse
import os
import sys

import numpy as np
import zarr

BLOCK = 1000  # frames per read/write block: ~230 MB of clouds, never a whole array


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    if os.path.exists(args.output):
        sys.exit(f"{args.output} already exists -- delete it first, refusing to overwrite.")

    src = zarr.open(args.input, mode="r")
    dst = zarr.group(args.output)
    dst.attrs.update(src.attrs.asdict())

    # meta is copied as raw store keys, byte for byte: zarr 2.x's group array
    # iterator asserts on any path starting with "meta", so it cannot be walked.
    zarr.copy_store(src.store, dst.store, source_path="meta", dest_path="meta")
    print("  meta: copied verbatim")

    for gname in ("data",):
        sg, dg = src[gname], dst.create_group(gname)
        dg.attrs.update(sg.attrs.asdict())
        for name in sorted(sg.array_keys()):
            arr = sg[name]
            per_frame = name.startswith("point_cloud")
            if not per_frame:
                zarr.copy(arr, dg, name=name)
                print(f"  {gname}/{name}: copied, chunks {arr.chunks}")
                continue
            out = dg.create_dataset(name, shape=arr.shape, dtype=arr.dtype,
                                    chunks=(1,) + arr.shape[1:],
                                    compressor=arr.compressor)
            out.attrs.update(arr.attrs.asdict())
            for lo in range(0, arr.shape[0], BLOCK):
                out[lo:lo + BLOCK] = arr[lo:lo + BLOCK]
            print(f"  {gname}/{name}: {arr.chunks} -> {out.chunks}")

    # Verify from a fresh open, so we compare what actually landed on disk.
    chk = zarr.open(args.output, mode="r")
    meta_keys = [k for k in src.store.keys() if k.startswith("meta/")]
    for k in meta_keys:
        if chk.store[k] != src.store[k]:
            sys.exit(f"MISMATCH: {k}")
    for gname in ("data",):
        if dict(chk[gname].attrs) != dict(src[gname].attrs):
            sys.exit(f"MISMATCH: {gname} attrs")
        for name in sorted(src[gname].array_keys()):
            arr = src[gname][name]
            got = chk[gname][name]
            if got.shape != arr.shape or got.dtype != arr.dtype:
                sys.exit(f"MISMATCH: {gname}/{name} shape/dtype")
            for lo in range(0, max(arr.shape[0], 1), BLOCK):
                if not np.array_equal(got[lo:lo + BLOCK], arr[lo:lo + BLOCK]):
                    sys.exit(f"MISMATCH: {gname}/{name} frames {lo}..{lo + BLOCK}")
    print(f"OK: {args.output} verified bit-identical to {args.input}")


if __name__ == "__main__":
    main()
