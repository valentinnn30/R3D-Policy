#!/usr/bin/env python
"""Step 0.2 of the learned link remover (docs/learned_remover_v0.md): remover unit test on
real zarr frames.  Run in r3d_sim from the repo root, after add_link_labels_mimicgen.py.

    python scripts/sim/check_link_remover.py

Checks (`r3d/model/vision/link_remover.py`), at N = 1024 and 4096:
  1 a fixed rho table keeps exactly round(rho_l * n_l) points of every region (the kept
    points = FPS within the region), every non-robot point is kept, order is preserved
  2 one draw per window: both observation frames are thinned with the same rho
  3 rho = 1 everywhere returns the input unchanged, length N
  4 rho = 0 everywhere == label_cut.cut_points_by_label(arm + gripper), bit for bit
  5 already-padded input (lengths < N): padding never enters a region or the output
  6 the encoder (DP3Encoder, deterministic FPS) on padded cloud + point_cloud_len equals
    the encoder on the trimmed cloud, bit for bit -- the R1 padding contract
  7 fill-up: a draw keeping < M points gets removed region points back (FPS) to exactly M;
    clouds already >= M are untouched
  8 contract violations raise: N_kept < M (without fill), unknown cell id, rho outside
    [0, 1], shape; mu bounds; inference rho = sigmoid(mu), snapped below snap_rho_below
"""

import os
import pathlib
import sys

import numpy as np
import torch
import zarr

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "R3D"))
from r3d.common import robosuite_pcd as RP  # noqa: E402
from r3d.common.label_cut import cut_points_by_label  # noqa: E402
from r3d.common.token_allocation import box_normalize  # noqa: E402
from r3d.model.vision.link_remover import FPS_MIN_WIDTH, LinkRemover, thin_regions, thin_window  # noqa: E402
from pytorch3d.ops import sample_farthest_points  # noqa: E402

ZARR = ROOT / "R3D" / "data" / "square_d2-agentview_p4096-200.zarr"
DEV = "cuda"
FAILS = []


def check(name, ok, detail=""):
    print(f"  [{'ok' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))
    if not ok:
        FAILS.append(name)


def load_windows(z, N, n_win=12, seed=0):
    """(B, 2, N, 6) clouds, (B, 2, N) cells, (B, 2, N) labels: frames t, t+1 of random windows."""
    rng = np.random.default_rng(seed)
    ends = z["meta/episode_ends"][:]
    starts = np.concatenate([[0], ends[:-1]])
    eps = rng.choice(len(ends), n_win, replace=False)
    ts = [int(rng.integers(starts[e], ends[e] - 1)) for e in eps]
    idx = sorted(set(ts) | {t + 1 for t in ts})
    pos = {t: i for i, t in enumerate(idx)}
    pc = z["data/point_cloud"].get_orthogonal_selection((idx, slice(0, N)))
    cell = z["data/point_link"].get_orthogonal_selection((idx, slice(0, N)))
    lab = z["data/point_label"].get_orthogonal_selection((idx, slice(0, N)))
    sel = [[pos[t], pos[t + 1]] for t in ts]
    return pc[sel], cell[sel].astype(np.int64), lab[sel]


def main():
    z = zarr.open(str(ZARR), "r")
    if "point_link" not in z["data"]:
        raise KeyError(f"{ZARR} has no data/point_link -- run scripts/sim/add_link_labels_mimicgen.py")
    rem = LinkRemover(z["meta"].attrs["point_link_names"], RP.default_region_groups()).to(DEV)
    c2r = rem.cell_to_region
    L = rem.n_regions
    torch.manual_seed(0)

    for N in (1024, 4096):
        print(f"\n=== N = {N} ===")
        pc, cell, lab = load_windows(z, N)
        B = len(pc)
        P = torch.as_tensor(pc, device=DEV)
        Cc = torch.as_tensor(cell, device=DEV)
        # tag every point with its cell and original index, so the output can be audited
        idx_ch = torch.arange(N, device=DEV, dtype=torch.float32)[None, None, :, None].expand(B, 2, N, 1)
        tagged = torch.cat([P, Cc[..., None].float(), idx_ch], -1)

        # 1 + 2: fixed rho table (includes exact 0 and 1), one draw per window
        rho = torch.rand(B, L, device=DEV)
        rho[:, 0], rho[:, -1] = 0.0, 1.0
        out, ln, n, k, _ = thin_window(tagged, Cc, rho, c2r)
        ok_count = ok_nonrobot = ok_order = ok_fps = ok_share = True
        reg_np = c2r.cpu().numpy()
        for b in range(B):
            for f in range(2):
                o = out[b, f, :ln[b, f]].cpu().numpy()
                src_idx = o[:, 7].astype(np.int64)
                ok_order &= bool(np.all(np.diff(src_idx) > 0))
                ok_nonrobot &= bool(np.array_equal(src_idx[cell[b, f][src_idx] < 0], np.flatnonzero(cell[b, f] < 0)))
                region = np.where(cell[b, f] >= 0, reg_np[np.clip(cell[b, f], 0, None)], -1)
                for l in range(L):
                    nl = int((region == l).sum())
                    want = int(np.floor(rho[b, l].item() * nl + 0.5))   # same float32 rho, rounded half up
                    got = int((region[src_idx] == l).sum())
                    ok_count &= got == want == int(k[b, f, l]) and nl == int(n[b, f, l])
                    if 0 < want and l in (5, 8):          # FPS within the region (deterministic start)
                        members = np.flatnonzero(region == l)
                        x = P[b, f, members][None, :, :3]
                        if x.shape[1] < FPS_MIN_WIDTH:     # pytorch3d CUDA FPS bug below 8 points
                            x = torch.cat([x, x[:, :1].expand(1, FPS_MIN_WIDTH - x.shape[1], 3)], 1)
                        _, sel = sample_farthest_points(x.contiguous(), lengths=torch.tensor([len(members)], device=DEV),
                                                        K=want, random_start_point=False)
                        ok_fps &= set(members[sel[0].cpu().numpy()]) == set(src_idx[region[src_idx] == l])
        check("kept per region == round(rho * n), counts returned", ok_count)
        check("every non-robot point kept, original order preserved", ok_nonrobot and ok_order)
        check("kept region points == FPS within the region (deterministic start)", ok_fps)
        # same rho for both frames: k / n follow rho[b] in frame 0 AND frame 1 (checked above per frame
        # with rho[b]); also check thin_window did not draw anything per frame
        out2, ln2, _, _, _ = thin_window(tagged, Cc, rho, c2r)
        check("one draw per window: both frames use rho[b]; thinning is deterministic",
              ok_count and torch.equal(out, out2) and torch.equal(ln, ln2))

        # 3: rho = 1
        o1, l1, _, _, _ = thin_window(P, Cc, torch.ones(B, L, device=DEV), c2r)
        check("rho = 1 -> input unchanged, length N", torch.equal(o1, P) and bool((l1 == N).all()))

        # 4: rho = 0 == oracle label cut of arm + gripper
        o0, l0, _, _, _ = thin_window(P, Cc, torch.zeros(B, L, device=DEV), c2r)
        same = True
        for b in range(B):
            for f in range(2):
                ref, nref = cut_points_by_label(pc[b, f], lab[b, f], [RP.RIGHT_ARM, RP.RIGHT_GRIPPER])
                same &= nref == int(l0[b, f]) and np.array_equal(ref, o0[b, f].cpu().numpy())
        check("rho = 0 == cut_points_by_label([5, 7]), padded arrays bit-identical", same,
              f"kept {int(l0.min())}..{int(l0.max())} of {N}")

        # 5: padded input
        flatP, flatC = P.reshape(B * 2, N, 6), Cc.reshape(B * 2, N)
        lens = torch.randint(N // 2, N, (B * 2,), device=DEV)
        o5, l5, n5, k5, _ = thin_regions(flatP, flatC, rho.repeat_interleave(2, 0), c2r, lengths=lens)
        o5t, l5t, n5t, k5t = [], [], [], []
        okp = True
        for c in range(B * 2):
            m = int(lens[c])
            a, la, na, ka, _ = thin_regions(flatP[c:c + 1, :m], flatC[c:c + 1, :m], rho.repeat_interleave(2, 0)[c:c + 1], c2r)
            okp &= int(la) == int(l5[c]) and torch.equal(a[0, :int(la)], o5[c, :int(la)]) and torch.equal(na[0], n5[c])
        check("already-padded input: padding never thinned or kept", okp)

        # 7: fill-up to M -- rho = 0 everywhere, M above the non-robot count of some clouds
        M = int(l0.float().median())
        of, lf, _, kf, filled = thin_window(tagged, Cc, torch.zeros(B, L, device=DEV), c2r,
                                            min_keep=M, fill_to_min_keep=True)
        okf = bool((lf == torch.clamp(l0, min=M)).all()) and bool((filled == (M - l0).clamp(min=0)).all())
        for b in range(B):
            for f in range(2):
                o = of[b, f, :lf[b, f]].cpu().numpy()
                src = o[:, 7].astype(np.int64)
                okf &= bool(np.all(np.diff(src) > 0))
                okf &= bool(np.isin(np.flatnonzero(cell[b, f] < 0), src).all())          # non-robot all kept
                okf &= int((cell[b, f][src] >= 0).sum()) == int(filled[b, f])          # added = robot pts
                if int(filled[b, f]) == 0:
                    okf &= torch.equal(of[b, f, :, :6], o0[b, f])
        check("fill-up: clouds below M refilled with removed robot points to exactly M, others untouched", okf,
              f"M = {M}, {int((filled > 0).sum())}/{2 * B} clouds filled")

        # 6: encoder padded + lengths == trimmed (the R1 contract), deterministic FPS
        enc_out_ok, detail = encoder_check(o0.reshape(B * 2, N, 6), l0.reshape(-1), z["meta"].attrs["pcd_crop_bbox"], N)
        check("encoder: padded + point_cloud_len == trimmed cloud, bit for bit", enc_out_ok, detail)

    # 7: contract violations
    print("\n=== contract violations ===")
    pc, cell, lab = load_windows(z, 1024, n_win=2)
    P, Cc = torch.as_tensor(pc, device=DEV), torch.as_tensor(cell, device=DEV)
    L = rem.n_regions
    cases = {
        "N_kept < M": lambda: thin_window(P, Cc, torch.zeros(2, L, device=DEV), c2r, min_keep=1024),
        "unknown cell id": lambda: thin_window(P, torch.full_like(Cc, RP.N_CELLS), torch.ones(2, L, device=DEV), c2r),
        "rho > 1": lambda: thin_window(P, Cc, torch.full((2, L), 1.5, device=DEV), c2r),
        "rho NaN": lambda: thin_window(P, Cc, torch.full((2, L), float("nan"), device=DEV), c2r),
        "cells shape": lambda: thin_regions(P[:, 0], Cc[:, 0, :10], torch.ones(2, L, device=DEV), c2r),
        "unknown cell name in region_groups": lambda: LinkRemover(RP.CELL_NAMES, [{"name": "x", "cells": ["link9"]}]),
        "cell in two regions": lambda: LinkRemover(RP.CELL_NAMES, [{"name": "a", "cells": ["link3"]},
                                                                   {"name": "b", "cells": ["link3"]}]),
    }
    for nm, fn in cases.items():
        try:
            fn()
            check(f"raises on {nm}", False)
        except (ValueError, RuntimeError, KeyError) as e:
            check(f"raises on {nm}", True, type(e).__name__)

    # mu bounds + deterministic inference
    r2 = LinkRemover(RP.CELL_NAMES, RP.default_region_groups())
    with torch.no_grad():
        r2.mu.copy_(torch.linspace(-20, 20, r2.n_regions))
    r2.clamp_()
    check("mu clamped to [-8, 6]", float(r2.mu.min()) == -8.0 and float(r2.mu.max()) == 6.0)
    check("inference rho = sigmoid(mu), no sampling", torch.equal(r2.rho(), torch.sigmoid(r2.mu.detach())))

    r3 = LinkRemover(RP.CELL_NAMES, RP.default_region_groups(), snap_rho_below=0.05)
    with torch.no_grad():
        r3.mu.copy_(torch.linspace(-6, 4, r3.n_regions))
    det = torch.sigmoid(r3.mu.detach())
    snapped = r3.rho()
    zs = r3.sample(4096)
    check("snap: deterministic rho < 0.05 -> 0, the rest unchanged; training draws never snapped",
          torch.equal(snapped, torch.where(det < 0.05, torch.zeros_like(det), det)) and bool((snapped == 0).any())
          and torch.equal(r3.rho(zs), torch.sigmoid(zs)) and bool((r3.rho(zs) < 0.05).any()))

    print(f"\n{'ALL CHECKS PASSED' if not FAILS else 'FAILED: ' + ', '.join(FAILS)}")
    sys.exit(1 if FAILS else 0)


def encoder_check(clouds, lengths, bbox, N):
    """DP3Encoder from the real policy config, FPS made deterministic; each cloud alone."""
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf
    import hydra
    from r3d.model.vision.pointnet_extractor import KNNGrouper
    if not OmegaConf.has_resolver("eval"):
        OmegaConf.register_new_resolver("eval", eval, replace=True)
    with initialize_config_dir(config_dir=str(ROOT / "R3D" / "r3d" / "config"), version_base=None):
        cfg = compose(config_name="r3d_robotwin2", overrides=[
            "task=mimicgen_sim", "task_name=square_d2", "setting=agentview_p4096-200",
            f"task.num_points={N}", "task.num_group=64"])
    cwd = os.getcwd()
    os.chdir(ROOT)            # pretrained weight paths are repo-relative, as train.py chdirs
    try:
        policy = hydra.utils.instantiate(cfg.policy)
    finally:
        os.chdir(cwd)
    enc = policy.obs_encoder.to(DEV).eval()
    for m in enc.modules():
        if isinstance(m, KNNGrouper):
            m.fps_random_config = {"use_random": False}
    xyz = torch.as_tensor(box_normalize(clouds[..., :3].cpu().numpy(), bbox), device=DEV)
    x = torch.cat([xyz, clouds[..., 3:6] * 2 - 1], -1)
    agent = torch.zeros(len(x), 11, device=DEV)
    worst = 0.0
    with torch.no_grad():
        for c in range(len(x)):
            m = int(lengths[c])
            fa, pa = enc({"point_cloud": x[c:c + 1], "agent_pos": agent[c:c + 1],
                          "point_cloud_len": lengths[c:c + 1]}, eval=True)
            fb, pb = enc({"point_cloud": x[c:c + 1, :m], "agent_pos": agent[c:c + 1]}, eval=True)
            if not (torch.equal(fa, fb) and torch.equal(pa, pb)):
                worst = max(worst, float((fa - fb).abs().max()), float((pa - pb).abs().max()))
    return worst == 0.0, f"{len(x)} clouds" + ("" if worst == 0.0 else f", max |diff| {worst:.3g}")


if __name__ == "__main__":
    main()
