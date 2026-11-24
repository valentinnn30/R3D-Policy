#!/usr/bin/env python
"""Step 0.2 of the learned link remover (docs/learned_remover_v0.md): remover unit test on
real zarr frames.  Run in r3d_sim from the repo root, after add_link_labels_mimicgen.py.

    python scripts/sim/check_link_remover.py

Checks (`r3d/model/vision/link_remover.py`), at N = 1024 and 4096:
  1 a fixed rho table keeps exactly round(rho_l * n_l) points of every region (the kept
    points = FPS within the region), every non-robot point is kept, order is preserved
  2 one draw per window: both observation frames are thinned with the same rho
  3 rho = 1 everywhere returns the input unchanged, length N
  4 rho = 0 everywhere == label_cut.cut_points_by_label of the region points, bit for bit
  5 already-padded input (lengths < N): padding never enters a region or the output
  6 the encoder (DP3Encoder, deterministic FPS) on padded cloud + point_cloud_len equals
    the encoder on the trimmed cloud, bit for bit -- the R1 padding contract
  7 fill-up: a draw keeping < M points gets removed region points back (FPS) to exactly M;
    clouds already >= M are untouched
  8 contract violations raise: N_kept < M (without fill), unknown cell id, rho outside
    [0, 1], shape; mu bounds; inference rho = sigmoid(mu), snapped below snap_rho_below
  9 shared within-region FPS starts (CRN): masks are nested -- with rho1 <= rho2 the k1
    kept points of a region are exactly the first k1 of the k2 selection

Policy checks (DP3LinkRemover, N = 1024, 64 tokens; skip with --no-policy):
 10 remover off: compute_loss and predict_action bit-identical to baseline DP3
 11 common-random-numbers reward: the same mask twice -> reward difference exactly 0;
    the same (masks, seeds) twice -> identical rewards; other seeds -> other rewards
 12 the theta pass keeps independent random starts: the encoder's token FPS draws its own
    start / shuffle and the within-region FPS its own starts (nothing shared is passed)
 13 fixed-mask specialist (fixed_removed: [link5, link6], round 3): the training and the eval
    clouds have exactly link5 + link6 removed and every other point kept, in order; no phi
    parameter changes over a controller step; the controller has no phi optimizer, phase
    "fixed", and its diagnostics are the mechanism only
 14 density specialist (fixed_rho, default+scene): rho / draws == the table exactly, nothing
    learnable, contract violations raise; the training and eval clouds keep exactly
    round(rho_l * n_l) points of every region (snapped regions 0), the controller is "fixed"
"""

import math
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
from r3d.model.vision.link_remover import FPS_MIN_WIDTH, NON_ROBOT, LinkRemover, thin_regions, thin_window  # noqa: E402
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
                in_reg = (cell[b, f] >= 0) & (reg_np[np.clip(cell[b, f], 0, None)] >= 0)
                ref, nref = cut_points_by_label(pc[b, f], in_reg.astype(np.int64), [1])
                same &= nref == int(l0[b, f]) and np.array_equal(ref, o0[b, f].cpu().numpy())
        check("rho = 0 == label cut of every region point (link0-4 stay), padded arrays bit-identical", same,
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
                in_reg = (cell[b, f] >= 0) & (reg_np[np.clip(cell[b, f], 0, None)] >= 0)
                okf &= bool(np.isin(np.flatnonzero(~in_reg), src).all())                # non-region all kept
                okf &= int(in_reg[src].sum()) == int(filled[b, f])                     # added = region pts
                if int(filled[b, f]) == 0:
                    okf &= torch.equal(of[b, f, :, :6], o0[b, f])
        check("fill-up: clouds below M refilled with removed robot points to exactly M, others untouched", okf,
              f"M = {M}, {int((filled > 0).sum())}/{2 * B} clouds filled")

        # 6: encoder padded + lengths == trimmed (the R1 contract), deterministic FPS
        enc_out_ok, detail = encoder_check(o0.reshape(B * 2, N, 6), l0.reshape(-1), z["meta"].attrs["pcd_crop_bbox"], N)
        check("encoder: padded + point_cloud_len == trimmed cloud, bit for bit", enc_out_ok, detail)

    # 9: nested masks under shared within-region starts
    print("\n=== shared within-region FPS starts (CRN) ===")
    pc, cell, lab = load_windows(z, 4096, n_win=8, seed=3)
    P, Cc = torch.as_tensor(pc, device=DEV), torch.as_tensor(cell, device=DEV)
    Bn, L = len(P), rem.n_regions
    u_reg = torch.rand(Bn, 2, L, device=DEV)
    rho2 = torch.rand(Bn, L, device=DEV)
    rho1 = rho2 * torch.rand(Bn, L, device=DEV)
    reg = c2r.to(DEV)[Cc.clamp(min=0)].masked_fill(Cc < 0, -1)
    nested, strict = True, 0
    keeps = []
    for r_ in (rho1, rho2):
        *_, kp = thin_window(P, Cc, r_, c2r, u_region=u_reg, return_keep=True)
        keeps.append(kp)
    in_reg = reg >= 0
    nested = bool(((keeps[0] & in_reg) <= (keeps[1] & in_reg)).all())
    strict = int(((keeps[1] & ~keeps[0]) & in_reg).sum())
    # and the k1 points ARE the first k1 of the FPS-k2 order (prefix, not just a subset)
    from r3d.model.vision.link_remover import _fps_from
    prefix_ok = True
    for b in range(Bn):
        for l in (0, 1, 3):
            members = torch.nonzero(reg[b, 0] == l).squeeze(1)
            nl = len(members)
            k1, k2 = int(np.floor(rho1[b, l].item() * nl + 0.5)), int(np.floor(rho2[b, l].item() * nl + 0.5))
            if nl == 0 or k2 == 0:
                continue
            x = P[b, 0, members, :3][None]
            if x.shape[1] < FPS_MIN_WIDTH:
                x = torch.cat([x, x[:, :1].expand(1, FPS_MIN_WIDTH - x.shape[1], 3)], 1)
            seq = _fps_from(x, torch.tensor([nl], device=DEV), torch.tensor([k2], device=DEV), u_reg[b, 0, l:l + 1])[0]
            first = set(members[seq[:k1]].tolist())
            prefix_ok &= first == set(torch.nonzero(keeps[0][b, 0] & (reg[b, 0] == l)).squeeze(1).tolist())
    check("nested masks: rho1 <= rho2 with shared starts -> kept1 subset of kept2", nested,
          f"{strict} extra points in the larger masks")
    check("nested masks: kept1 == first k1 points of the shared-start FPS-k2 order", prefix_ok)
    a1 = thin_window(P, Cc, rho2, c2r, u_region=u_reg, return_keep=True)[5]
    a2 = thin_window(P, Cc, rho2, c2r, u_region=torch.rand_like(u_reg), return_keep=True)[5]
    check("different shared starts -> different selections (the start is really used)",
          bool((a1 != a2).any()))

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
    with torch.no_grad():
        r2.gate.copy_(torch.linspace(-20, 20, r2.n_regions))
    r2.clamp_()
    check("mu clamped to [-9, 6], gate logits to [-5, 5]", float(r2.mu.min()) == -9.0 and float(r2.mu.max()) == 6.0
          and float(r2.gate.min()) == -5.0 and float(r2.gate.max()) == 5.0)
    det = torch.where(torch.sigmoid(r2.gate.detach()) >= 0.5, torch.sigmoid(r2.mu.detach()), torch.zeros(r2.n_regions))
    check("inference rho = 1[keep prob >= 0.5] * sigmoid(mu), no sampling", torch.equal(r2.rho(), det)
          and bool((r2.rho() == 0).any()))

    # removal gate: sampling, log P, closed gate -> region gone and its density not in log P
    print("\n=== removal gate ===")
    r4 = LinkRemover(RP.CELL_NAMES, RP.default_region_groups(), init_keep=0.8, sigma=0.7)
    torch.manual_seed(0)
    dr = r4.sample(20000)
    zz, gg = r4.split(dr)
    check("gates open with P(keep) = init_keep", abs(gg.mean().item() - 0.8) < 0.01, f"{gg.mean().item():.3f}")
    check("draw rho = g * sigmoid(z): a closed gate removes the whole region",
          torch.equal(r4.rho(dr), gg * torch.sigmoid(zz)) and bool((r4.rho(dr)[gg == 0] == 0).all()))
    lp = r4.log_prob(dr[:5])
    a_, m_, s_ = r4.gate.detach(), r4.mu.detach(), r4.sigma
    ref = (gg[:5] * torch.log(torch.sigmoid(a_)) + (1 - gg[:5]) * torch.log(1 - torch.sigmoid(a_))
           + gg[:5] * torch.distributions.Normal(m_, s_).log_prob(zz[:5])).sum(-1)
    check("log P = sum[log Bern(g) + g log N(z)]", torch.allclose(lp, ref, atol=1e-5))
    closed = dr[:1].clone()
    closed[0, r4.n_regions:] = 0.0                      # every gate closed
    r4.zero_grad()
    r4.log_prob(closed).sum().backward()
    check("closed gates: no gradient on the density logits, gradient on the gate logits",
          bool((r4.mu.grad == 0).all()) and bool((r4.gate.grad != 0).all()))

    r3 = LinkRemover(RP.CELL_NAMES, RP.default_region_groups(), snap_rho_below=0.05, init_keep=0.99)
    with torch.no_grad():
        r3.mu.copy_(torch.linspace(-6, 4, r3.n_regions))
    det = torch.sigmoid(r3.mu.detach())
    snapped = r3.rho()
    zs = r3.sample(4096)
    z3, g3 = r3.split(zs)
    check("snap: deterministic rho < 0.05 -> 0, the rest unchanged; training draws never snapped",
          torch.equal(snapped, torch.where(det < 0.05, torch.zeros_like(det), det)) and bool((snapped == 0).any())
          and torch.equal(r3.rho(zs), g3 * torch.sigmoid(z3)) and bool(((r3.rho(zs) < 0.05) & (g3 > 0)).any()))

    print("\n=== gates only (learn_density: false) ===")
    r5 = LinkRemover(RP.CELL_NAMES, RP.default_region_groups(), init_keep=0.8, snap_rho_below=0.05,
                     learn_density=False)
    torch.manual_seed(0)
    d5 = r5.sample(2000)
    z5, g5 = r5.split(d5)
    check("gates only: draw rho == g (a kept region keeps every point), z unused",
          torch.equal(r5.rho(d5), g5) and bool((z5 == 0).all()))
    with torch.no_grad():
        r5.gate.copy_(torch.linspace(-3, 3, r5.n_regions))
    check("gates only: inference rho = 1[keep prob >= 0.5], no snap of the density",
          torch.equal(r5.rho(), (r5.keep_prob() >= 0.5).float()))
    lp5 = r5.log_prob(d5[:5])
    a5 = r5.gate.detach()
    ref5 = (g5[:5] * torch.nn.functional.logsigmoid(a5) + (1 - g5[:5]) * torch.nn.functional.logsigmoid(-a5)).sum(-1)
    r5.zero_grad()
    lp5.sum().backward()
    check("gates only: log P = sum log Bern(g); gradient on the gate logits only, mu frozen",
          torch.allclose(lp5, ref5, atol=1e-6) and r5.mu.grad is None and not r5.mu.requires_grad
          and bool((r5.gate.grad != 0).all()) and r5.phi_parameters() == [r5.gate])
    P5 = torch.as_tensor(np.asarray(z["data"]["point_cloud"][:1, :1024]), device=DEV)
    C5 = torch.as_tensor(np.asarray(z["data"]["point_link"][:1, :1024]), device=DEV).long()
    o5, l5 = thin_regions(P5, C5, torch.ones(1, r5.n_regions, device=DEV), r5.cell_to_region.to(DEV))[:2]
    check("gates only: an open gate (rho = 1) keeps the cloud unchanged", torch.equal(o5, P5) and int(l5[0]) == 1024)

    round2_module_checks(z)
    fixed_module_checks()
    fixed_rho_module_checks()
    all_links_checks(z)
    if "--no-policy" not in sys.argv:
        policy_checks(z)
        fixed_policy_checks(z)
        fixed_rho_policy_checks(z)

    print(f"\n{'ALL CHECKS PASSED' if not FAILS else 'FAILED: ' + ', '.join(FAILS)}")
    sys.exit(1 if FAILS else 0)


def round2_module_checks(z):
    import itertools
    groups = RP.default_region_groups()
    scene_groups = groups + [{"name": "scene", "cells": [NON_ROBOT]}]

    print("\n=== density only (learn_gates: false) ===")
    rd = LinkRemover(RP.CELL_NAMES, groups, learn_gates=False, init_rho=0.9)
    torch.manual_seed(0)
    dd = rd.sample(500)
    zd, gd = rd.split(dd)
    rd.zero_grad()
    rd.log_prob(dd[:5]).sum().backward()
    check("density only: every gate open, rho = sigmoid(z), gradient on mu only",
          bool((gd == 1).all()) and torch.equal(rd.rho(dd), torch.sigmoid(zd)) and rd.gate.grad is None
          and not rd.gate.requires_grad and bool((rd.mu.grad != 0).all()) and rd.phi_parameters() == [rd.mu]
          and bool((rd.keep_prob() == 1).all()))
    try:
        LinkRemover(RP.CELL_NAMES, groups, learn_gates=False, learn_density=False)
        check("raises on learn_gates = learn_density = false", False)
    except ValueError:
        check("raises on learn_gates = learn_density = false", True)

    print("\n=== scene region (label-free, density only) ===")
    try:
        LinkRemover(RP.CELL_NAMES, scene_groups)
        check("scene region with learned gates raises", False)
    except ValueError:
        check("scene region with learned gates raises", True)
    rs = LinkRemover(RP.CELL_NAMES, scene_groups, learn_gates=False, init_rho=0.9, scene_rho_min=0.1,
                     snap_rho_below=0.05)
    S, L = rs.scene_region, rs.n_regions
    N = 4096
    P = torch.as_tensor(np.asarray(z["data"]["point_cloud"][:6, :N]), device=DEV)
    Cc = torch.as_tensor(np.asarray(z["data"]["point_link"][:6, :N]), device=DEV).long()
    c2r = rs.cell_to_region.to(DEV)
    one = torch.ones(6, L, device=DEV)
    o1, l1 = thin_regions(P, Cc, one, c2r, scene_region=S)[:2]
    check("scene: rho = 1 everywhere -> cloud unchanged", torch.equal(o1, P) and bool((l1 == N).all()))
    rho = one.clone()
    rho[:, S] = 0.25
    o2, l2, n2, k2, _, keep = thin_regions(P, Cc, rho, c2r, scene_region=S, return_keep=True)
    n_scene = (Cc < 0).sum(1)
    link04 = (Cc >= 0) & (Cc <= 4)
    check("scene: rho_scene = 0.25 keeps round(0.25 n) scene points; robot and link0-4 untouched",
          torch.equal(k2[:, S], torch.floor(0.25 * n_scene.double() + 0.5).long())
          and int((keep & (Cc < 0)).sum()) == int(k2[:, S].sum()) and bool(keep[link04].all())
          and bool(keep[Cc > 4].all()), f"scene points {n_scene.tolist()} -> {k2[:, S].tolist()}")
    u = torch.rand(6, L, device=DEV)
    rlo, rhi = one.clone(), one.clone()
    rlo[:, S], rhi[:, S] = 0.2, 0.5
    klo = thin_regions(P, Cc, rlo, c2r, scene_region=S, u_region=u, return_keep=True)[5]
    khi = thin_regions(P, Cc, rhi, c2r, scene_region=S, u_region=u, return_keep=True)[5]
    check("scene: nested masks with shared starts (0.2 subset of 0.5)", bool((klo <= khi).all()))
    with torch.no_grad():
        rs.mu.fill_(-20.0)
    rs.clamp_()
    zlow = torch.cat([torch.full((3, L), -30.0), torch.ones(3, L)], -1)
    check("scene: rho floored at scene_rho_min (mu bound, every draw, inference never snaps it to 0)",
          abs(float(torch.sigmoid(rs.mu[S])) - 0.1) < 1e-6 and float(rs.mu[[l for l in range(L) if l != S]].max()) == -9.0
          and bool((rs.rho(zlow)[:, S] == 0.1).all()) and abs(float(rs.rho()[S]) - 0.1) < 1e-6
          and bool((rs.rho()[[l for l in range(L) if l != S]] == 0).all()))

    print("\n=== structured mix ===")
    masks = [[], ["link5", "link6", "link7"], ["link5", "link6", "link7", "hand_f1side", "hand_centre", "hand_f2side"], "all"]
    rm = LinkRemover(RP.CELL_NAMES, groups, learn_density=False, init_keep=0.8, mix_eps=0.25, mix_masks=masks)
    with torch.no_grad():
        rm.gate.copy_(torch.linspace(-1.5, 2.5, rm.n_regions))
    Lg = rm.n_regions
    pats = torch.tensor(list(itertools.product([0.0, 1.0], repeat=Lg)))
    allp = torch.cat([torch.zeros_like(pats), pats], -1)
    tot = torch.exp(rm.log_prob(allp)).sum().item()
    check("mix: P summed over all 256 gate patterns = 1", abs(tot - 1.0) < 1e-5, f"{tot:.8f}")
    torch.manual_seed(1)
    dm = rm.sample(200000)
    _, gm = rm.split(dm)
    mix = rm.mix_table("cpu")
    pi = torch.sigmoid(rm.gate.detach())
    bern = lambda m: float(torch.prod(torch.where(m > 0, pi, 1 - pi)))
    ok = True
    for i in range(mix.shape[0]):
        f_emp = (gm == mix[i]).all(-1).float().mean().item()
        f_exp = 0.75 * bern(mix[i]) + 0.25 / mix.shape[0]
        ok &= abs(f_emp - f_exp) < 4 * math.sqrt(f_exp * (1 - f_exp) / 200000) + 1e-4
    marg = gm.mean(0)
    marg_exp = 0.75 * pi + 0.25 * mix.mean(0)
    ok &= bool((marg - marg_exp).abs().max() < 0.005)
    check("mix: sampled frequencies match (1-eps) Bern + eps/S per mask, and the gate marginals", ok,
          f"max marginal error {(marg - marg_exp).abs().max():.4f}")
    rm.zero_grad()
    rm.log_prob(dm[:64]).sum().backward()
    check("mix: gradient reaches the gate logits only", rm.mu.grad is None and bool((rm.gate.grad != 0).any()))
    # mix off == the pre-mix sampler / log_prob (reference formulas inline)
    r0 = LinkRemover(RP.CELL_NAMES, groups, init_keep=0.8, sigma=0.7)
    torch.manual_seed(5)
    d0 = r0.sample(1000)
    torch.manual_seed(5)
    zr = r0.mu[None].detach() + r0.sigma * torch.randn(1000, Lg)
    gr = (torch.rand(1000, Lg) < torch.sigmoid(r0.gate.detach())[None]).to(zr.dtype)
    lg = gr * torch.nn.functional.logsigmoid(r0.gate.detach()) + (1 - gr) * torch.nn.functional.logsigmoid(-r0.gate.detach())
    lz = -0.5 * ((zr - r0.mu.detach()) / r0.sigma) ** 2 - torch.log(r0.sigma) - 0.5 * math.log(2 * math.pi)
    check("mix off: sample and log_prob bit-identical to the pre-mix formulas",
          torch.equal(d0, torch.cat([zr, gr], -1)) and torch.equal(r0.log_prob(d0).detach(), (lg + gr * lz).sum(-1)))


SPEC = ["link5", "link6"]


def fixed_module_checks():
    print("\n=== fixed mask (specialist) ===")
    groups = RP.default_region_groups()
    rf = LinkRemover(RP.CELL_NAMES, groups, learn_density=False, fixed_removed=SPEC)
    want = torch.tensor([0.0 if n in SPEC else 1.0 for n in rf.region_names])
    d = rf.sample(500)
    check("fixed: deterministic rho == the mask exactly (listed 0, rest 1); every draw is the mask",
          torch.equal(rf.rho(), want) and bool((rf.split(d)[1] == want).all()) and torch.equal(rf.rho(d), want.expand(500, -1)))
    check("fixed: gate logits at -/+ gate_max, frozen; phi_parameters() empty",
          torch.equal(rf.gate.detach(), (want * 2 - 1) * rf.gate_max) and not rf.gate.requires_grad
          and not rf.mu.requires_grad and rf.phi_parameters() == [])
    cases = {
        "unknown region name": lambda: LinkRemover(RP.CELL_NAMES, groups, learn_density=False, fixed_removed=["link9"]),
        "fixed_removed with learn_density": lambda: LinkRemover(RP.CELL_NAMES, groups, fixed_removed=SPEC),
        "fixed_removed with mix": lambda: LinkRemover(RP.CELL_NAMES, groups, learn_density=False, fixed_removed=SPEC,
                                                      mix_eps=0.25, mix_masks=[[], "all"]),
        "log_prob of a fixed remover": lambda: rf.log_prob(d),
    }
    for nm, fn in cases.items():
        try:
            fn()
            check(f"fixed: raises on {nm}", False)
        except (ValueError, RuntimeError, KeyError) as e:
            check(f"fixed: raises on {nm}", True, type(e).__name__)
    # off: no fixed_removed -> the same parameters and the same sampler as before
    torch.manual_seed(4)
    a = LinkRemover(RP.CELL_NAMES, groups, learn_density=False, init_keep=0.5)
    da = a.sample(300)
    torch.manual_seed(4)
    b = LinkRemover(RP.CELL_NAMES, groups, learn_density=False, init_keep=0.5, fixed_removed=None)
    db = b.sample(300)
    check("fixed_removed=None: parameters and draws identical to the plain remover",
          torch.equal(da, db) and all(torch.equal(x, y) for x, y in zip(a.state_dict().values(), b.state_dict().values()))
          and a.phi_parameters() == [a.gate])


FIXED_RHO = {"link5": 0.6, "link6": 0.03, "link7": 0.9, "hand_f1side": 0.8, "hand_centre": 0.85,
             "hand_f2side": 0.75, "finger1": 0.5, "finger2": 1.0, "scene": 0.25}   # link6 below the snap


def fixed_rho_module_checks():
    print("\n=== fixed densities (density specialist) ===")
    groups = RP.default_region_groups() + [{"name": "scene", "cells": [NON_ROBOT]}]
    kw = dict(learn_gates=False, snap_rho_below=0.05)
    rf = LinkRemover(RP.CELL_NAMES, groups, fixed_rho=FIXED_RHO, **kw)
    table = torch.tensor([FIXED_RHO[n] for n in rf.region_names])
    det = torch.where(table < 0.05, torch.zeros_like(table), table)
    d = rf.sample(300)
    check("fixed_rho: density() == table, deterministic rho == table snapped (< 0.05 -> 0), every draw == table",
          torch.equal(rf.density(), table) and torch.equal(rf.rho(), det)
          and torch.equal(rf.rho(d), table.expand(300, -1)) and bool((rf.split(d)[1] == 1).all()))
    check("fixed_rho: nothing learnable (mu frozen, phi_parameters() empty, is_fixed)",
          not rf.mu.requires_grad and not rf.gate.requires_grad and rf.phi_parameters() == [] and rf.is_fixed)
    rl = LinkRemover(RP.CELL_NAMES, groups, fixed_rho=list(table.tolist()), **kw)
    check("fixed_rho: a list in region order == the dict", torch.equal(rl.rho(), rf.rho()))
    bad_scene = dict(FIXED_RHO, scene=0.05)
    cases = {
        "learned gates": lambda: LinkRemover(RP.CELL_NAMES, RP.default_region_groups(), fixed_rho={k: v for k, v in FIXED_RHO.items() if k != "scene"}),
        "unknown region": lambda: LinkRemover(RP.CELL_NAMES, groups, fixed_rho=dict(FIXED_RHO, link9=0.5), **kw),
        "missing region": lambda: LinkRemover(RP.CELL_NAMES, groups, fixed_rho={k: v for k, v in FIXED_RHO.items() if k != "link5"}, **kw),
        "rho outside [0, 1]": lambda: LinkRemover(RP.CELL_NAMES, groups, fixed_rho=dict(FIXED_RHO, link5=1.2), **kw),
        "scene below scene_rho_min": lambda: LinkRemover(RP.CELL_NAMES, groups, fixed_rho=bad_scene, **kw),
        "wrong list length": lambda: LinkRemover(RP.CELL_NAMES, groups, fixed_rho=[0.5] * 3, **kw),
        "log_prob of a fixed remover": lambda: rf.log_prob(d),
    }
    for nm, fn in cases.items():
        try:
            fn()
            check(f"fixed_rho: raises on {nm}", False)
        except (ValueError, RuntimeError, KeyError) as e:
            check(f"fixed_rho: raises on {nm}", True, type(e).__name__)
    torch.manual_seed(5)
    a = LinkRemover(RP.CELL_NAMES, groups, learn_gates=False, init_rho=0.5)
    da = a.sample(300)
    torch.manual_seed(5)
    b = LinkRemover(RP.CELL_NAMES, groups, learn_gates=False, init_rho=0.5, fixed_rho=None)
    db = b.sample(300)
    check("fixed_rho=None: parameters and draws identical to the plain density remover",
          torch.equal(da, db) and all(torch.equal(x, y) for x, y in zip(a.state_dict().values(), b.state_dict().values()))
          and a.phi_parameters() == [a.mu] and not a.is_fixed)


def fixed_rho_policy_checks(z):
    """The density specialist policy on real windows: every training and eval cloud keeps
    exactly round(rho * n) points per region (the table, snapped), no fill, nothing learned."""
    import hydra
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf
    from r3d.common.link_remover_training import RemoverController
    print("\n=== policy: density specialist (fixed_rho) ===")
    N = 4096
    tbl = ",".join(f"{k}:{v}" for k, v in FIXED_RHO.items())
    with initialize_config_dir(config_dir=str(ROOT / "R3D" / "r3d" / "config"), version_base=None):
        cfg = compose(config_name="r3d_robotwin2_remover", overrides=[
            "task=mimicgen_sim_remover", "task_name=square_d2", "setting=agentview_p4096-200",
            f"task.num_points={N}", "task.num_group=64", "policy.link_remover.learn_gates=false",
            "policy.link_remover.region_groups=default+scene", f"policy.link_remover.fixed_rho={{{tbl}}}"])
    torch.manual_seed(321)
    cwd = os.getcwd()
    os.chdir(ROOT)
    try:
        pol = hydra.utils.instantiate(cfg.policy)
    finally:
        os.chdir(cwd)
    pol.set_normalizer(make_normalizer(z))
    pol = pol.to(DEV)
    rem = pol.link_remover
    want = rem.rho().to(DEV)
    check("fixed_rho policy: hydra dict override -> deterministic rho == the table snapped",
          torch.allclose(want.cpu(), torch.tensor([0.0 if FIXED_RHO[n] < 0.05 else FIXED_RHO[n] for n in rem.region_names])),
          " ".join(f"{n} {v:.2f}" for n, v in zip(rem.region_names, want.tolist())))
    seen = []
    orig = pol._thin_obs
    def rec(obs, cells, rho, **kw):
        res = orig(obs, cells, rho, **kw)
        seen.append((rho.detach(), res))
        return res
    pol._thin_obs = rec

    def audit():
        ok, nf = True, 0
        for rho, (tobs, (out, ln, n, k, filled)) in seen:
            exp = torch.floor(want.double()[None, None] * n + 0.5).long().clamp(max=n)
            ok &= bool((rho == want[None]).all()) and torch.equal(k, exp) and bool((filled == 0).all())
            nf += n.shape[0] * n.shape[1]
        return ok, nf

    ctl = RemoverController(pol, OmegaConf.create(dict(
        lr=3e-3, phi_every=1, warmup_frac=0.2, finetune_frac=0.1, sigma_start=1.0, sigma_end=0.2,
        diag_every=1, diag_batches=2, diag_masks=4)), num_epochs=10)
    check("fixed_rho: controller has no phi optimizer; phase 'fixed' in every epoch",
          ctl.optimizer is None and all(ctl.phase_of(e) == "fixed" for e in range(10)))
    pol.train()
    phi_before = {k: v.detach().clone() for k, v in rem.state_dict().items()}
    B = 8
    batch = make_batch(z, N, B=B, seed=41)
    ctl.begin_batch(pol, 5, 0.0)
    ctl.remember(batch, pol.n_obs_steps)
    pol.remover_mode = "sample"                 # stale mode: must be ignored
    loss, ld = pol.compute_loss(batch)
    loss.backward()
    log = ctl.after_opt_step(pol, 4, contextlib_null)
    ok_train, nf = audit()
    grads = [n for n, p_ in pol.named_parameters() if n.startswith("link_remover.") and p_.grad is not None]
    check("fixed_rho: training clouds keep round(rho * n) per region (table), no fill; phi unchanged, no phi step",
          ok_train and not grads and all(torch.equal(phi_before[k], rem.state_dict()[k]) for k in phi_before)
          and log.get("remover_phase") == 3, f"{nf} frames")
    seen.clear()
    pol.eval()
    with torch.no_grad():
        act = pol.predict_action({k: v for k, v in make_batch(z, N, B=B, seed=42)["obs"].items() if k != "point_label"})
    ok_eval, nf = audit()
    check("fixed_rho: eval clouds (predict_action) keep round(rho * n) per region", ok_eval and act["action"].shape[0] == B,
          f"{nf} frames")
    pol._thin_obs = orig
    del pol
    torch.cuda.empty_cache()


def all_links_checks(z):
    """region_groups all_links (13 regions, gates for link0..link4 too): closing link0..7 IS the
    oracle arm cut (label 5), bit for bit -- a clean removal is reachable."""
    print("\n=== all_links: 13 regions, the arm cut is reachable ===")
    groups = RP.all_links_region_groups()
    ra = LinkRemover(RP.CELL_NAMES, groups, learn_density=False, init_keep=0.5)
    check("all_links: 13 regions, link0..link7 + hand x3 + fingers x2, every cell of the arm and gripper in one",
          ra.n_regions == 13 and ra.region_names[:8] == [f"link{i}" for i in range(8)]
          and bool((ra.cell_to_region >= 0).all()))
    ok, worst = True, 0
    for N in (1024, 4096):
        pc, cell, lab = load_windows(z, N, n_win=20, seed=7)
        B = len(pc)
        P, Cc = torch.as_tensor(pc, device=DEV), torch.as_tensor(cell, device=DEV)
        rho = torch.tensor([0.0 if n.startswith("link") else 1.0 for n in ra.region_names], device=DEV).expand(B, -1)
        out, ln, _, _, _ = thin_window(P, Cc, rho, ra.cell_to_region.to(DEV))
        for b in range(B):
            for f in range(2):
                ref, nref = cut_points_by_label(pc[b, f], lab[b, f], [5])
                ok &= nref == int(ln[b, f]) and np.array_equal(ref, out[b, f].cpu().numpy())
                worst = max(worst, int(((cell[b, f] >= 0) & (cell[b, f] <= 4)).sum()))
    check("all_links: closing link0..7 == label-5 arm cut, padded arrays bit-identical (N 1024, 4096)", ok,
          f"most link0-4 points in a checked frame: {worst}")


def fixed_policy_checks(z):
    """The specialist policy on real windows: the clouds theta trains on and the clouds
    inference sees, audited point by point against the link labels."""
    import hydra
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf
    from r3d.common.link_remover_training import RemoverController
    print("\n=== policy: fixed-mask specialist (link5 + link6 removed) ===")
    N = 1024
    with initialize_config_dir(config_dir=str(ROOT / "R3D" / "r3d" / "config"), version_base=None):
        cfg = compose(config_name="r3d_robotwin2_remover", overrides=[
            "task=mimicgen_sim_remover", "task_name=square_d2", "setting=agentview_p4096-200",
            f"task.num_points={N}", "task.num_group=64", "policy.link_remover.learn_density=false",
            "policy.link_remover.fixed_removed=[link5,link6]"])
    torch.manual_seed(123)
    cwd = os.getcwd()
    os.chdir(ROOT)
    try:
        pol = hydra.utils.instantiate(cfg.policy)
    finally:
        os.chdir(cwd)
    pol.set_normalizer(make_normalizer(z))
    pol = pol.to(DEV)
    rem = pol.link_remover
    c2r = rem.cell_to_region.cpu().numpy()
    removed = [rem.region_names.index(n) for n in SPEC]
    B = 8

    seen = []
    orig = pol._thin_obs
    def rec(obs, cells, rho, **kw):
        res = orig(obs, cells, rho, **kw)
        seen.append((obs["point_cloud"][:, :pol.n_obs_steps].detach(), cells[:, :pol.n_obs_steps].detach(), rho.detach(), res))
        return res
    pol._thin_obs = rec

    def audit():
        """every recorded thinning: kept == every valid point not in link5 / link6, in order."""
        ok, n_rm, n_frames = True, 0, 0
        for P, Cc, rho, (tobs, (out, ln, n, k, filled)) in seen:
            ok &= bool((filled == 0).all())
            for b in range(P.shape[0]):
                for f in range(P.shape[1]):
                    cell = Cc[b, f].cpu().numpy()
                    reg = np.where(cell >= 0, c2r[np.clip(cell, 0, None)], -1)
                    keep = ~np.isin(reg, removed)
                    m = int(ln[b, f])
                    ok &= m == int(keep.sum()) and torch.equal(out[b, f, :m], P[b, f][torch.as_tensor(keep, device=DEV)])
                    ok &= torch.equal(tobs["point_cloud"][b, f, :m], out[b, f, :m]) and int(tobs["point_cloud_len"][b, f]) == m
                    n_rm += int((~keep).sum())
                    n_frames += 1
        return ok, n_rm, n_frames

    # training: the controller sets the mode; a stale "sample" mode is ignored by compute_loss
    ctl = RemoverController(pol, OmegaConf.create(dict(
        lr=3e-3, phi_every=1, warmup_frac=0.2, finetune_frac=0.1, sigma_start=1.0, sigma_end=0.2,
        diag_every=1, diag_batches=2, diag_masks=4)), num_epochs=10)
    check("fixed: controller has no phi optimizer; phase 'fixed' in every epoch",
          ctl.optimizer is None and all(ctl.phase_of(e) == "fixed" for e in range(10)))
    pol.train()
    phi_before = {k: v.detach().clone() for k, v in rem.state_dict().items()}
    batch = make_batch(z, N, B=B, seed=31)
    ctl.begin_batch(pol, 5, 0.0)
    mode_ok = pol.remover_mode == "deterministic"
    ctl.remember(batch, pol.n_obs_steps)
    pol.remover_mode = "sample"
    loss, ld = pol.compute_loss(batch)
    loss.backward()
    log = ctl.after_opt_step(pol, 4, contextlib_null)
    ok_train, n_rm, nf = audit()
    check("fixed: training clouds have exactly link5 + link6 removed, every other point kept, in order",
          ok_train and n_rm > 0, f"{nf} frames, {n_rm} points removed")
    grads = [n for n, p_ in pol.named_parameters() if n.startswith("link_remover.") and p_.grad is not None]
    phi_after = rem.state_dict()
    check("fixed: no phi parameter changes over a controller step (no grad, no phi step, no phi log)",
          mode_ok and not grads and all(torch.equal(phi_before[k], phi_after[k]) for k in phi_before)
          and log.get("remover_phase") == 3 and not any(k.startswith("remover_grad_") for k in log))
    # eval: predict_action sees the same mask
    seen.clear()
    pol.eval()
    with torch.no_grad():
        act = pol.predict_action({k: v for k, v in make_batch(z, N, B=B, seed=32)["obs"].items() if k != "point_label"})
    ok_eval, n_rm, nf = audit()
    check("fixed: eval clouds (predict_action) have exactly link5 + link6 removed, every other point kept",
          ok_eval and n_rm > 0 and act["action"].shape[0] == B, f"{nf} frames, {n_rm} points removed")
    pol._thin_obs = orig
    # diagnostics: mechanism only
    out = ctl.end_epoch(pol, [make_batch(z, N, B=B, seed=33)], DEV, contextlib_null, 0, {})
    tok = lambda r: out.get(f"mech/deterministic/tokens/{r}", float("nan"))
    check("fixed: diagnostics = mechanism only (no held-out check, no diagnostic B); 0 tokens on link5 / link6",
          any(k.startswith("mech/deterministic/") for k in out) and not any(k.startswith("mech/sampled/") for k in out)
          and [k for k in out if k.startswith("remover_check/")] == ["remover_check/n_batches"]
          and tok("link5") == 0.0 and tok("link6") == 0.0,
          f"uncut link5/6 tokens {out.get('mech/none/tokens/link5', float('nan')):.1f}/{out.get('mech/none/tokens/link6', float('nan')):.1f}, "
          f"nut cov {out.get('mech/none/coverage/nut', float('nan')):.2f} -> {out.get('mech/deterministic/coverage/nut', float('nan')):.2f}")
    del pol
    torch.cuda.empty_cache()


def contextlib_null():
    import contextlib
    return contextlib.nullcontext()


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


def make_normalizer(z):
    """RobotwinSimDataset.get_normalizer on the zarr's own arrays, without loading the clouds."""
    from r3d.dataset.robotwin_sim_dataset import RobotwinSimDataset
    ds = RobotwinSimDataset.__new__(RobotwinSimDataset)
    ds.use_target_ee = False
    ds.replay_buffer = {"action": z["data/action"][:], "state": z["data/state"][:], "point_cloud": z["data/point_cloud"]}
    ds.pcd_crop_bbox = np.asarray(z["meta"].attrs["pcd_crop_bbox"], dtype=np.float64)
    return ds.get_normalizer()


def make_batch(z, N, B, horizon=16, To=2, seed=0):
    """B training-style windows straight from the zarr (no augmentation)."""
    rng = np.random.default_rng(seed)
    ends = z["meta/episode_ends"][:]
    starts = np.concatenate([[0], ends[:-1]])
    eps = rng.choice(len(ends), B, replace=False)
    t0 = [int(rng.integers(starts[e], ends[e] - horizon)) for e in eps]
    obs_idx = [t + i for t in t0 for i in range(To)]
    pc = z["data/point_cloud"].get_orthogonal_selection((obs_idx, slice(0, N))).reshape(B, To, N, 6)
    link = z["data/point_link"].get_orthogonal_selection((obs_idx, slice(0, N))).reshape(B, To, N)
    plab = z["data/point_label"].get_orthogonal_selection((obs_idx, slice(0, N))).reshape(B, To, N)
    state = z["data/state"].get_orthogonal_selection((obs_idx,)).reshape(B, To, -1)
    act = np.stack([z["data/action"][t:t + horizon] for t in t0])
    t = lambda x, dt=torch.float32: torch.as_tensor(np.ascontiguousarray(x), dtype=dt, device=DEV)
    return {"obs": {"point_cloud": t(pc), "agent_pos": t(state), "point_link": t(link, torch.long),
                    "point_label": t(plab, torch.int8)},
            "action": t(act)}


def controller_step(pol, batch, z_unused=None):
    """One real phi step through RemoverController (phase phi, opt_step % phi_every == 0)."""
    import contextlib
    from omegaconf import OmegaConf
    from r3d.common.link_remover_training import RemoverController
    ctl = RemoverController(pol, OmegaConf.create(dict(
        lr=3e-3, phi_every=1, warmup_frac=0.1, finetune_frac=0.1, sigma_start=1.0, sigma_end=0.2,
        diag_every=10, diag_batches=2, diag_masks=4)), num_epochs=10)
    ctl.begin_batch(pol, 5, 0.0)
    ctl.remember(batch, pol.n_obs_steps)
    before = [p.detach().clone() for p in pol.link_remover.phi_parameters()]
    log = ctl.after_opt_step(pol, 4, contextlib.nullcontext)
    moved = any(not torch.equal(b, p.detach()) for b, p in zip(before, pol.link_remover.phi_parameters()))
    return log, moved


def policy_checks(z):
    import hydra
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf
    import r3d.model.vision.pointnet_extractor as PE
    import r3d.policy.dp3_link_remover as DLR
    if not OmegaConf.has_resolver("eval"):
        OmegaConf.register_new_resolver("eval", eval, replace=True)
    N = 1024
    with initialize_config_dir(config_dir=str(ROOT / "R3D" / "r3d" / "config"), version_base=None):
        cfg = compose(config_name="r3d_robotwin2", overrides=[
            "task=mimicgen_sim", "task_name=square_d2", "setting=agentview_p4096-200",
            f"task.num_points={N}", "task.num_group=64"])
    pcfg = OmegaConf.to_container(cfg.policy, resolve=True)
    normalizer = make_normalizer(z)

    def build(target, link_remover="absent"):
        c = dict(pcfg, _target_=target)
        if link_remover != "absent":
            c["link_remover"] = link_remover
        torch.manual_seed(123)
        cwd = os.getcwd()
        os.chdir(ROOT)
        try:
            pol = hydra.utils.instantiate(c)
        finally:
            os.chdir(cwd)
        pol.set_normalizer(normalizer)
        return pol.to(DEV)

    print("\n=== policy: remover off == baseline DP3 ===")
    base = build("r3d.policy.dp3.DP3")
    off = build("r3d.policy.dp3_link_remover.DP3LinkRemover", None)
    off.load_state_dict(base.state_dict())
    batch = make_batch(z, N, B=4)
    plain = {"obs": {k: v for k, v in batch["obs"].items() if k not in ("point_link", "point_label")},
             "action": batch["action"]}
    same = True
    for mode in ("train", "eval"):
        base.train(mode == "train")
        off.train(mode == "train")
        torch.manual_seed(7)
        la, da = base.compute_loss(plain)
        torch.manual_seed(7)
        lb, db = off.compute_loss(plain)
        same &= torch.equal(la, lb) and da == db
    check("remover off: compute_loss bit-identical to DP3 (train and eval mode)", same, f"loss {la.item():.6f}")
    base.eval()
    off.eval()
    with torch.no_grad():
        torch.manual_seed(8)
        pa = base.predict_action(plain["obs"])
        torch.manual_seed(8)
        pb = off.predict_action(plain["obs"])
    check("remover off: predict_action bit-identical to DP3",
          all(torch.equal(pa[k], pb[k]) for k in pa) and pa.keys() == pb.keys())
    try:
        off.compute_loss(batch)
        check("remover off: an obs carrying point_link raises (config mismatch)", False)
    except ValueError:
        check("remover off: an obs carrying point_link raises (config mismatch)", True)
    del base, off
    torch.cuda.empty_cache()

    print("\n=== policy: common-random-numbers reward ===")
    pol = build("r3d.policy.dp3_link_remover.DP3LinkRemover", {"enabled": True, "k_masks": 2})
    pol.train()
    rem = pol.link_remover
    B, L = 6, rem.n_regions
    batch = make_batch(z, N, B=B, seed=1)
    torch.manual_seed(0)
    zd = rem.sample(B * 3).view(B, 3, -1)
    zd[:, 1] = zd[:, 0]                                      # mask 1 == mask 0, mask 2 differs
    seeds = torch.randint(0, 2 ** 62, (B,))
    r1 = pol.crn_reward(batch, zd, seeds)
    r2 = pol.crn_reward(batch, zd, seeds)
    r3 = pol.crn_reward(batch, zd, torch.randint(0, 2 ** 62, (B,)))
    diff_same = (r1[:, 0] - r1[:, 1]).abs().max().item()
    check("same mask twice -> reward difference exactly 0", diff_same == 0.0, f"max |diff| {diff_same}")
    check("same masks + seeds twice -> identical rewards", torch.equal(r1, r2))
    check("different masks -> different rewards (the mask is seen)", bool((r1[:, 0] != r1[:, 2]).all()),
          f"|diff| mean {(r1[:, 0] - r1[:, 2]).abs().mean().item():.2e}")
    check("other seeds -> other rewards (the shared draws are random, not fixed)", not torch.equal(r1, r3))
    check("reward pass leaves the training flag as it was", pol.training)
    # the shared token-FPS start: a NON-ROBOT point (cell -1), the same physical point in every mask
    draws = pol.crn_draws(seeds, B, 2, batch["action"].shape[1], DEV)
    cells = batch["obs"]["point_link"]
    obs_ = {k: v for k, v in batch["obs"].items() if k not in ("point_link", "point_label")}
    starts_xyz, start_ok = [], True
    for k in range(3):
        tobs, res = pol._thin_obs(obs_, cells, rem.rho(zd[:, k]), u_region=draws["u_region"],
                                  u_fill=draws["u_fill"], return_keep=True)
        st = pol._encoder_start(cells[:, :2], res[5], draws["u_enc"]).view(B, 2)
        xyz = torch.stack([tobs["point_cloud"][b, f, st[b, f], :3] for b in range(B) for f in range(2)])
        starts_xyz.append(xyz)
        for b in range(B):
            for f in range(2):
                orig = (batch["obs"]["point_cloud"][b, f, :, :3] == xyz[b * 2 + f]).all(-1)
                start_ok &= bool((cells[b, f][orig] < 0).all()) and bool(orig.any())
    check("CRN token-FPS start: a non-robot point (not link0-4), identical physical point in every mask",
          start_ok and all(torch.equal(starts_xyz[0], x) for x in starts_xyz[1:]))

    print("\n=== policy: theta pass keeps independent random starts ===")
    seen = {"fps": [], "thin": []}
    orig_fps, orig_thin = PE.fps, DLR.thin_window

    def rec_fps(*a, **kw):
        seen["fps"].append((kw.get("start_indices"), kw.get("perm")))
        return orig_fps(*a, **kw)

    def rec_thin(*a, **kw):
        seen["thin"].append((kw.get("u_region"), kw.get("u_fill"), kw.get("random_start")))
        return orig_thin(*a, **kw)

    PE.fps, DLR.thin_window = rec_fps, rec_thin
    try:
        torch.manual_seed(1)
        pol.compute_loss(batch)
        theta = dict(seen)
        seen["fps"], seen["thin"] = [], []
        pol.crn_reward(batch, zd[:, :2], seeds)
        reward = dict(seen)
    finally:
        PE.fps, DLR.thin_window = orig_fps, orig_thin
    check("theta pass: token FPS gets no shared start / shuffle (draws its own)",
          len(theta["fps"]) > 0 and all(s is None and p is None for s, p in theta["fps"]))
    check("theta pass: within-region FPS gets no shared starts, random_start=True",
          len(theta["thin"]) == 1 and theta["thin"][0] == (None, None, True))
    check("reward pass: token FPS and within-region FPS DO get the shared draws",
          all(s is not None and p is not None for s, p in reward["fps"])
          and all(u is not None for u, _, _ in reward["thin"]))
    # and the theta pass's random draws really differ between calls
    torch.manual_seed(1)
    l1, d1 = pol.compute_loss(batch)
    torch.manual_seed(2)
    l2, d2 = pol.compute_loss(batch)
    torch.manual_seed(1)
    l3, _ = pol.compute_loss(batch)
    check("theta pass: other RNG state -> other loss; same RNG state -> same loss",
          not torch.equal(l1, l2) and torch.equal(l1, l3))
    # held-out memorisation check: with the SAME seeds, the same batch must agree with itself exactly
    sd = torch.randint(0, 2 ** 62, (B,))
    hc_self = pol.heldout_check(batch, batch, n_masks=6, seeds=[sd, sd, sd])
    check("heldout_check: same batch + same seeds -> rank 1, grad cosine 1, signs agree (floor and held-out)",
          all(abs(hc_self[f"remover_check/{t}_mask_rank_spearman"] - 1) < 1e-6
              and abs(hc_self[f"remover_check/{t}_grad_cosine"] - 1) < 1e-4
              and hc_self[f"remover_check/{t}_grad_sign_agreement"] == 1.0 for t in ("floor", "heldout")))
    hc = pol.heldout_check(batch, make_batch(z, N, B=B, seed=9), n_masks=6)
    check("heldout_check: fresh seeds, another batch -> finite metrics (untrained policy: noise)",
          all(np.isfinite(v) for v in hc.values()),
          ", ".join(f"{k.split('/')[1]} {v:+.2f}" for k, v in hc.items()))
    # diagnostic B: full DDIM sampling under the same shared draws; identical masks -> identical scores
    ab_same = pol.rank_check_ab(batch, z=rem.sample(1).expand(3, -1).contiguous())
    ab = pol.rank_check_ab(batch, n_masks=4)
    check("diagnostic B: identical masks -> identical A and identical B scores; otherwise finite",
          len(set(ab_same["a_scores"])) == 1 and len(set(ab_same["b_scores"])) == 1
          and np.isfinite(ab["ab_rank_spearman"]),
          f"untrained: A/B rank corr {ab['ab_rank_spearman']:+.2f}")
    # mechanism readout: 64 tokens per cloud; with every gate closed no token sits on a region
    mech = pol.mechanism_check(batch, masks=("none", "deterministic"))
    G = int(pol.num_tokens)
    tok = lambda m: sum(v for k, v in mech.items() if k.startswith(f"mech/{m}/tokens/"))
    gate_save = rem.gate.detach().clone()
    with torch.no_grad():
        rem.gate.fill_(-5.0)
    closed = pol.mechanism_check(batch, masks=("deterministic",))
    with torch.no_grad():
        rem.gate.copy_(gate_save)
    reg_tok = sum(closed[f"mech/deterministic/tokens/{n}"] for n in rem.region_names)
    check("mechanism_check: tokens per cloud sum to num_group; all gates closed -> 0 region tokens",
          abs(tok("none") - G) < 1e-6 and abs(tok("deterministic") - G) < 1e-6 and reg_tok == 0.0,
          f"uncut nut cov {mech.get('mech/none/coverage/nut', float('nan')):.2f}, all-closed nut cov "
          f"{closed.get('mech/deterministic/coverage/nut', float('nan')):.2f}")
    surr, stats = pol.remover_loss(batch)
    check("remover_loss: per-region effect stats (gate effect, density slope, counts) logged",
          all(f"remover_n_closed/{n}" in stats for n in rem.region_names)
          and any(k.startswith("remover_effect_gate/") for k in stats))
    surr.backward()
    grads = {n: p.grad for n, p in pol.named_parameters() if p.grad is not None}
    check("remover_loss: gradient reaches phi (density mu, gate logits) only",
          sorted(grads) == ["link_remover.gate", "link_remover.mu"]
          and all(bool((g != 0).any()) for g in grads.values()))

    print("\n=== policy: gates only (learn_density: false) ===")
    del pol
    torch.cuda.empty_cache()
    pg = build("r3d.policy.dp3_link_remover.DP3LinkRemover",
               {"enabled": True, "k_masks": 4, "learn_density": False})
    pg.train()
    rg = pg.link_remover
    batch = make_batch(z, N, B=B, seed=1)
    surr, stats = pg.remover_loss(batch)
    surr.backward()
    grads = {n: p.grad for n, p in pg.named_parameters() if p.grad is not None}
    check("gates only: remover_loss gradient on the gate logits only, no density slope logged",
          sorted(grads) == ["link_remover.gate"] and bool((grads["link_remover.gate"] != 0).any())
          and not any(k.startswith("remover_effect_density/") for k in stats)
          and all(stats[f"remover_rho/{n}"] == 1.0 for n in rg.region_names))
    hg = pg.heldout_check(batch, make_batch(z, N, B=B, seed=9), n_masks=6)
    check("gates only: heldout_check runs (gradient over the gate logits)",
          all(np.isfinite(v) for v in hg.values()))
    from r3d.common.link_remover_training import RemoverController
    ctl = RemoverController(pg, OmegaConf.create(dict(
        lr=3e-3, phi_every=4, warmup_frac=0.2, finetune_frac=0.1, sigma_start=1.0, sigma_end=0.2,
        diag_every=10, diag_batches=4, diag_masks=8)), num_epochs=10)
    ps = [p for g in ctl.optimizer.param_groups for p in g["params"]]
    check("gates only: the phi optimizer holds the gate logits only", len(ps) == 1 and ps[0] is rg.gate)

    print("\n=== controller: finetune_mask ===")
    from r3d.common.link_remover_training import RemoverController as RC
    base_cfg = dict(lr=3e-3, phi_every=4, warmup_frac=0.2, finetune_frac=0.1, sigma_start=1.0, sigma_end=0.2,
                    diag_every=10, diag_batches=4, diag_masks=8)
    gate_before = rg.gate.detach().clone()
    c_off = RC(pg, OmegaConf.create(base_cfg), num_epochs=10)
    c_off.begin_batch(pg, 9, 0.0)
    off_ok = torch.equal(rg.gate.detach(), gate_before) and pg.remover_mode == "deterministic"
    c_on = RC(pg, OmegaConf.create(dict(base_cfg, finetune_mask=["link5", "link6", "link7"])), num_epochs=10)
    c_on.begin_batch(pg, 5, 0.0)
    phi_untouched = torch.equal(rg.gate.detach(), gate_before)
    c_on.begin_batch(pg, 9, 0.0)
    want = torch.tensor([0.0 if n in ("link5", "link6", "link7") else 1.0 for n in rg.region_names], device=rg.gate.device)
    c_none = RC(pg, OmegaConf.create(dict(base_cfg, finetune_mask=[])), num_epochs=10)
    on_ok = torch.equal(rg.rho(), want) and pg.remover_mode == "deterministic"
    c_none.begin_batch(pg, 9, 0.0)
    check("finetune_mask: null leaves the gates alone; a mask is set only on entering the fine-tune "
          "(deterministic rho == that mask); [] keeps everything",
          off_ok and phi_untouched and on_ok and bool((rg.rho() == 1).all()))
    with torch.no_grad():
        rg.gate.copy_(gate_before)

    print("\n=== policy: decoupled theta exploration (theta_keep) ===")
    del pg
    torch.cuda.empty_cache()
    pt = build("r3d.policy.dp3_link_remover.DP3LinkRemover",
               {"enabled": True, "k_masks": 4, "learn_density": False, "init_keep": 0.95, "theta_keep": 0.5})
    pt.train()
    pt.remover_mode = "sample"
    seen = []
    orig = pt._thin_obs
    pt._thin_obs = lambda obs, cells, rho, **kw: (seen.append(rho.detach().clone()), orig(obs, cells, rho, **kw))[1]
    for i in range(6):
        pt.compute_loss(make_batch(z, N, B=B, seed=20 + i))
    pt._thin_obs = orig
    frac = torch.cat(seen).gt(0).float().mean().item()
    zt = pt.link_remover.sample(4000)
    check("theta_keep 0.5: theta-pass gates open ~50% while phi's keep prob is 0.95",
          abs(frac - 0.5) < 0.1 and abs(pt.link_remover.split(zt)[1].mean().item() - 0.95) < 0.02, f"theta open {frac:.2f}")

    print("\n=== policy: density only + scene region ===")
    del pt
    torch.cuda.empty_cache()
    ps_ = build("r3d.policy.dp3_link_remover.DP3LinkRemover",
                {"enabled": True, "k_masks": 4, "learn_gates": False, "init_rho": 0.9,
                 "region_groups": "default+scene", "snap_rho_below": 0.05})
    ps_.train()
    rem = ps_.link_remover
    batch = make_batch(z, N, B=B, seed=1)
    torch.manual_seed(3)
    zd = rem.sample(B * 3).view(B, 3, -1)
    zd[:, 1] = zd[:, 0]
    starts = []
    orig_ss = ps_._scene_start
    def rec(cells, keep, u, lengths=None):
        idx = orig_ss(cells, keep, u, lengths)
        starts.append((cells, keep, idx))
        return idx
    ps_._scene_start = rec
    sd = torch.randint(0, 2 ** 62, (B,))
    r1 = ps_.crn_reward(batch, zd, sd)
    ps_._scene_start = orig_ss
    # the start as an index into the INPUT cloud, per mask
    def input_idx(cells, keep, idx):
        kk = keep.reshape(-1, keep.shape[-1])
        return torch.stack([torch.nonzero(kk[i])[idx[i], 0] for i in range(kk.shape[0])])
    phys = [input_idx(*t) for t in starts]
    check("scene CRN: same mask twice -> reward difference exactly 0; the token-FPS start is the same "
          "physical point in every mask", float((r1[:, 0] - r1[:, 1]).abs().max()) == 0.0
          and all(torch.equal(phys[0], p) for p in phys[1:]) and not torch.equal(r1[:, 0], r1[:, 2]))
    surr, stats = ps_.remover_loss(batch)
    surr.backward()
    grads = {n: p.grad for n, p in ps_.named_parameters() if p.grad is not None}
    check("density only + scene: remover_loss gradient on mu only (incl. the scene), no gate effect",
          sorted(grads) == ["link_remover.mu"] and float(grads["link_remover.mu"][rem.scene_region].abs()) > 0
          and not any(k.startswith("remover_effect_gate/") for k in stats)
          and "remover_reward_filled_frac" in stats)
    mech = ps_.mechanism_check(batch, masks=("none", "deterministic"))
    log, moved = controller_step(ps_, make_batch(z, N, B=B, seed=4))
    check("density only + scene: a real controller phi step runs and moves mu", moved and "remover_grad_mu/scene" in log
          and not any(k.startswith("remover_grad_gate/") for k in log))
    check("density only + scene: mechanism_check runs; predict_action runs",
          any(k.startswith("mech/deterministic/coverage/nut") for k in mech)
          and ps_.predict_action({k: v for k, v in batch["obs"].items() if k != "point_label"})["action"].shape[0] == B)

    print("\n=== policy: decoupled theta density (theta_rho, DObal) ===")
    del ps_
    torch.cuda.empty_cache()
    pr = build("r3d.policy.dp3_link_remover.DP3LinkRemover",
               {"enabled": True, "k_masks": 4, "learn_gates": False, "init_rho": 0.9,
                "region_groups": "default+scene", "theta_rho": [0.1, 1.0]})
    pr.train()
    pr.remover_mode = "sample"
    seen = []
    orig = pr._thin_obs
    pr._thin_obs = lambda obs, cells, rho, **kw: (seen.append(rho.detach().clone()), orig(obs, cells, rho, **kw))[1]
    for i in range(6):
        pr.compute_loss(make_batch(z, N, B=B, seed=40 + i))
    pr.remover_mode = "deterministic"
    pr.compute_loss(make_batch(z, N, B=B, seed=46))
    pr._thin_obs = orig
    th = torch.cat(seen[:-1])
    phi_rho = pr.link_remover.rho(pr.link_remover.sample(4000))
    sr = pr.link_remover.scene_region
    check("theta_rho [0.1, 1]: theta-pass rho ~ U[0.1, 1] per region (mean ~0.55), phi's own draws stay at ~0.9; "
          "fine-tune uses the deterministic rho",
          float(th.min()) >= 0.1 and float(th.max()) <= 1.0 and abs(float(th.mean()) - 0.55) < 0.05
          and float(th.std()) > 0.2 and abs(float(phi_rho.mean()) - 0.85) < 0.1
          and torch.equal(seen[-1], pr.link_remover.rho()[None].expand(B, -1)),
          f"theta mean {float(th.mean()):.3f} sd {float(th.std()):.3f}, phi mean {float(phi_rho.mean()):.3f}, "
          f"scene min {float(th[:, sr].min()):.3f}")
    try:
        build("r3d.policy.dp3_link_remover.DP3LinkRemover", {"enabled": True, "theta_rho": [0.1, 1.0]})
        check("theta_rho with learned gates raises", False)
    except Exception as e:                       # hydra wraps it in an InstantiationException
        check("theta_rho with learned gates raises", "theta_rho" in str(e), type(e).__name__)
    del pr
    torch.cuda.empty_cache()

    print("\n=== policy: all_links (gates for link0..link4) ===")
    pa = build("r3d.policy.dp3_link_remover.DP3LinkRemover",
               {"enabled": True, "k_masks": 4, "learn_density": False, "init_keep": 0.5, "theta_keep": 0.5,
                "region_groups": "all_links"})
    pa.train()
    torch.manual_seed(12)
    la, _ = pa.compute_loss(make_batch(z, N, B=B, seed=50))
    surr, stats = pa.remover_loss(make_batch(z, N, B=B, seed=51))
    surr.backward()
    g = pa.link_remover.gate.grad
    check("all_links policy: theta pass + reward pass run; 13 gate logits get a gradient",
          bool(torch.isfinite(la)) and g is not None and g.shape[0] == 13 and "remover_n_closed/link4" in stats)
    del pa
    torch.cuda.empty_cache()

    print("\n=== policy: structured mix ===")
    torch.cuda.empty_cache()
    pm = build("r3d.policy.dp3_link_remover.DP3LinkRemover",
               {"enabled": True, "k_masks": 4, "learn_density": False, "init_keep": 0.8, "mix_eps": 0.25,
                "mix_masks": [[], ["link5", "link6", "link7"],
                              ["link5", "link6", "link7", "hand_f1side", "hand_centre", "hand_f2side"], "all"]})
    pm.train()
    torch.manual_seed(11)
    surr, stats = pm.remover_loss(make_batch(z, N, B=16, seed=2))
    surr.backward()
    grads = {n: p.grad for n, p in pm.named_parameters() if p.grad is not None}
    names = pm.link_remover.mix_names
    log, moved = controller_step(pm, make_batch(z, N, B=16, seed=5))
    check("mix: a real controller phi step runs, moves the gates, logs the per-mask diagnostic",
          moved and any(k.startswith("remover_mask_n/") for k in log))
    check("mix: per-mask diagnostic logged; gradient on the gate logits only",
          all(f"remover_mask_n/{n}" in stats for n in names) and any(k.startswith("remover_mask_effect/") for k in stats)
          and sorted(grads) == ["link_remover.gate"], f"masks {names}, counts {[stats[f'remover_mask_n/{n}'] for n in names]}")


if __name__ == "__main__":
    main()
