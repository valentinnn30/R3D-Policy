"""Learned link remover v0 (C2) -- docs/learned_remover_v0.md.

Thins the robot's points PER REGION before the encoder's token FPS; the token budget,
kNN and FPS itself are unchanged. Per region l (a group of the stored link cells,
`robosuite_pcd.CELL_NAMES`) one draw per training window of TWO random variables:

    g_l ~ Bernoulli(pi_l),  pi_l = sigmoid(a_l)         REMOVAL GATE: keep the region at all?
    z_l ~ N(mu_l, sigma^2), rho_l = sigmoid(z_l)        DENSITY of a kept region

and region l keeps g_l * round(rho_l * n_l) of its n_l points, chosen by FPS WITHIN the
region. Why two: FPS places the encoder's tokens by surface EXTENT, and FPS thinning keeps a
region's extent -- measured on Square D2 (N = 4096, 64 tokens), thinning link5-7 to 2% (47
points) still leaves them 28.3 of their 34.6 tokens (patches 2 -> 14 cm) and nut coverage
0.55, against 0.81 only at complete removal. So density is a PATCH-SIZE lever and only
removal moves tokens; the gate explores removal directly instead of needing a draw at
rho ~ 0 (z ~ -7) to find it. Points outside every region (object, scene, cells in no
group) are always kept. Both observation frames of a window share the draw.

phi = (a, mu) gets no backprop gradient (selection is discrete); REINFORCE with a
leave-one-out baseline (`rloo_surrogate`) on

    log P(g, z) = sum_l [ log Bern(g_l; pi_l) + g_l * log N(z_l; mu_l, sigma^2) ]

(z_l is irrelevant to the reward when the gate is closed, so it is left out -- a valid
density of the mixed variable, with less variance). Bounds after every step: mu in
[mu_min, mu_max] (default [-9, 6]: -9 keeps 0 points of any region below ~4050 points), a in
[-gate_max, gate_max] (default 5: pi in [0.007, 0.993], so a closed region is still reopened
now and then and can recover). Bound the logits, never rho / pi.

Inference is deterministic: gate = 1[pi_l >= 0.5], density = sigmoid(mu_l), snapped to 0
below `snap_rho_below`.

The thinned cloud is compacted (kept points first, in their original order) and padded
with copies of the first kept point; the valid count is returned as `point_cloud_len`,
the robot-cut padding path that masks padding out of the encoder's FPS and kNN (same
convention as `r3d/common/label_cut.py`).
"""

import math

import torch
import torch.nn as nn

NOT_THINNED = -1
FPS_MIN_WIDTH = 8


def region_table(cell_names, region_groups):
    """region_groups: [{'name': str, 'cells': [cell name, ...]}, ...] -> (cell_to_region
    LongTensor (n_cells,), region names). A cell in no group is never thinned (-1)."""
    cell_names = list(cell_names)
    table = torch.full((len(cell_names),), NOT_THINNED, dtype=torch.long)
    names = []
    for r, g in enumerate(region_groups):
        names.append(str(g["name"]))
        for c in g["cells"]:
            if c not in cell_names:
                raise KeyError(f"region '{g['name']}': unknown link cell '{c}' (known: {cell_names})")
            i = cell_names.index(c)
            if table[i] != NOT_THINNED:
                raise ValueError(f"link cell '{c}' is in two regions")
            table[i] = r
    if len(set(names)) != len(names):
        raise ValueError(f"duplicate region names {names}")
    return table, names


def region_of(cells, cell_to_region):
    """per-point region id (C, N), NOT_THINNED (-1) for points in no region."""
    c2r = cell_to_region.to(cells.device)
    return torch.where(cells >= 0, c2r[cells.clamp(min=0)], torch.full_like(cells, NOT_THINNED))


def _fps_from(xyz, lengths, K, u):
    """pytorch3d FPS with an explicit start: the floor(u * length)-th point of each cloud.
    xyz (R, P, 3) with P >= FPS_MIN_WIDTH, lengths / K (R,) long with 1 <= K <= length,
    u (R,) in [0, 1). Same kernel as `sample_farthest_points`, whose random start is a
    per-cloud .item() loop and cannot be shared between masks. -> idx (R, max K), -1 padded."""
    from pytorch3d import _C
    start = torch.minimum((u.to(xyz.device, torch.float64) * lengths).long(), lengths - 1).clamp(min=0)
    return _C.sample_farthest_points(xyz.float().contiguous(), lengths, K, start)


def thin_regions(points, cells, rho, cell_to_region, lengths=None, min_keep=None,
                 fill_to_min_keep=False, random_start=False, u_region=None, u_fill=None,
                 return_keep=False):
    """Keep round(rho[c, l] * n[c, l]) points of region l in cloud c, by FPS within the region.

    points (C, N, D) (xyz first), cells (C, N) long link-cell id or -1, rho (C, L) in [0, 1],
    cell_to_region (n_cells,) long, lengths (C,) valid points (rest is padding) or None.
    min_keep: the encoder's token count M; N_kept >= M must hold. A cloud below it raises,
    or with fill_to_min_keep gets removed region points back, chosen by FPS over the removed
    points, until it holds M (a deterministic function of the draw, so log P(z) and
    REINFORCE are unchanged; `filled` reports how often it fires).

    FPS start points: u_region (C, L) / u_fill (C,) uniforms in [0, 1) pick the start as the
    floor(u * n)-th member. Given by the caller, they are SHARED between masks of one window
    (common random numbers): a region's members do not depend on the mask, so every mask
    starts at the same physical point, and FPS being greedy, a mask keeping k1 < k2 points
    of a region keeps exactly the first k1 of the k2 (nested masks). Not given: drawn here
    (random_start) or 0 (deterministic).
    -> points (C, N, D) compacted + padded, lengths (C,) long, n (C, L) points per region
       before thinning, k (C, L) kept per region by the draw, filled (C,) points added back
       [, keep (C, N) bool over the INPUT points if return_keep].
    """
    C, N, _ = points.shape
    L = rho.shape[1]
    dev = points.device
    if cells.shape != (C, N):
        raise ValueError(f"cells {tuple(cells.shape)} do not match points {tuple(points.shape)}")
    if rho.shape[0] != C:
        raise ValueError(f"rho has {rho.shape[0]} rows for {C} clouds")
    if not torch.isfinite(rho).all() or (rho < 0).any() or (rho > 1).any():
        raise ValueError("rho must lie in [0, 1]")
    n_cells = cell_to_region.shape[0]
    if (cells >= n_cells).any() or (cells < NOT_THINNED).any():
        raise ValueError(f"link cell ids outside [-1, {n_cells - 1}]")
    if int(cell_to_region.max()) >= L:
        raise ValueError(f"cell_to_region names region {int(cell_to_region.max())} but rho has {L}")
    for name, u, shape in (("u_region", u_region, (C, L)), ("u_fill", u_fill, (C,))):
        if u is not None and (tuple(u.shape) != shape or bool((u < 0).any()) or bool((u >= 1).any())):
            raise ValueError(f"{name} must be {shape} uniforms in [0, 1)")

    region = region_of(cells, cell_to_region)
    valid = torch.ones(C, N, dtype=torch.bool, device=dev) if lengths is None else \
        torch.arange(N, device=dev)[None] < lengths.to(dev)[:, None]
    in_region = valid & (region >= 0)

    # points per region and kept count
    onehot_idx = torch.where(in_region, region, torch.full_like(region, L))      # L = bin of nothing
    n = torch.zeros(C, L + 1, dtype=torch.long, device=dev).scatter_add_(
        1, onehot_idx, torch.ones_like(onehot_idx))[:, :L]
    k = torch.floor(rho.to(torch.float64) * n + 0.5).long().clamp(max=n)        # round half up

    keep = valid & ~in_region
    if n.max() > 0:
        # gather every region of every cloud into its own padded cloud: (C*L, n_max), members
        # in their original order. >= 8: pytorch3d 0.7.8's CUDA FPS makes an illegal memory
        # access on a tensor with fewer than 8 points (seen once the caching allocator is
        # warm); `lengths` masks the rest
        n_max = max(int(n.max()), FPS_MIN_WIDTH)
        order = torch.argsort(onehot_idx * (N + 1) + torch.arange(N, device=dev)[None], dim=1)   # stable by region
        sorted_region = torch.gather(onehot_idx, 1, order)
        start = torch.cumsum(n, 1) - n                                               # first slot of region l
        slot = torch.arange(N, device=dev)[None] - torch.gather(
            torch.cat([start, torch.zeros(C, 1, dtype=torch.long, device=dev)], 1), 1, sorted_region)
        member = torch.full((C, L + 1, n_max), -1, dtype=torch.long, device=dev)
        ok = sorted_region < L
        cc = torch.arange(C, device=dev)[:, None].expand(C, N)
        member[cc[ok], sorted_region[ok], slot[ok]] = order[ok]
        member = member[:, :L].reshape(C * L, n_max)
        xyz = torch.gather(points[..., :3], 1, member.clamp(min=0).view(C, L * n_max, 1).expand(-1, -1, 3))
        xyz = xyz.view(C * L, n_max, 3)
        kk = k.reshape(-1)
        u = u_region.reshape(-1) if u_region is not None else (
            torch.rand(C * L, device=dev) if random_start else torch.zeros(C * L, device=dev))
        # pytorch3d needs K >= 1 and length >= 1 per row; those rows are masked by slot < K
        sel = _fps_from(xyz, n.reshape(-1).clamp(min=1), kk.clamp(min=1), u)
        sel_ok = torch.arange(sel.shape[1], device=dev)[None] < kk[:, None]
        chosen = torch.gather(member, 1, sel.clamp(min=0))                           # point index in cloud
        rows = torch.arange(C, device=dev).repeat_interleave(L)[:, None].expand_as(chosen)
        keep[rows[sel_ok], chosen[sel_ok]] = True

    filled = torch.zeros(C, dtype=torch.long, device=dev)
    deficit = (min_keep - keep.sum(1)).clamp(min=0) if min_keep is not None else filled
    if (deficit > 0).any():
        if not fill_to_min_keep:
            raise RuntimeError(f"link remover kept {int(keep.sum(1).min())} points, fewer than the "
                               f"{min_keep} tokens the encoder samples (N_kept >= M violated)")
        if (valid.sum(1) < min_keep).any():
            raise RuntimeError(f"a cloud has fewer than min_keep={min_keep} valid points before thinning")
        rd = torch.nonzero(deficit > 0).squeeze(1)
        removed = in_region[rd] & ~keep[rd]                     # deficit <= removed count by construction
        n_rem = removed.sum(1)
        w = max(int(n_rem.max()), FPS_MIN_WIDTH)
        idx = torch.argsort((~removed).long() * (N + 1) + torch.arange(N, device=dev)[None], dim=1)[:, :w]
        xyz = torch.gather(points[rd][..., :3], 1, idx[..., None].expand(-1, -1, 3))
        u = u_fill[rd] if u_fill is not None else (
            torch.rand(len(rd), device=dev) if random_start else torch.zeros(len(rd), device=dev))
        sel = _fps_from(xyz, n_rem, deficit[rd], u)
        sel_ok = torch.arange(sel.shape[1], device=dev)[None] < deficit[rd][:, None]
        chosen = torch.gather(idx, 1, sel.clamp(min=0))
        keep[rd[:, None].expand_as(chosen)[sel_ok], chosen[sel_ok]] = True
        filled[rd] = deficit[rd]

    out_len = keep.sum(1)
    if (out_len == 0).any():
        raise RuntimeError("link remover kept no point of a cloud")
    # kept points first, in original order; pad with copies of the first kept point
    order = torch.argsort((~keep).long() * (N + 1) + torch.arange(N, device=dev)[None], dim=1)
    out = torch.gather(points, 1, order[..., None].expand_as(points))
    pad = torch.arange(N, device=dev)[None] >= out_len[:, None]
    out = torch.where(pad[..., None], out[:, :1].expand_as(out), out)
    if return_keep:
        return out, out_len, n, k, filled, keep
    return out, out_len, n, k, filled


def thin_window(points, cells, rho, cell_to_region, lengths=None, u_region=None, u_fill=None,
                return_keep=False, **kw):
    """points (B, To, N, D), cells (B, To, N), rho (B, L): ONE draw per window, applied to
    each of its To frames (each frame's FPS runs on its own points). u_region (B, To, L) /
    u_fill (B, To): shared FPS-start uniforms (see thin_regions).
    -> points (B, To, N, D), lengths (B, To), n (B, To, L), k (B, To, L), filled (B, To)
       [, keep (B, To, N)]."""
    B, To = points.shape[:2]
    flat = lambda x: None if x is None else x.reshape(B * To, *x.shape[2:])
    res = thin_regions(flat(points), flat(cells), rho.repeat_interleave(To, 0), cell_to_region,
                       flat(lengths), u_region=flat(u_region), u_fill=flat(u_fill),
                       return_keep=return_keep, **kw)
    out, ln, n, k, filled = res[:5]
    res_w = (out.view(points.shape), ln.view(B, To), n.view(B, To, -1), k.view(B, To, -1), filled.view(B, To))
    return res_w + ((res[5].view(B, To, -1),) if return_keep else ())


class LinkRemover(nn.Module):
    """phi = (a: removal-gate logit, mu: density logit) per region, and its sampling.

    A draw is ONE tensor (..., 2L): [z (L) | g (L)], g in {0., 1.} -- so a batch of draws
    moves through sample / log_prob / rho without a second container."""

    def __init__(self, cell_names, region_groups, init_rho=0.8, init_keep=0.8, sigma=1.0,
                 mu_min=-9.0, mu_max=6.0, gate_max=5.0, snap_rho_below=None):
        super().__init__()
        table, self.region_names = region_table(cell_names, region_groups)
        self.cell_names = list(cell_names)
        self.register_buffer("cell_to_region", table)
        L = len(self.region_names)
        if L == 0:
            raise ValueError("link remover with no regions")
        for name, v in (("init_rho", init_rho), ("init_keep", init_keep)):
            if not 0.0 < v < 1.0:
                raise ValueError(f"{name} must be in (0, 1)")
        self.mu_min, self.mu_max, self.gate_max = float(mu_min), float(mu_max), float(gate_max)
        # post-training snap: deterministic rho below this -> 0 (complete removal). Inference
        # only -- training draws are never snapped, so no region can get stuck at rho = 0;
        # sparse remnants of a mostly-removed region would otherwise be FPS magnets.
        if snap_rho_below is not None and not 0.0 < float(snap_rho_below) < 1.0:
            raise ValueError("snap_rho_below must be in (0, 1) or None")
        self.snap_rho_below = None if snap_rho_below is None else float(snap_rho_below)
        logit = lambda p: math.log(p / (1 - p))
        self.mu = nn.Parameter(torch.full((L,), logit(init_rho)))
        self.gate = nn.Parameter(torch.full((L,), logit(init_keep)))
        self.register_buffer("sigma", torch.tensor(float(sigma)))
        self.clamp_()

    @property
    def n_regions(self):
        return self.mu.shape[0]

    @torch.no_grad()
    def clamp_(self):
        self.mu.clamp_(self.mu_min, self.mu_max)
        self.gate.clamp_(-self.gate_max, self.gate_max)

    @torch.no_grad()
    def set_sigma(self, sigma):
        self.sigma.fill_(float(sigma))

    def keep_prob(self):
        return torch.sigmoid(self.gate.detach())

    def sample(self, n):
        """-> draws (n, 2L) = [z | g], detached."""
        with torch.no_grad():
            L, dev = self.n_regions, self.mu.device
            z = self.mu[None] + self.sigma * torch.randn(n, L, device=dev)
            g = (torch.rand(n, L, device=dev) < torch.sigmoid(self.gate)[None]).to(z.dtype)
            return torch.cat([z, g], -1)

    def split(self, draw):
        L = self.n_regions
        if draw.shape[-1] != 2 * L:
            raise ValueError(f"a draw is (..., 2L = {2 * L}), got {tuple(draw.shape)}")
        return draw[..., :L], draw[..., L:]

    def log_prob(self, draw):
        """sum_l [log Bern(g_l; sigmoid(a_l)) + g_l log N(z_l; mu_l, sigma^2)] -> (...,);
        the gradient reaches a and mu only."""
        z, g = self.split(draw.detach())
        s = self.sigma
        lp_z = -0.5 * ((z - self.mu) / s) ** 2 - torch.log(s) - 0.5 * math.log(2 * math.pi)
        lp_g = g * torch.nn.functional.logsigmoid(self.gate) + (1 - g) * torch.nn.functional.logsigmoid(-self.gate)
        return (lp_g + g * lp_z).sum(-1)

    def rho(self, draw=None):
        """effective keep rates g * sigmoid(z) for a training draw (never snapped); without a
        draw the deterministic inference rates 1[pi >= 0.5] * sigmoid(mu), the density snapped
        to 0 below `snap_rho_below`."""
        if draw is not None:
            z, g = self.split(draw)
            return g * torch.sigmoid(z)
        rho = torch.sigmoid(self.mu.detach())
        if self.snap_rho_below is not None:
            rho = torch.where(rho < self.snap_rho_below, torch.zeros_like(rho), rho)
        return torch.where(self.keep_prob() >= 0.5, rho, torch.zeros_like(rho))


def rloo_surrogate(log_prob, loss):
    """REINFORCE with a leave-one-out baseline over K draws per window.

    log_prob (B, K) with grad, loss (B, K) the per-draw reward (lower is better).
    adv_ik = loss_ik - mean_{j != k} loss_ij (detached). Minimising mean(adv * log_prob)
    lowers the probability of draws that did worse than their siblings.
    -> (surrogate scalar, adv (B, K))."""
    if log_prob.shape != loss.shape or log_prob.dim() != 2:
        raise ValueError(f"log_prob {tuple(log_prob.shape)} and loss {tuple(loss.shape)} must both be (B, K)")
    K = loss.shape[1]
    if K < 2:
        raise ValueError("the leave-one-out baseline needs K >= 2 draws per window")
    loss = loss.detach()
    adv = loss - (loss.sum(1, keepdim=True) - loss) / (K - 1)
    return (adv * log_prob).mean(), adv
