"""Learned link remover v0 (C2) -- docs/learned_remover_v0.md.

Thins the robot's points PER REGION before the encoder's token FPS; the token budget,
kNN and FPS itself are unchanged. One logit mu_l per region (a group of the stored link
cells, `robosuite_pcd.CELL_NAMES`); per training window one draw

    z_l ~ N(mu_l, sigma^2),   rho_l = sigmoid(z_l)     (keep rate in (0, 1))

and region l keeps round(rho_l * n_l) of its n_l points, chosen by FPS WITHIN the region
(the kept part looks like that region in a sparser cloud). Points outside every region
(object, scene, cells excluded from all groups) are always kept. Both observation frames of
a window share the draw. Inference is deterministic: rho_l = sigmoid(mu_l).

mu gets no backprop gradient (selection is discrete); it is trained by REINFORCE with a
leave-one-out baseline (`rloo_surrogate`) on log P(z) = sum_l log N(z_l; mu_l, sigma^2).
mu is clamped to [mu_min, mu_max] after every step -- bound mu, never rho: a rho floor
would forbid complete removal, an unbounded mu could drift to -inf and never come back.

The thinned cloud is compacted (kept points first, in their original order) and padded
with copies of the first kept point; the valid count is returned as `point_cloud_len`,
the robot-cut padding path that masks padding out of the encoder's FPS and kNN (same
convention as `r3d/common/label_cut.py`).
"""

import math

import torch
import torch.nn as nn
from pytorch3d.ops import sample_farthest_points

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


def thin_regions(points, cells, rho, cell_to_region, lengths=None, min_keep=None,
                 fill_to_min_keep=False, random_start=False):
    """Keep round(rho[c, l] * n[c, l]) points of region l in cloud c, by FPS within the region.

    points (C, N, D) (xyz first), cells (C, N) long link-cell id or -1, rho (C, L) in [0, 1],
    cell_to_region (n_cells,) long, lengths (C,) valid points (rest is padding) or None.
    min_keep: the encoder's token count M; N_kept >= M must hold. A cloud below it raises,
    or with fill_to_min_keep gets removed region points back, chosen by FPS over the removed
    points, until it holds M (a deterministic function of the draw, so log P(z) and
    REINFORCE are unchanged; `filled` reports how often it fires).
    -> points (C, N, D) compacted + padded, lengths (C,) long, n (C, L) points per region
       before thinning, k (C, L) kept per region by the draw, filled (C,) points added back.
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

    cell_to_region = cell_to_region.to(dev)
    region = torch.where(cells >= 0, cell_to_region[cells.clamp(min=0)], torch.full_like(cells, NOT_THINNED))
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
        # gather every region of every cloud into its own padded cloud: (C*L, n_max)
        # >= 8: pytorch3d 0.7.8's CUDA FPS makes an illegal memory access on a tensor with
        # fewer than 8 points (seen once the caching allocator is warm); `lengths` masks the rest
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
        _, sel = sample_farthest_points(xyz, lengths=n.reshape(-1), K=kk.clamp(min=1),
                                        random_start_point=random_start)
        # pytorch3d returns index 0 even for K = 0 or an empty cloud: mask by slot < K
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
        xyz = torch.gather(points[rd][..., :3], 1, idx[..., None].expand(-1, -1, 3)).contiguous()
        _, sel = sample_farthest_points(xyz, lengths=n_rem, K=deficit[rd], random_start_point=random_start)
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
    return out, out_len, n, k, filled


def thin_window(points, cells, rho, cell_to_region, lengths=None, **kw):
    """points (B, To, N, D), cells (B, To, N), rho (B, L): ONE draw per window, applied to
    each of its To frames (each frame's FPS runs on its own points).
    -> points (B, To, N, D), lengths (B, To), n (B, To, L), k (B, To, L), filled (B, To)."""
    B, To = points.shape[:2]
    flat = lambda x: x.reshape(B * To, *x.shape[2:])
    out, ln, n, k, filled = thin_regions(flat(points), flat(cells), rho.repeat_interleave(To, 0),
                                         cell_to_region, None if lengths is None else flat(lengths), **kw)
    return out.view(points.shape), ln.view(B, To), n.view(B, To, -1), k.view(B, To, -1), filled.view(B, To)


class LinkRemover(nn.Module):
    """The remover's parameters phi = mu (one logit per region) and its sampling."""

    def __init__(self, cell_names, region_groups, init_rho=0.8, sigma=1.0, mu_min=-8.0, mu_max=6.0,
                 snap_rho_below=None):
        super().__init__()
        table, self.region_names = region_table(cell_names, region_groups)
        self.cell_names = list(cell_names)
        self.register_buffer("cell_to_region", table)
        L = len(self.region_names)
        if L == 0:
            raise ValueError("link remover with no regions")
        if not 0.0 < init_rho < 1.0:
            raise ValueError("init_rho must be in (0, 1)")
        self.mu_min, self.mu_max = float(mu_min), float(mu_max)
        # post-training snap: deterministic rho below this -> 0 (complete removal). Inference
        # only -- training draws are never snapped, so no region can get stuck at rho = 0;
        # sparse remnants of a mostly-removed region would otherwise be FPS magnets.
        if snap_rho_below is not None and not 0.0 < float(snap_rho_below) < 1.0:
            raise ValueError("snap_rho_below must be in (0, 1) or None")
        self.snap_rho_below = None if snap_rho_below is None else float(snap_rho_below)
        self.mu = nn.Parameter(torch.full((L,), math.log(init_rho / (1 - init_rho))))
        self.register_buffer("sigma", torch.tensor(float(sigma)))
        self.clamp_()

    @property
    def n_regions(self):
        return self.mu.shape[0]

    @torch.no_grad()
    def clamp_(self):
        self.mu.clamp_(self.mu_min, self.mu_max)

    @torch.no_grad()
    def set_sigma(self, sigma):
        self.sigma.fill_(float(sigma))

    def sample(self, n):
        """-> z (n, L), detached."""
        with torch.no_grad():
            return self.mu[None] + self.sigma * torch.randn(n, self.n_regions, device=self.mu.device)

    def log_prob(self, z):
        """sum_l log N(z_l; mu_l, sigma^2) -> (...,); gradient reaches mu only."""
        s = self.sigma
        return (-0.5 * ((z.detach() - self.mu) / s) ** 2 - torch.log(s) - 0.5 * math.log(2 * math.pi)).sum(-1)

    def rho(self, z=None):
        """keep rates: sigmoid(z) for a training draw (never snapped); without a draw the
        deterministic inference rates sigmoid(mu), with rates below `snap_rho_below` set to 0."""
        if z is not None:
            return torch.sigmoid(z)
        rho = torch.sigmoid(self.mu.detach())
        if self.snap_rho_below is not None:
            rho = torch.where(rho < self.snap_rho_below, torch.zeros_like(rho), rho)
        return rho


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
