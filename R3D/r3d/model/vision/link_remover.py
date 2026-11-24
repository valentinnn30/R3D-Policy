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

Variants (all default off; tuning round 2, docs/sim_token_allocation_learnings.md):
  learn_gates=False      density only: g = 1 always, phi = mu.
  scene region           a group with the pseudo cell NON_ROBOT ("<non_robot>") is every
                         point outside the robot cells (cell -1: nut, peg, rest of the scene),
                         label-free. Density only (needs learn_gates=False); its rho is floored
                         at scene_rho_min in every draw and at inference and is never snapped,
                         so the objects can never be thinned away.
  mix_eps / mix_masks    with probability eps a draw is one of S fixed structured gate masks
                         (uniformly), otherwise the independent Bernoullis:
                         P(g) = (1 - eps) prod_l Bern(g_l; pi_l) + eps/S * 1[g in masks].
  fixed_removed          the SPECIALIST (round 3): a fixed mask, nothing learned. The listed
                         regions are removed, every other region is kept whole: gate logits at
                         -gate_max / +gate_max (so the deterministic mask is exactly that), no
                         density (needs learn_density=False), phi_parameters() empty, and every
                         draw IS the fixed mask.
  fixed_rho              the DENSITY SPECIALIST (2026-11-19): fixed per-region densities, nothing
                         learned -- {region: rho} for every region (density only: needs
                         learn_gates=False). mu holds logit(rho) for the record only; density(),
                         every draw and the deterministic rho return the table exactly (the snap
                         below snap_rho_below still applies, as for a learned remover).
                         phi_parameters() empty, no log P.
"""

import math

import torch
import torch.nn as nn

NOT_THINNED = -1
FPS_MIN_WIDTH = 8
NON_ROBOT = "<non_robot>"   # pseudo cell: a scene region of every point with cell -1


def region_table(cell_names, region_groups):
    """region_groups: [{'name': str, 'cells': [cell name, ...]}, ...] -> (cell_to_region
    LongTensor (n_cells,), region names). A cell in no group is never thinned (-1)."""
    cell_names = list(cell_names)
    table = torch.full((len(cell_names),), NOT_THINNED, dtype=torch.long)
    names = []
    for r, g in enumerate(region_groups):
        names.append(str(g["name"]))
        for c in g["cells"]:
            if c == NON_ROBOT:
                if list(g["cells"]) != [NON_ROBOT]:
                    raise ValueError(f"region '{g['name']}': {NON_ROBOT} must be the only cell of its group")
                continue                       # no table entry: region_of maps cell -1 to it
            if c not in cell_names:
                raise KeyError(f"region '{g['name']}': unknown link cell '{c}' (known: {cell_names})")
            i = cell_names.index(c)
            if table[i] != NOT_THINNED:
                raise ValueError(f"link cell '{c}' is in two regions")
            table[i] = r
    if len(set(names)) != len(names):
        raise ValueError(f"duplicate region names {names}")
    return table, names


def scene_region_of(region_groups):
    """index of the group made of NON_ROBOT, or None."""
    idx = [r for r, g in enumerate(region_groups) if NON_ROBOT in g["cells"]]
    if len(idx) > 1:
        raise ValueError(f"more than one {NON_ROBOT} region")
    return idx[0] if idx else None


def region_of(cells, cell_to_region, scene_region=None):
    """per-point region id (C, N), NOT_THINNED (-1) for points in no region; with a scene
    region, points with cell -1 belong to it."""
    c2r = cell_to_region.to(cells.device)
    rest = NOT_THINNED if scene_region is None else int(scene_region)
    return torch.where(cells >= 0, c2r[cells.clamp(min=0)], torch.full_like(cells, rest))


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
                 return_keep=False, scene_region=None):
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

    region = region_of(cells, cell_to_region, scene_region)
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
                 mu_min=-9.0, mu_max=6.0, gate_max=5.0, snap_rho_below=None, learn_density=True,
                 learn_gates=True, scene_rho_min=0.1, mix_eps=0.0, mix_masks=None, fixed_removed=None,
                 fixed_rho=None):
        super().__init__()
        self.learn_gates = bool(learn_gates)
        if not self.learn_gates and not learn_density:
            raise ValueError("learn_gates and learn_density are both off: nothing to learn")
        # learn_density False = gates only: a kept region keeps ALL its points (rho = g), mu is
        # not learned (no gradient, left at logit(init_rho) for checkpoint compatibility)
        self.learn_density = bool(learn_density)
        table, self.region_names = region_table(cell_names, region_groups)
        self.scene_region = scene_region_of(region_groups)
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
        self.mu = nn.Parameter(torch.full((L,), logit(init_rho)), requires_grad=self.learn_density)
        self.gate = nn.Parameter(torch.full((L,), logit(init_keep)), requires_grad=self.learn_gates)
        self.register_buffer("sigma", torch.tensor(float(sigma)))
        # scene region: density only, floored (plain tensors, not buffers: old state_dicts load)
        self.scene_rho_min = None
        if self.scene_region is not None:
            if self.learn_gates or not learn_density:
                raise ValueError("a scene region is density only: learn_gates=False, learn_density=True")
            if not 0.0 < float(scene_rho_min) < 1.0:
                raise ValueError("scene_rho_min must be in (0, 1)")
            self.scene_rho_min = float(scene_rho_min)
            self._mu_lo = torch.full((L,), self.mu_min)
            self._mu_lo[self.scene_region] = logit(self.scene_rho_min)
        # structured mix
        self.mix_eps = float(mix_eps)
        if not 0.0 <= self.mix_eps < 1.0:
            raise ValueError("mix_eps must be in [0, 1)")
        self.mix_names, self._mix = [], None
        if self.mix_eps > 0:
            if not self.learn_gates:
                raise ValueError("mix_eps > 0 needs learned gates")
            if not mix_masks:
                raise ValueError("mix_eps > 0 needs mix_masks")
            rows = []
            for m in mix_masks:
                removed = list(self.region_names) if m == "all" else list(m)
                bad = [r for r in removed if r not in self.region_names]
                if bad:
                    raise KeyError(f"mix mask: unknown regions {bad} (known: {self.region_names})")
                rows.append([0.0 if r in removed else 1.0 for r in self.region_names])
                self.mix_names.append("none_removed" if not removed else
                                      "all_removed" if len(set(removed)) == L else "+".join(removed))
            self._mix = torch.tensor(rows)
            if len({tuple(r) for r in rows}) != len(rows):
                raise ValueError("mix_masks must be distinct")
        # fixed mask (specialist): the gates hold the mask and are not learned
        self.fixed_removed = None
        if fixed_removed is not None:
            if learn_density or not self.learn_gates:
                raise ValueError("fixed_removed needs learn_density=False and learn_gates=True")
            if self.mix_eps > 0 or self.scene_region is not None:
                raise ValueError("fixed_removed excludes mix_eps and a scene region")
            removed = list(fixed_removed)
            bad = [r for r in removed if r not in self.region_names]
            if bad:
                raise KeyError(f"fixed_removed: unknown regions {bad} (known: {self.region_names})")
            self.fixed_removed = removed
            with torch.no_grad():
                self.gate.copy_(torch.tensor([-self.gate_max if r in removed else self.gate_max
                                              for r in self.region_names]))
            self.gate.requires_grad_(False)
        # fixed densities (density specialist): the table is the remover, nothing is learned
        self._fixed_rho = None
        if fixed_rho is not None:
            if self.learn_gates or not self.learn_density:
                raise ValueError("fixed_rho is density only: learn_gates=False, learn_density=True")
            if self.mix_eps > 0 or self.fixed_removed is not None:
                raise ValueError("fixed_rho excludes mix_eps and fixed_removed")
            if isinstance(fixed_rho, dict):
                bad = [r for r in fixed_rho if r not in self.region_names]
                missing = [r for r in self.region_names if r not in fixed_rho]
                if bad or missing:
                    raise KeyError(f"fixed_rho: unknown regions {bad}, missing regions {missing} "
                                   f"(known: {self.region_names})")
                vals = [float(fixed_rho[r]) for r in self.region_names]
            else:
                vals = [float(v) for v in fixed_rho]
                if len(vals) != L:
                    raise ValueError(f"fixed_rho has {len(vals)} values for {L} regions")
            if not all(0.0 <= v <= 1.0 for v in vals):
                raise ValueError(f"fixed_rho values must lie in [0, 1]: {vals}")
            if self.scene_region is not None and vals[self.scene_region] < self.scene_rho_min:
                raise ValueError(f"fixed_rho: scene {vals[self.scene_region]} below scene_rho_min {self.scene_rho_min}")
            self._fixed_rho = torch.tensor(vals, dtype=torch.float32)
            eps = 1e-6
            with torch.no_grad():
                self.mu.copy_(torch.tensor([logit(min(max(v, eps), 1 - eps)) for v in vals]))
            self.mu.requires_grad_(False)
        self.clamp_()

    @property
    def is_fixed(self):
        """a specialist: fixed gate mask (fixed_removed) or fixed densities (fixed_rho)."""
        return self.fixed_removed is not None or self._fixed_rho is not None

    @property
    def n_regions(self):
        return self.mu.shape[0]

    @torch.no_grad()
    def clamp_(self):
        if self.scene_region is None:
            self.mu.clamp_(self.mu_min, self.mu_max)
        else:
            self.mu.copy_(torch.clamp(torch.maximum(self.mu, self._mu_lo.to(self.mu.device)), max=self.mu_max))
        self.gate.clamp_(-self.gate_max, self.gate_max)

    @torch.no_grad()
    def set_sigma(self, sigma):
        self.sigma.fill_(float(sigma))

    def phi_parameters(self):
        if self.is_fixed:
            return []
        return [p for p, on in ((self.gate, self.learn_gates), (self.mu, self.learn_density)) if on]

    def fixed_gates(self, device=None):
        """(L,) gates of the fixed mask (1 = kept), or None."""
        if self.fixed_removed is None:
            return None
        return torch.tensor([0.0 if r in self.fixed_removed else 1.0 for r in self.region_names],
                            device=self.gate.device if device is None else device)

    def mix_table(self, device):
        """(S, L) gates of the structured masks (1 = kept), or None."""
        return None if self._mix is None else self._mix.to(device)

    def density(self):
        """deterministic density of a kept region, unsnapped: sigmoid(mu), or 1 gates-only
        (the fixed table for a density specialist)."""
        if self._fixed_rho is not None:
            return self._fixed_rho.to(self.mu.device).clone()
        return torch.sigmoid(self.mu.detach()) if self.learn_density else torch.ones_like(self.mu.detach())

    def keep_prob(self):
        if not self.learn_gates:
            return torch.ones_like(self.gate.detach())
        return torch.sigmoid(self.gate.detach())

    def sample(self, n):
        """-> draws (n, 2L) = [z | g], detached."""
        with torch.no_grad():
            L, dev = self.n_regions, self.mu.device
            if self._fixed_rho is not None:     # no noise: every draw IS the table (see rho)
                return torch.cat([self.mu.detach()[None].expand(n, -1), torch.ones(n, L, device=dev)], -1)
            z = self.mu[None] + self.sigma * torch.randn(n, L, device=dev) if self.learn_density \
                else torch.zeros(n, L, device=dev)
            if not self.learn_gates:
                return torch.cat([z, torch.ones_like(z)], -1)
            if self.fixed_removed is not None:
                return torch.cat([z, self.fixed_gates(dev)[None].expand(n, -1)], -1)
            g = (torch.rand(n, L, device=dev) < torch.sigmoid(self.gate)[None]).to(z.dtype)
            if self.mix_eps > 0:                # no extra RNG at all when the mix is off
                use = torch.rand(n, device=dev) < self.mix_eps
                pick = torch.randint(self._mix.shape[0], (n,), device=dev)
                g = torch.where(use[:, None], self.mix_table(dev)[pick].to(g.dtype), g)
            return torch.cat([z, g], -1)

    def split(self, draw):
        L = self.n_regions
        if draw.shape[-1] != 2 * L:
            raise ValueError(f"a draw is (..., 2L = {2 * L}), got {tuple(draw.shape)}")
        return draw[..., :L], draw[..., L:]

    def log_prob(self, draw):
        """sum_l [log Bern(g_l; sigmoid(a_l)) + g_l log N(z_l; mu_l, sigma^2)] -> (...,);
        the gradient reaches a and mu only. Gates only: the Bernoulli term alone."""
        if self.is_fixed:
            raise RuntimeError("fixed (specialist) remover: nothing to learn, no log P")
        z, g = self.split(draw.detach())
        if not self.learn_gates:
            if bool((g != 1).any()):
                raise ValueError("density-only remover got a draw with a closed gate")
            s = self.sigma
            return (-0.5 * ((z - self.mu) / s) ** 2 - torch.log(s) - 0.5 * math.log(2 * math.pi)).sum(-1)
        lp_g = g * torch.nn.functional.logsigmoid(self.gate) + (1 - g) * torch.nn.functional.logsigmoid(-self.gate)
        if self.mix_eps > 0:
            mix = self.mix_table(g.device)
            hit = (g[..., None, :] == mix).all(-1).any(-1)
            lp_bern = math.log(1 - self.mix_eps) + lp_g.sum(-1)
            lp_mix = torch.where(hit, torch.full_like(lp_bern, math.log(self.mix_eps / mix.shape[0])),
                                 torch.full_like(lp_bern, -math.inf))
            lp_gates = torch.logsumexp(torch.stack([lp_bern, lp_mix]), 0)
            if not self.learn_density:
                return lp_gates
            s = self.sigma
            lp_z = -0.5 * ((z - self.mu) / s) ** 2 - torch.log(s) - 0.5 * math.log(2 * math.pi)
            return lp_gates + (g * lp_z).sum(-1)
        if not self.learn_density:
            return lp_g.sum(-1)
        s = self.sigma
        lp_z = -0.5 * ((z - self.mu) / s) ** 2 - torch.log(s) - 0.5 * math.log(2 * math.pi)
        return (lp_g + g * lp_z).sum(-1)

    def rho(self, draw=None):
        """effective keep rates g * sigmoid(z) for a training draw (never snapped); without a
        draw the deterministic inference rates 1[pi >= 0.5] * sigmoid(mu), the density snapped
        to 0 below `snap_rho_below`."""
        if draw is not None:
            z, g = self.split(draw)
            if self._fixed_rho is not None:
                return self._fixed_rho.to(z.device).expand_as(z).clone()
            rho = g * torch.sigmoid(z) if self.learn_density else g
            if self.scene_region is not None:
                rho = rho.clone()
                rho[..., self.scene_region] = rho[..., self.scene_region].clamp(min=self.scene_rho_min)
            return rho
        rho = self.density()
        if self.snap_rho_below is not None and self.learn_density:
            snap = rho < self.snap_rho_below
            if self.scene_region is not None:
                snap[self.scene_region] = False
                rho = rho.clone()
                rho[self.scene_region] = rho[self.scene_region].clamp(min=self.scene_rho_min)
            rho = torch.where(snap, torch.zeros_like(rho), rho)
        elif self.scene_region is not None:
            rho = rho.clone()
            rho[self.scene_region] = rho[self.scene_region].clamp(min=self.scene_rho_min)
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
