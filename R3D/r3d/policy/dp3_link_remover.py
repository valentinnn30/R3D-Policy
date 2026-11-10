"""DP3 with the learned link remover v0 (C2) -- docs/learned_remover_v0.md.

`DP3LinkRemover` is DP3 plus `self.link_remover` (r3d/model/vision/link_remover.py). With
`link_remover=None` (or `enabled: false`) every method IS DP3's: training and evaluation
are bit-identical to the baseline (checked in scripts/sim/check_link_remover.py).

Observations carry `point_link` (B, T, N) -- the per-point link cell of the stored cloud
(robosuite_pcd.CELL_NAMES). It is a label, not a quantity: it is consumed here, before the
normalizer, and never reaches the encoder. Thinning runs on the RAW (metric) cloud, like
the stored FPS, so a thinned region looks like that region in a sparser cloud.

Three passes, three kinds of randomness:

- theta pass (`compute_loss`): one fresh draw z per window, within-region FPS from random
  starts, the encoder's own random token FPS, the standard DP3 diffusion loss. Unchanged
  from DP3 except for the cloud it sees.
- reward pass (`crn_reward`, no grad, eval mode): K masks per window compared under COMMON
  RANDOM NUMBERS. One seed per window, shared by its K masks and fresh per window and per
  phi step, draws everything that is random but not the mask: the token-FPS start (a
  NON-ROBOT point -- nut, peg, other scene points -- kept by every mask, so every mask
  starts at the same physical point; not link0-4, which are never thinned but almost never
  in view, their few points mostly silhouette mislabels), the token shuffle, the within-region FPS starts (masks are nested), the fill-up
  start, and the diffusion noise eps. t is fixed ({20, 50, 80}). Two masks of a window then
  differ ONLY by the mask; the same mask twice gives exactly the same reward. Random (not
  fixed) starts keep the comparison unbiased in expectation over seeds -- a fixed start
  would bias phi towards one token layout.
  Reward = x0-prediction error ||A0 - A_hat||^2 on the executed steps [To-1, To-1+Ta) of
  the normalized action, averaged over t; lower is better.
- inference (`predict_action`): deterministic rho = sigmoid(mu) (snapped below
  `snap_rho_below`), within-region and token FPS with their normal independent random
  starts.

`remover_mode` (set by the training controller, r3d/common/link_remover_training.py):
"sample" -- the theta pass draws z per window (warm-up and phi phase); "deterministic" --
the theta pass uses the inference rho, snapped (the final phi-frozen fine-tune, so theta
trains on exactly the clouds evaluation will show it).

`link_remover.fixed_removed` (round 3 SPECIALIST, default null): a fixed mask, nothing
learned -- the listed regions removed, the rest kept whole. The theta pass always uses the
deterministic mask (remover_mode is ignored), so training and evaluation see the same clouds.

`link_remover.theta_rho: [lo, hi]` (DObal, round 4, default null): the density counterpart of
`theta_keep` for a density-only remover. The theta pass draws every region's density from
U[lo, hi] per window, independent of phi (the scene region still floored at scene_rho_min);
phi keeps learning from its own reward draws. Fine-tune (deterministic mode) is unchanged.
"""

import torch
from omegaconf import DictConfig, ListConfig, OmegaConf

from r3d.common import robosuite_pcd as RP
from r3d.common.pytorch_util import dict_apply
from r3d.model.vision.link_remover import NON_ROBOT, LinkRemover, rloo_surrogate, thin_window
from r3d.policy.dp3 import DP3

LINK_KEY = "point_link"
LABEL_KEY = "point_label"     # analytics only (nut / peg), never a model input


class DP3LinkRemover(DP3):

    def __init__(self, *args, link_remover=None, **kwargs):
        # popped before DP3: DP3 forwards unknown kwargs to scheduler.step
        super().__init__(*args, **kwargs)
        cfg = link_remover
        if isinstance(cfg, (DictConfig, ListConfig)):
            cfg = OmegaConf.to_container(cfg, resolve=True)
        if cfg is None or not cfg.get("enabled", True):
            self.link_remover = None
            return
        groups = cfg.get("region_groups", "default")
        if groups == "default":
            groups = RP.default_region_groups()
        elif groups == "all_links":                     # + learned gates for link0..link4
            groups = RP.all_links_region_groups()
        elif groups == "default+scene":                 # + a label-free scene region (density only)
            groups = RP.default_region_groups() + [{"name": "scene", "cells": [NON_ROBOT]}]
        self.link_remover = LinkRemover(
            RP.CELL_NAMES, groups, init_rho=cfg.get("init_rho", 0.8), init_keep=cfg.get("init_keep", 0.8),
            sigma=cfg.get("sigma", 1.0), mu_min=cfg.get("mu_min", -9.0), mu_max=cfg.get("mu_max", 6.0),
            gate_max=cfg.get("gate_max", 5.0), snap_rho_below=cfg.get("snap_rho_below", None),
            learn_density=cfg.get("learn_density", True), learn_gates=cfg.get("learn_gates", True),
            scene_rho_min=cfg.get("scene_rho_min", 0.1), mix_eps=cfg.get("mix_eps", 0.0),
            mix_masks=cfg.get("mix_masks", None), fixed_removed=cfg.get("fixed_removed", None))
        # theta pass gates drawn at this fixed keep probability instead of phi's (decoupled
        # exploration); the reward pass still samples from phi. None = phi's own draws.
        tk = cfg.get("theta_keep", None)
        if tk is not None and not (0.0 < float(tk) < 1.0 and self.link_remover.learn_gates):
            raise ValueError("theta_keep must be in (0, 1) and needs learned gates")
        self.theta_keep = None if tk is None else float(tk)
        tr = cfg.get("theta_rho", None)
        if tr is not None:
            lo, hi = (float(v) for v in tr)
            if not (0.0 <= lo < hi <= 1.0) or self.link_remover.learn_gates or not self.link_remover.learn_density:
                raise ValueError("theta_rho [lo, hi] needs 0 <= lo < hi <= 1 and a density-only remover")
            tr = (lo, hi)
        self.theta_rho = tr
        self.fixed_mask = self.link_remover.fixed_removed is not None
        if self.fixed_mask and self.theta_keep is not None:
            raise ValueError("fixed_removed: theta_keep has no meaning (the mask is fixed)")
        self.fill_to_min_keep = bool(cfg.get("fill_to_min_keep", True))
        self.reward_t = [int(t) for t in cfg.get("reward_t", [20, 50, 80])]
        self.k_masks = int(cfg.get("k_masks", 2))
        T_train = self.noise_scheduler.config.num_train_timesteps
        if not all(0 <= t < T_train for t in self.reward_t):
            raise ValueError(f"reward_t {self.reward_t} outside [0, {T_train})")
        if self.k_masks < 2:
            raise ValueError("the leave-one-out baseline needs k_masks >= 2")
        enc = self._uni3d()
        if enc is None or self.unfused or self.pc_encoder_extract_global_feature \
                or not self.obs_as_global_cond or self.use_target_ee:
            raise NotImplementedError("link remover: fused uni3d token encoder with global "
                                      "conditioning only")
        self.num_tokens = int(enc.num_group)
        self.remover_mode = "sample"

    # ---- helpers ----------------------------------------------------------------------
    def _split_link(self, obs):
        obs = {k: v for k, v in obs.items() if k != LABEL_KEY}
        if LINK_KEY not in obs:
            if self.link_remover is not None:
                raise KeyError(f"link remover on but obs has no '{LINK_KEY}' -- dataset load_point_link")
            return obs, None
        if self.link_remover is None:
            raise ValueError(f"obs carries '{LINK_KEY}' but the policy has no link remover")
        rest = {k: v for k, v in obs.items() if k != LINK_KEY}
        return rest, obs[LINK_KEY].long()

    def _thin_obs(self, obs, cells, rho, **kw):
        """Slice obs to n_obs_steps (DP3 reads no more), thin the first To clouds with one
        rho per window. -> obs with thinned point_cloud + point_cloud_len, stats."""
        To = self.n_obs_steps
        obs = {k: v[:, :To] for k, v in obs.items()}
        out = thin_window(obs["point_cloud"], cells[:, :To], rho, self.link_remover.cell_to_region,
                          lengths=obs.get("point_cloud_len"), min_keep=self.num_tokens,
                          fill_to_min_keep=self.fill_to_min_keep,
                          scene_region=self.link_remover.scene_region, **kw)
        obs["point_cloud"], obs["point_cloud_len"] = out[0], out[1]
        return obs, out

    # ---- theta pass -------------------------------------------------------------------
    def compute_loss(self, batch):
        obs, cells = self._split_link(batch["obs"])
        if self.link_remover is None:
            return super().compute_loss(batch)
        B = cells.shape[0]
        mode = "deterministic" if self.fixed_mask else self.remover_mode
        if mode == "sample":
            draw = self.link_remover.sample(B)
            if self.theta_keep is not None:
                z_, g_ = self.link_remover.split(draw)
                g_ = (torch.rand(g_.shape, device=g_.device) < self.theta_keep).to(g_.dtype)
                draw = torch.cat([z_, g_], -1)
            rho = self.link_remover.rho(draw)
            if self.theta_rho is not None:
                lo, hi = self.theta_rho
                rho = lo + (hi - lo) * torch.rand(rho.shape, device=rho.device)
                sr = self.link_remover.scene_region
                if sr is not None:
                    rho[:, sr] = rho[:, sr].clamp(min=self.link_remover.scene_rho_min)
        elif mode == "deterministic":
            rho = self.link_remover.rho()[None].expand(B, -1)
        else:
            raise ValueError(f"remover_mode {mode!r}")
        obs, (_, ln, n, k, filled) = self._thin_obs(obs, cells, rho, random_start=True)
        loss, loss_dict = super().compute_loss({**batch, "obs": obs})
        loss_dict.update(self._kept_stats(ln, n, k, filled))
        return loss, loss_dict

    def _kept_stats(self, ln, n, k, filled):
        d = {"remover_points_kept": ln.float().mean().item(),
             "remover_filled_frac": (filled > 0).float().mean().item()}
        nn_ = n.reshape(-1, n.shape[-1]).float().sum(0)
        kk = k.reshape(-1, k.shape[-1]).float().sum(0)
        for l, name in enumerate(self.link_remover.region_names):
            d[f"remover_kept_frac/{name}"] = (kk[l] / nn_[l]).item() if nn_[l] > 0 else float("nan")
        return d

    # ---- phi: common-random-numbers reward + RLOO surrogate ---------------------------
    def crn_draws(self, seeds, B, To, T, device):
        """Everything random in the reward pass except the mask, from ONE seed per window
        (shared by its K masks). -> dict of tensors on `device`."""
        L, G, Da = self.link_remover.n_regions, self.num_tokens, self.action_dim
        u_enc, perm, u_reg, u_fill, eps = [], [], [], [], []
        for s in seeds.tolist():
            g = torch.Generator().manual_seed(int(s))
            u_enc.append(torch.rand(To, generator=g))
            perm.append(torch.stack([torch.randperm(G, generator=g) for _ in range(To)]))
            u_reg.append(torch.rand(To, L, generator=g))
            u_fill.append(torch.rand(To, generator=g))
            eps.append(torch.randn(T, Da, generator=g))
        st = lambda xs: torch.stack(xs).to(device)
        return {"u_enc": st(u_enc), "perm": st(perm), "u_region": st(u_reg), "u_fill": st(u_fill),
                "eps": st(eps)}

    def _crn_start(self, cells, keep, draws, obs):
        """The shared token-FPS start for a reward-pass mask (see _encoder_start)."""
        sr = self.link_remover.scene_region
        if sr is None:
            return self._encoder_start(cells, keep, draws["u_enc"])
        ln = obs.get("point_cloud_len")
        return self._scene_start(cells, keep, draws["u_region"][..., sr], None if ln is None else ln[:, :cells.shape[1]])

    def _scene_start(self, cells, keep, u_scene, lengths=None):
        """With a scene region the non-robot points are thinned too, so "the u-th kept
        non-robot point" would differ between masks. Use the scene region's own shared FPS
        start instead: the floor(u * n)-th scene member over ALL members (as `_fps_from`
        picks it). FPS inside the scene starts there and the scene keeps >= 1 point (its rho
        floor), so it is kept by every mask: the identical physical point.
        cells / keep (B, To, N), u_scene (B, To) -> (B * To,) long."""
        N = cells.shape[-1]
        valid = torch.ones_like(keep) if lengths is None else \
            torch.arange(N, device=cells.device) < lengths[..., None].to(cells.device)
        members = (cells < 0) & valid
        n = members.sum(-1)
        if (n == 0).any():
            raise RuntimeError("a frame has no scene point to start the token FPS from")
        target = torch.minimum((u_scene.to(torch.float64) * n).long(), n - 1) + 1     # 1-based rank
        s_ = torch.argmax((torch.cumsum(members.long(), -1) >= target[..., None]).long(), -1)
        if not bool(keep.gather(-1, s_[..., None]).all()):
            raise RuntimeError("the scene's shared FPS start was not kept (scene rho floor violated?)")
        new = torch.cumsum(keep.long(), -1).gather(-1, s_[..., None])[..., 0] - 1
        return new.reshape(-1)

    def _encoder_start(self, cells, keep, u_enc):
        """Token-FPS start per thinned cloud: the floor(u * n)-th NON-ROBOT point of the
        frame (cell -1: nut, peg, scene; kept by every mask, so the identical physical point
        for every mask), mapped to its index in the compacted cloud.
        cells / keep (B, To, N), u_enc (B, To) -> (B * To,) long."""
        free = (cells < 0) & keep                            # padding is never kept
        n_free = free.sum(-1)
        if (n_free == 0).any():
            raise RuntimeError("a frame has no non-robot point to start the token FPS from")
        target = torch.minimum((u_enc * n_free).long(), n_free - 1) + 1  # 1-based rank
        s = torch.argmax((torch.cumsum(free.long(), -1) >= target[..., None]).long(), -1)
        new = torch.cumsum(keep.long(), -1).gather(-1, s[..., None])[..., 0] - 1
        return new.reshape(-1)

    @torch.no_grad()
    def crn_reward(self, batch, z, seeds):
        """z (B, K, 2L) mask draws (LinkRemover.sample: [density z | gate g]), seeds (B,) int
        -> reward (B, K): x0 error on the executed steps, averaged over reward_t. No grad;
        encoder and head in eval mode."""
        if self.noise_scheduler.config.prediction_type != "sample":
            raise NotImplementedError("crn_reward reads the head output as x0 (prediction_type: sample)")
        obs, cells = self._split_link(batch["obs"])
        B, K = z.shape[:2]
        To, T, dev = self.n_obs_steps, batch["action"].shape[1], z.device
        draws = self.crn_draws(seeds, B, To, T, dev)
        nactions = self.normalizer["action"].normalize(batch["action"])
        s0, s1 = To - 1, To - 1 + self.n_action_steps
        was_training = self.training
        self.eval()
        try:
            out = torch.zeros(B, K, device=dev)
            n_filled = 0
            for k in range(K):
                tobs, res = self._thin_obs(obs, cells, self.link_remover.rho(z[:, k]),
                                           u_region=draws["u_region"], u_fill=draws["u_fill"],
                                           return_keep=True)
                n_filled += int((res[4] > 0).sum())
                start = self._crn_start(cells[:, :To], res[5], draws, obs)
                global_cond, pc_pe = self._encode(tobs, start, draws["perm"].reshape(B * To, -1))
                err = 0.0
                for t in self.reward_t:
                    tt = torch.full((B,), t, device=dev, dtype=torch.long)
                    noisy = self.noise_scheduler.add_noise(nactions, draws["eps"], tt)
                    pred = self.model(sample=noisy, timestep=tt, local_cond=None, global_cond=global_cond,
                                      pc_pe=pc_pe, n_obs_steps=self.n_obs_steps)
                    err = err + ((pred - nactions)[:, s0:s1].float() ** 2).mean(dim=(1, 2))
                out[:, k] = err / len(self.reward_t)
            # fraction of the reward pass's clouds that needed fill-up (read by remover_loss)
            self.reward_filled_frac = n_filled / float(B * To * K)
        finally:
            self.train(was_training)
        return out

    def _encode(self, obs, fps_start, fps_perm):
        """DP3.compute_loss's observation path (normalize, clamp, encode), with the shared
        token-FPS start / shuffle. -> global_cond (B, To * G, D), pc_pe."""
        self._check_obs_widths(obs)
        obs, pc_len = self._split_lengths(obs)
        nobs = self.normalizer.normalize(obs)
        nobs["point_cloud"] = torch.clamp(nobs["point_cloud"], min=-1 - 1e-6, max=1 + 1e-6)
        if not self.use_pc_color:
            nobs["point_cloud"] = nobs["point_cloud"][..., :3]
        B = nobs["point_cloud"].shape[0]
        this = dict_apply(nobs, lambda x: x[:, :self.n_obs_steps].reshape(-1, *x.shape[2:]))
        this["point_cloud_len"] = pc_len.reshape(-1)
        this["fps_start_idx"], this["fps_perm"] = fps_start, fps_perm
        feats, pc_pe = self.obs_encoder(this, eval=True)
        G = pc_pe.shape[1]
        return (feats.reshape(B, self.n_obs_steps * feats.shape[1], -1),
                pc_pe.reshape(B, self.n_obs_steps * G, -1))

    def remover_loss(self, batch, k_masks=None):
        """phi surrogate for one phi step: K masks per window, CRN reward, RLOO.
        -> (surrogate with grad on mu only, stats)."""
        K = self.k_masks if k_masks is None else int(k_masks)
        B = batch["action"].shape[0]
        rem = self.link_remover
        z = rem.sample(B * K).view(B, K, -1)
        seeds = torch.randint(0, 2 ** 62, (B,))
        reward = self.crn_reward(batch, z, seeds)
        surr, adv = rloo_surrogate(rem.log_prob(z), reward)
        stats = {"remover_reward_mean": reward.mean().item(), "remover_reward_std": reward.std().item(),
                 "remover_adv_std": adv.std().item(), "remover_adv_absmean": adv.abs().mean().item(),
                 "remover_sigma": rem.sigma.item(), "remover_reward_filled_frac": self.reward_filled_frac}
        stats.update(self._effect_stats(z, adv))
        stats.update(self._mask_stats(z, adv))
        keep = rem.keep_prob()
        for l, name in enumerate(rem.region_names):
            stats[f"remover_mu/{name}"] = rem.mu[l].item()
            stats[f"remover_rho/{name}"] = rem.density()[l].item()
            stats[f"remover_gate/{name}"] = rem.gate[l].item()
            stats[f"remover_keep_prob/{name}"] = keep[l].item()
        return surr, stats

    @torch.no_grad()
    def heldout_check(self, train_batch, heldout_batch, n_masks=8, seeds=None, z=None):
        """Memorisation check of the proxy reward (docs/learned_remover_v0.md, risks):
        on training windows theta may have memorised the action, so removing points looks
        cheap there and phi would learn "removal is free" from memorisation. Draw n_masks
        mask draws SHARED by all windows and score each with the CRN reward on
          A  the training batch,             seeds s1
          A' the SAME training batch again,   seeds s2   -> noise floor of the comparison
          H  a batch from the held-out demos, seeds s3
        and report, for A vs A' (`floor_*`) and A vs H (`heldout_*`):
          mask_rank_spearman   rank correlation of the masks' mean reward
          grad_cosine          cosine of the RLOO phi-gradient (one value per region)
          grad_sign_agreement  fraction of regions whose gradient has the same sign
        plus reward_gap = mean(H) - mean(A). Read heldout_* AGAINST floor_*: if held-out
        agreement is as high as the floor, the training reward ranks masks as held-out data
        does; clearly lower = phi is (partly) learning from memorisation. The gradient is
        over all of phi (gate logits a, then density logits mu).
        `seeds` (3 tensors of (B,)) / `z` (J, 2L) only for tests; default: fresh."""
        rem = self.link_remover
        z = rem.sample(n_masks) if z is None else z                       # (J, 2L), shared
        n_masks = z.shape[0]
        sets = [(train_batch, "A"), (train_batch, "A2"), (heldout_batch, "H")]
        scores, grads = {}, {}
        for i, (batch, name) in enumerate(sets):
            B = batch["action"].shape[0]
            zb = z[None].expand(B, -1, -1).contiguous()
            sd = torch.randint(0, 2 ** 62, (B,)) if seeds is None else seeds[i]
            r = self.crn_reward(batch, zb, sd)                               # (B, J)
            scores[name] = r.mean(0)
            adv = r - (r.sum(1, keepdim=True) - r) / (n_masks - 1)          # leave-one-out
            with torch.enable_grad():
                g_phi = torch.autograd.grad((adv * rem.log_prob(zb)).mean(), rem.phi_parameters())
            grads[name] = torch.cat(g_phi).float()

        def rank(x):
            r = torch.argsort(torch.argsort(x)).float()
            return r - r.mean()

        out = {}
        for tag, other in (("floor", "A2"), ("heldout", "H")):
            ra, rb = rank(scores["A"]), rank(scores[other])
            out[f"remover_check/{tag}_mask_rank_spearman"] = (ra @ rb / (ra.norm() * rb.norm())).item()
            out[f"remover_check/{tag}_grad_cosine"] = torch.nn.functional.cosine_similarity(
                grads["A"], grads[other], dim=0).item()
            out[f"remover_check/{tag}_grad_sign_agreement"] = (
                torch.sign(grads["A"]) == torch.sign(grads[other])).float().mean().item()
        out["remover_check/reward_gap"] = (scores["H"].mean() - scores["A"].mean()).item()
        return out

    @torch.no_grad()
    def rank_check_ab(self, batch, n_masks=8, seeds=None, z=None):
        """Diagnostic B (docs/learned_remover_v0.md): is the proxy reward A (x0 error at a few
        fixed t) ranking masks like the error of the FULLY SAMPLED action (DDIM,
        num_inference_steps, same start noise and the same shared FPS draws for every mask),
        on the executed steps? n_masks draws shared by all windows.
        -> {ab_rank_spearman, a_scores, b_scores}."""
        rem = self.link_remover
        obs, cells = self._split_link(batch["obs"])
        B = cells.shape[0]
        To, T, dev = self.n_obs_steps, batch["action"].shape[1], cells.device
        z = rem.sample(n_masks) if z is None else z
        n_masks = z.shape[0]
        zb = z[None].expand(B, -1, -1).contiguous()
        seeds = torch.randint(0, 2 ** 62, (B,)) if seeds is None else seeds
        a_scores = self.crn_reward(batch, zb, seeds).mean(0)                  # (J,)
        draws = self.crn_draws(seeds, B, To, T, dev)                          # same shared draws
        nactions = self.normalizer["action"].normalize(batch["action"])
        s0, s1 = To - 1, To - 1 + self.n_action_steps
        was_training = self.training
        self.eval()
        try:
            b_scores = torch.zeros(n_masks, device=dev)
            for j in range(n_masks):
                tobs, res = self._thin_obs(obs, cells, rem.rho(zb[:, j]), u_region=draws["u_region"],
                                           u_fill=draws["u_fill"], return_keep=True)
                start = self._crn_start(cells[:, :To], res[5], draws, obs)
                global_cond, pc_pe = self._encode(tobs, start, draws["perm"].reshape(B * To, -1))
                traj = draws["eps"].to(nactions.dtype).clone()                # shared start noise
                self.noise_scheduler.set_timesteps(self.num_inference_steps)
                for t in self.noise_scheduler.timesteps:
                    out = self.model(sample=traj, timestep=t, local_cond=None, global_cond=global_cond,
                                     pc_pe=pc_pe, n_obs_steps=self.n_obs_steps)
                    # as DP3.conditional_sample: no kwargs (self.kwargs holds config leftovers)
                    traj = self.noise_scheduler.step(out, t, traj).prev_sample
                b_scores[j] = ((traj - nactions)[:, s0:s1].float() ** 2).mean()
        finally:
            self.train(was_training)

        def rank(x):
            r = torch.argsort(torch.argsort(x)).float()
            return r - r.mean()

        ra, rb = rank(a_scores), rank(b_scores)
        return {"ab_rank_spearman": (ra @ rb / (ra.norm() * rb.norm())).item(),
                "a_scores": a_scores.tolist(), "b_scores": b_scores.tolist()}

    def _mask_stats(self, z, adv):
        """Structured-mix diagnostic: per structured mask s, the mean advantage of the draws
        equal to it (adv is already relative to the other masks of the same window; > 0 =
        worse) and their count. Empty without a mix."""
        rem = self.link_remover
        mix = rem.mix_table(z.device)
        if mix is None:
            return {}
        _, g = rem.split(z)
        a = adv.detach().float()
        out = {}
        for i, name in enumerate(rem.mix_names):
            hit = (g == mix[i]).all(-1)
            out[f"remover_mask_n/{name}"] = int(hit.sum())
            if bool(hit.any()):
                out[f"remover_mask_effect/{name}"] = a[hit].mean().item()
        return out

    def _effect_stats(self, z, adv):
        """What the reward says about each region in THIS phi step (adv > 0 = worse than the
        window's other masks): gate effect = mean adv with the gate closed - open (< 0: removing
        the region helps), density slope = least-squares slope of adv on z over open draws
        (< 0: denser helps), plus the counts behind them. Accumulated over steps by
        scripts/sim/analyze_remover_run.py into estimates with confidence intervals."""
        rem = self.link_remover
        zz, gg = rem.split(z)
        a = adv.detach().float()
        out = {}
        for l, name in enumerate(rem.region_names):
            closed = gg[..., l] == 0
            n_c, n_o = int(closed.sum()), int((~closed).sum())
            out[f"remover_n_closed/{name}"] = n_c
            if n_c and n_o:
                out[f"remover_effect_gate/{name}"] = (a[closed].mean() - a[~closed].mean()).item()
            if n_o > 2 and rem.learn_density:
                zo, ao = zz[..., l][~closed].float(), a[~closed]
                zc = zo - zo.mean()
                if float((zc ** 2).sum()) > 0:
                    out[f"remover_effect_density/{name}"] = (float((zc * (ao - ao.mean())).sum()) / float((zc ** 2).sum()))
        return out

    @torch.no_grad()
    def mechanism_check(self, batch, masks=("none", "deterministic", "sampled")):
        """Where the encoder's tokens land on the thinned clouds -- the mechanism the remover
        is meant to move (FPS allocates tokens by surface extent; removing surface frees them).
        For the CURRENT observation frame of each window and each mask
          none           no thinning (the uncut baseline's cloud)
          deterministic  the inference mask (gates 1[pi >= 0.5], snapped density)
          sampled        one draw per window (what theta trains on)
        runs the encoder's grouping (token_allocation: FPS to num_group centres on the
        normalized xyz, random start as in training, kNN group_size) and reports per class
        (each region, link0-4, nut, peg, other): token centres, coverage (share of the class's
        points in some patch) and the median patch radius in cm of tokens centred on it.
        -> {"mech/<mask>/<stat>/<class>": value}."""
        import numpy as np
        from r3d.common.token_allocation import token_allocation
        rem, enc = self.link_remover, self._uni3d()
        if LABEL_KEY not in batch["obs"]:
            raise KeyError(f"mechanism_check needs obs '{LABEL_KEY}' (dataset load_point_link)")
        lab_all = batch["obs"][LABEL_KEY]
        obs, cells = self._split_link(batch["obs"])
        To = self.n_obs_steps
        f = To - 1
        B = cells.shape[0]
        L = rem.n_regions
        names = list(rem.region_names) + ["link0-4", "nut", "peg", "other"]
        out = {}
        for mask in masks:
            if mask == "none":
                rho = torch.ones(B, L, device=cells.device)
            elif mask == "deterministic":
                rho = rem.rho()[None].expand(B, -1)
            elif mask == "sampled":
                rho = rem.rho(rem.sample(B))
            else:
                raise ValueError(mask)
            tobs, res = self._thin_obs(obs, cells, rho, random_start=True, return_keep=True)
            keep = res[5][:, f]
            acc = {k: [] for k in ("tokens", "coverage")}
            radii = [[] for _ in names]
            for b in range(B):
                kb = keep[b]
                pts = obs["point_cloud"][b, f][kb]
                cb, lb = cells[b, f][kb], lab_all[b, f][kb].long()
                reg = torch.where(cb >= 0, rem.cell_to_region[cb.clamp(min=0)], torch.full_like(cb, -1))
                cls = torch.where(reg >= 0, reg, torch.where(cb >= 0, torch.full_like(cb, L),
                      torch.where(lb == RP.OBJECT, torch.full_like(cb, L + 1),
                      torch.where(lb == RP.TARGET, torch.full_like(cb, L + 2), torch.full_like(cb, L + 3)))))
                xyz_n = self.normalizer["point_cloud"].normalize(pts)[:, :3]
                ta = token_allocation(xyz_n[None].float().cpu().numpy(), cls[None].cpu().numpy(), len(names),
                                      num_group=int(enc.num_group),
                                      group_size=int(enc.group_size), random_start=True,
                                      device=str(pts.device), return_idx=True)
                acc["tokens"].append(ta["centres"][0])
                acc["coverage"].append(ta["coverage"][0])
                ci, ki = ta["centre_idx"][0], ta["knn_idx"][0]
                p = pts[:, :3].float().cpu().numpy()
                r = np.linalg.norm(p[ki] - p[ci][:, None], axis=-1).max(-1)
                cl = cls.cpu().numpy()[ci]
                for c in range(len(names)):
                    radii[c].extend(r[cl == c].tolist())
            tok = np.mean(acc["tokens"], 0)
            cov = np.nanmean(np.stack(acc["coverage"]), 0) if len(acc["coverage"]) else None
            out[f"mech/{mask}/points_kept"] = float(res[1][:, f].float().mean())
            for c, nm in enumerate(names):
                out[f"mech/{mask}/tokens/{nm}"] = float(tok[c])
                if cov is not None and np.isfinite(cov[c]):
                    out[f"mech/{mask}/coverage/{nm}"] = float(cov[c])
                if radii[c]:
                    out[f"mech/{mask}/patch_cm/{nm}"] = 100.0 * float(np.median(radii[c]))
        return out

    # ---- inference ----------------------------------------------------------------------
    def predict_action(self, obs_dict):
        obs, cells = self._split_link(obs_dict)
        if self.link_remover is None:
            return super().predict_action(obs_dict)
        B = cells.shape[0]
        rho = self.link_remover.rho()[None].expand(B, -1)
        obs, _ = self._thin_obs(obs, cells, rho, random_start=True)
        return super().predict_action(obs)
