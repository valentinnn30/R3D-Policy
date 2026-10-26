"""Training schedule of the learned link remover v0 (docs/learned_remover_v0.md).

train.py builds a `RemoverController` only when the policy has a link remover
(`DP3LinkRemover` with `link_remover.enabled`); without one none of this runs and train.py
is the baseline. The controller owns phi's optimizer and the phases, by epoch:

  warm-up   [0, warmup_frac)            theta only; masks from the FIXED initial phi (gates
                                         open with init_keep, mu at logit(init_rho), sigma =
                                         sigma_start), so theta has seen removed and thinned
                                         regions before phi starts judging them
  phi       [warmup_frac, 1-finetune)    joint: every `phi_every` optimizer steps one phi
                                         step on the windows of that optimizer step (all its
                                         micro-batches: 2 x 64 = 128 at accum 2), K masks
                                         each, CRN reward, RLOO; sigma annealed linearly
                                         sigma_start -> sigma_end over the phase
  fine-tune [1-finetune_frac, 1]         phi frozen; theta trains on the deterministic,
                                         snapped rho -- exactly what evaluation shows it

phi = the remover's removal-gate logits a and density logits mu: one Adam, both clamped
after every step. theta's optimizer never sees them (`theta_parameters`). The EMA model gets
the LIVE remover state after every EMA step: evaluation = EMA theta + the current
deterministic gates and rho, not an average of phi.

Diagnostics (every `diag_every` epochs, on the live model, logged under remover_check/ and
mech/): the held-out memorisation check (`heldout_check`: recent training windows vs held-out
demos, against the same windows under fresh seeds as the noise floor), diagnostic B
(`rank_check_ab`: does the proxy reward rank masks like the fully sampled action?) and the
mechanism (`mechanism_check`: token centres / coverage / patch radius per region, nut, peg
on the uncut, the deterministic and a sampled thinned cloud).

Everything is also written to `<run>/remover_log.jsonl` -- one line per phi step
({"kind": "phi", ...}: gates, densities, logits, their gradients, reward / advantage stats,
per-region effect estimates) and one per epoch ({"kind": "epoch", ...}: theta-pass means,
losses, diagnostics, the current gates and densities) -- read by
scripts/sim/analyze_remover_run.py.
"""

import contextlib
import json
import os

import numpy as np
import torch

from r3d.common.pytorch_util import dict_apply


def theta_parameters(model):
    """Every parameter except the remover's mu -- theta's optimizer."""
    return [p for n, p in model.named_parameters() if not n.startswith("link_remover.")]


def _window_slice(batch, To):
    """Only what phi / the diagnostics read: obs[:, :To] and the action. Copied, so a kept
    window does not pin the whole horizon-long batch in GPU memory."""
    return {"obs": {k: v[:, :To].clone() for k, v in batch["obs"].items()}, "action": batch["action"].clone()}


def _cat(batches):
    return {"obs": {k: torch.cat([b["obs"][k] for b in batches]) for k in batches[0]["obs"]},
            "action": torch.cat([b["action"] for b in batches])}


class RemoverController:

    def __init__(self, model, cfg, num_epochs):
        self.rem = model.link_remover
        if self.rem is None:
            raise ValueError("RemoverController needs a policy with a link remover")
        self.lr = float(cfg.lr)
        self.phi_every = int(cfg.phi_every)
        self.num_epochs = int(num_epochs)
        self.warmup_end = int(round(float(cfg.warmup_frac) * num_epochs))
        self.finetune_start = num_epochs - int(round(float(cfg.finetune_frac) * num_epochs))
        if not 0 <= self.warmup_end < self.finetune_start <= num_epochs:
            raise ValueError(f"remover phases: warm-up ends at epoch {self.warmup_end}, fine-tune "
                             f"starts at {self.finetune_start} of {num_epochs} -- no phi phase")
        self.sigma_start, self.sigma_end = float(cfg.sigma_start), float(cfg.sigma_end)
        self.diag_every = int(cfg.diag_every)
        self.diag_batches = int(cfg.diag_batches)
        self.diag_masks = int(cfg.diag_masks)
        self.optimizer = torch.optim.Adam([self.rem.gate, self.rem.mu], lr=self.lr)
        self.pending, self.recent = [], []
        self.phase = None
        self._epoch_frac = 0.0
        self.log_path = None
        self.batch_acc = {}

    def set_output_dir(self, output_dir):
        self.log_path = os.path.join(str(output_dir), "remover_log.jsonl")

    def _write(self, rec):
        if self.log_path is not None:
            with open(self.log_path, "a") as fh:
                fh.write(json.dumps(rec) + "\n")

    def _phi_state(self):
        rho, kp, det = torch.sigmoid(self.rem.mu.detach()), self.rem.keep_prob(), self.rem.rho()
        return {name: {"mu": self.rem.mu[l].item(), "gate": self.rem.gate[l].item(), "rho": rho[l].item(),
                       "keep_prob": kp[l].item(), "det_rho": det[l].item()}
                for l, name in enumerate(self.rem.region_names)}

    # ---- per batch / per optimizer step ---------------------------------------------------
    def phase_of(self, epoch):
        if epoch < self.warmup_end:
            return "warmup"
        return "finetune" if epoch >= self.finetune_start else "phi"

    def begin_batch(self, model, epoch, frac):
        """Set the phase, sigma and theta's mask mode before the batch's compute_loss."""
        self._epoch_frac = epoch + frac
        self.phase = self.phase_of(epoch)
        if self.phase == "phi":
            prog = (epoch + frac - self.warmup_end) / (self.finetune_start - self.warmup_end)
            self.rem.set_sigma(self.sigma_start + (self.sigma_end - self.sigma_start) * min(max(prog, 0.0), 1.0))
        elif self.phase == "warmup":
            self.rem.set_sigma(self.sigma_start)
        model.remover_mode = "deterministic" if self.phase == "finetune" else "sample"

    def remember(self, batch, To):
        """Keep this micro-batch for the next phi step (phi phase) and the diagnostics."""
        w = _window_slice(batch, To)
        if self.phase == "phi":
            self.pending.append(w)
        self.recent = (self.recent + [w])[-self.diag_batches:]

    def after_opt_step(self, model, opt_step, amp):
        """After theta's optimizer step: a phi step every `phi_every` optimizer steps in the
        phi phase, on the windows of this optimizer step. -> log dict."""
        pending, self.pending = self.pending, []
        log = {"remover_sigma": self.rem.sigma.item(),
               "remover_phase": {"warmup": 0, "phi": 1, "finetune": 2}[self.phase]}
        if self.phase != "phi" or opt_step % self.phi_every != 0 or not pending:
            return log
        with amp():
            surr, stats = model.remover_loss(_cat(pending))
        self.optimizer.zero_grad(set_to_none=True)
        surr.backward()
        for l, name in enumerate(self.rem.region_names):
            log[f"remover_grad_gate/{name}"] = self.rem.gate.grad[l].item()
            log[f"remover_grad_mu/{name}"] = self.rem.mu.grad[l].item()
        self.optimizer.step()
        self.rem.clamp_()
        log.update(stats)
        log["remover_phi_windows"] = int(sum(p["action"].shape[0] for p in pending))
        skip = ("remover_mu/", "remover_rho/", "remover_gate/", "remover_keep_prob/")   # in "phi" below
        self._write({"kind": "phi", "opt_step": int(opt_step), "epoch_frac": self._epoch_frac,
                     **{k: v for k, v in log.items() if not k.startswith(skip)}, "phi": self._phi_state()})
        return log

    def sync_ema(self, ema_model):
        """EMA theta + the LIVE remover (mu is not averaged)."""
        ema_model.link_remover.load_state_dict(self.rem.state_dict())

    # ---- per epoch ------------------------------------------------------------------------
    def record_batch(self, loss_dict):
        """theta-pass remover stats of one micro-batch (kept fractions, ...), averaged per epoch."""
        for k, v in loss_dict.items():
            if k.startswith("remover_") and isinstance(v, (int, float)) and np.isfinite(v):
                self.batch_acc.setdefault(k, []).append(float(v))

    def end_epoch(self, model, val_dataloader, device, amp, epoch, step_log):
        """Every epoch: the epoch record (theta-pass means, losses, phi state); every
        diag_every epochs also the diagnostics. -> dict for the step log."""
        out = {f"{k}_epoch": float(np.mean(v)) for k, v in self.batch_acc.items()}
        self.batch_acc = {}
        if self.diag_every > 0 and epoch % self.diag_every == 0 and self.recent:
            out.update(self.diagnostics(model, val_dataloader, device, amp))
        self._write({"kind": "epoch", "epoch": int(epoch), "phase": self.phase_of(epoch),
                     "sigma": self.rem.sigma.item(),
                     **{k: float(v) for k, v in step_log.items()
                        if k in ("train_loss", "val_loss", "train_action_mse_error") and v is not None},
                     **out, "phi": self._phi_state()})
        return out

    def diagnostics(self, model, val_dataloader, device, amp):
        """held-out memorisation check + diagnostic B + mechanism. -> log dict."""
        To = model.n_obs_steps
        val = []
        for b in val_dataloader:
            val.append(_window_slice(dict_apply(b, lambda x: x.to(device, non_blocking=True)), To))
            if len(val) >= len(self.recent):
                break
        acc = {}
        with amp():
            for tr, ho in zip(self.recent, val):
                for k, v in model.heldout_check(tr, ho, n_masks=self.diag_masks).items():
                    acc.setdefault(k, []).append(v)
            for ho in val:
                acc.setdefault("remover_check/ab_rank_spearman", []).append(
                    model.rank_check_ab(ho, n_masks=self.diag_masks)["ab_rank_spearman"])
            for ho in val[:2]:
                for k, v in model.mechanism_check(ho).items():
                    acc.setdefault(k, []).append(v)
        log = {k: float(np.nanmean(v)) for k, v in acc.items()}
        log["remover_check/n_batches"] = len(val)
        return log

    @staticmethod
    def amp_context(enabled, dtype):
        return lambda: (torch.autocast(device_type="cuda", dtype=dtype) if enabled
                        else contextlib.nullcontext())
