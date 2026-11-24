#!/usr/bin/env python
"""Checks for the token-dropout control (standard PatchDropout, 2026-11-12). Run in r3d_sim:

    python scripts/sim/check_token_dropout.py

The policy is composed exactly as the run (square_d2, 4096 points, 64 tokens) with
policy.pointcloud_encoder_cfg.patch_dropout=0.25, on real training windows from the zarr.

1. Default off (patch_dropout 0): the encoder holds nn.Identity, so the code path is git HEAD's.
2. Training forward: 48 = int(64 * 0.75) tokens per obs frame, tokens and pc_pe the same length,
   kept centres == a subset of the full centres in their original order (same RNG state).
3. Masks are uniform and independent: every token position kept at ~0.75, per frame (not per
   sample) different masks, nothing class- or position-dependent.
4. Eval forward (predict_action path, eval=True): all 64 tokens, deterministic given the FPS seed.
5. compute_loss + backward run with dropout on; predict_action runs.
"""

import pathlib
import sys

import torch

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "R3D"))
FAILS = []


def check(name, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name}{'  -- ' + detail if detail else ''}", flush=True)
    if not ok:
        FAILS.append(name)


def compose(overrides):
    from hydra import compose as hcompose, initialize_config_dir
    from omegaconf import OmegaConf
    OmegaConf.register_new_resolver("eval", eval, replace=True)
    with initialize_config_dir(config_dir=str(ROOT / "R3D" / "r3d" / "config"), version_base=None):
        return hcompose("r3d_robotwin2", overrides=["task=mimicgen_sim", "task_name=square_d2",
                                                    "setting=agentview_p4096-200", "task.num_points=4096",
                                                    "task.num_group=64"] + overrides)


def main():
    import os
    import hydra
    from r3d.model.vision.pointnet_extractor import PatchDropout
    from r3d.common.pytorch_util import dict_apply
    os.chdir(ROOT)                               # zarr_path is relative to the repo root, as in train.py
    dev = torch.device("cuda:0")

    cfg0 = compose([])
    pol0 = hydra.utils.instantiate(cfg0.policy)
    check("default off: patch_dropout is Identity",
          isinstance(pol0.obs_encoder.extractor.patch_dropout, torch.nn.Identity))
    del pol0

    cfg = compose(["policy.pointcloud_encoder_cfg.patch_dropout=0.25"])
    policy = hydra.utils.instantiate(cfg.policy).to(dev)
    enc = policy.obs_encoder.extractor
    check("on: PatchDropout(0.25)", isinstance(enc.patch_dropout, PatchDropout) and enc.patch_dropout.prob == 0.25)

    dataset = hydra.utils.instantiate(cfg.task.dataset)
    policy.set_normalizer(dataset.get_normalizer())
    policy.to(dev)
    loader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True, num_workers=0)
    batch = dict_apply(next(iter(loader)), lambda x: x.to(dev))
    nobs = policy.normalizer.normalize(batch["obs"])
    nobs = dict_apply(nobs, lambda x: x[:, :policy.n_obs_steps].reshape(-1, *x.shape[2:]))
    pc = torch.clamp(nobs["point_cloud"], min=-1 - 1e-6, max=1 + 1e-6)   # as dp3.py before the encoder
    n = pc.shape[0]

    # 2. training forward: token count and centre alignment -------------------------------------
    policy.train()
    # one forward: capture the full centres (patch_embed output), the kept indices and the
    # centres that reach pe_layer -> the head's pc_pe must be built from exactly the kept centres
    enc_d = enc.patch_dropout
    captured = {}
    pe_fwd, emb_fwd, keep_fn = enc.pe_layer.forward, enc.patch_embed.forward, enc_d.keep_indices
    def emb_spy(*a, **k):
        out = emb_fwd(*a, **k)
        captured["full"] = (out[-1] if isinstance(out, list) else out)["centers"]
        return out
    def keep_spy(*a, **k):
        captured["idx"] = keep_fn(*a, **k)
        return captured["idx"]
    def pe_spy(c):
        captured["pe_in"] = c
        return pe_fwd(c)
    enc.patch_embed.forward, enc_d.keep_indices, enc.pe_layer.forward = emb_spy, keep_spy, pe_spy
    with torch.no_grad():
        feat, pe = enc(pc, eval=False)
    del enc.patch_embed.forward, enc_d.keep_indices, enc.pe_layer.forward
    check("train: 48 tokens per frame", feat.shape[1] == 48 and pe.shape[1] == 48,
          f"x {tuple(feat.shape)}, pc_pe {tuple(pe.shape)}")
    idx = captured["idx"]
    want = torch.gather(captured["full"], 1, idx.unsqueeze(-1).expand(-1, -1, 3))
    check("train: pc_pe built from the full centres at the kept indices (alignment)",
          torch.equal(captured["pe_in"], want))
    check("train: kept indices sorted, unique", bool((idx.diff(dim=1) > 0).all()))

    # 3. uniform, independent per frame -----------------------------------------------------------
    counts = torch.zeros(64, device=dev)
    reps = 400
    for _ in range(reps):
        k = enc_d.keep_indices(n, 64, dev)
        counts += torch.zeros(n, 64, device=dev).scatter_(1, k, 1.0).sum(0)
    frac = counts / (reps * n)
    check("masks: every position kept at ~0.75", bool(((frac - 0.75).abs() < 0.03).all()),
          f"min {frac.min():.3f} max {frac.max():.3f}")
    k = enc_d.keep_indices(n, 64, dev)
    distinct = len({tuple(r.tolist()) for r in k})
    check("masks: independent per frame", distinct == n, f"{distinct} distinct of {n}")

    # 4. eval forward: all tokens ---------------------------------------------------------------
    policy.eval()
    with torch.no_grad():
        fe, pee = enc(pc, eval=True)
    check("eval: 64 tokens", fe.shape[1] == 64 and pee.shape[1] == 64, f"{tuple(fe.shape)}")

    # 5. loss + backward, predict_action -----------------------------------------------------------
    policy.train()
    loss, _ = policy.compute_loss(batch)
    loss.backward()
    g = sum(p.grad.abs().sum().item() for p in enc.parameters() if p.grad is not None)
    check("compute_loss + backward", torch.isfinite(loss).item() and g > 0, f"loss {loss.item():.4f}")
    policy.eval()
    with torch.no_grad():
        out = policy.predict_action(batch["obs"])
    check("predict_action", torch.isfinite(out["action"]).all().item(), f"{tuple(out['action'].shape)}")

    print("ALL PASS" if not FAILS else f"FAILED: {FAILS}")
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
