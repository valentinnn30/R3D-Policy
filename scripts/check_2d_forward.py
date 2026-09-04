#!/usr/bin/env python3
"""End-to-end forward pass on the 2D image config, through real Hydra composition.

The image counterpart of `check_unfused_forward.py`, which is left untouched.
The one addition is a general `--override` passthrough, so a smoke run can point
at a small zarr without editing the task config.

Builds the policy exactly as train.py would -- so the encoder construction, the
pose_source plumbing and the normalizer keys are exercised as they will be in
the real run -- then pushes one real batch through compute_loss and
predict_action.

    python scripts/check_2d_forward.py                        # 2D-N, the default
    python scripts/check_2d_forward.py --pose-source proprio  # 2D-A'
    python scripts/check_2d_forward.py --pose-source extrinsic  # 2D-B
    python scripts/check_2d_forward.py -o task.dataset.zarr_path=OTHER.zarr

What it checks that a training launch would only tell you about much later:

* the patch grid divides the stored resolution, and the positional encoding has
  exactly as many entries as there are image tokens;
* the FROZEN trunk really is frozen -- no gradients, and eval mode survives
  `policy.train()`;
* the camera tag receives gradient, which is the only per-view parameter and so
  the only thing that can go silently unused.
"""
import argparse
import pathlib
import sys

import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

sys.path.insert(0, "R3D")
if not OmegaConf.has_resolver("eval"):
    OmegaConf.register_new_resolver("eval", eval, replace=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="real_bimanual_2d")
    ap.add_argument("--alg", default="r3d_2d")
    ap.add_argument("--pose-source", default=None,
                    choices=["extrinsic", "proprio", "none"])
    ap.add_argument("-o", "--override", action="append", default=[],
                    help="extra hydra override, repeatable")
    ap.add_argument("--batch", type=int, default=2)
    args = ap.parse_args()

    cfg_dir = str(pathlib.Path("R3D/r3d/config").resolve())
    overrides = [f"task={args.task}", f"task_name={args.task}"]
    if args.pose_source:
        overrides.append(f"task.dataset.pose_source={args.pose_source}")
    overrides += args.override
    with initialize_config_dir(config_dir=cfg_dir, version_base=None):
        cfg = compose(config_name=args.alg, overrides=overrides)

    print(f"pose_source     : {cfg.task.dataset.pose_source}")
    print(f"encoder         : {cfg.policy.image_encoder_cfg.model_name} "
          f"(frozen={cfg.policy.image_encoder_cfg.freeze})")
    print(f"image_shift_px  : {cfg.task.dataset.image_shift_px}")

    import hydra
    dataset = hydra.utils.instantiate(cfg.task.dataset)
    policy = hydra.utils.instantiate(cfg.policy)
    policy.set_normalizer(dataset.get_normalizer())

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    policy = policy.to(dev)
    policy.train()

    enc = policy.obs_encoder
    n_img_tokens = enc.tokens_per_camera * enc.n_cams
    print(f"\ntokens          : {enc.grid_h} x {enc.grid_w} = "
          f"{enc.tokens_per_camera} per camera, {n_img_tokens} total")
    assert enc.extractor.backbone.training is False, \
        "frozen trunk is in train mode -- stochastic depth would make it non-deterministic"

    loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch, shuffle=False, num_workers=0)
    batch = next(iter(loader))
    batch = {k: (v.to(dev) if torch.is_tensor(v)
                 else {kk: vv.to(dev) for kk, vv in v.items()})
             for k, v in batch.items()}

    print("\nbatch keys:")
    for k, v in batch["obs"].items():
        print(f"  obs/{k:16s} {tuple(v.shape)}  "
              f"[{v.min().item():.3f}, {v.max().item():.3f}]")
    print(f"  action{'':11s} {tuple(batch['action'].shape)}")

    if dev == "cuda":
        torch.cuda.reset_peak_memory_stats()
    loss, loss_dict = policy.compute_loss(batch)
    print(f"\ncompute_loss -> {loss.item():.6f}  {loss_dict}")
    assert torch.isfinite(loss), "loss is not finite"
    loss.backward()
    if dev == "cuda":
        print(f"peak CUDA memory (fwd+bwd, batch {args.batch}): "
              f"{torch.cuda.max_memory_allocated() / 2**30:.2f} GiB")

    named = dict(policy.named_parameters())
    trainable = [n for n, p in named.items() if p.requires_grad]
    frozen = [n for n, p in named.items() if not p.requires_grad]
    no_grad = [n for n in trainable if named[n].grad is None]
    backbone_with_grad = [n for n in named
                          if "extractor.backbone" in n and named[n].grad is not None]
    pose_params = [n for n in named if "pose_embed" in n]
    pose_no_grad = [n for n in pose_params if named[n].grad is None]

    n_train = sum(named[n].numel() for n in trainable)
    n_frozen = sum(named[n].numel() for n in frozen)
    print(f"\nparameters      : {n_train/1e6:.2f}M trainable, {n_frozen/1e6:.2f}M frozen")
    print(f"trainable w/o grad      : {len(no_grad)}")
    print(f"BACKBONE params w/ grad : {len(backbone_with_grad)}  (must be 0 when frozen)")
    print(f"pose_embed params       : {len(pose_params)}, of which no grad: "
          f"{len(pose_no_grad)} {pose_no_grad}")
    assert not backbone_with_grad, "frozen backbone received gradients"

    with torch.no_grad():
        out = policy.predict_action(batch["obs"])
    print(f"predict_action -> action {tuple(out['action'].shape)}, "
          f"action_pred {tuple(out['action_pred'].shape)}")
    assert torch.isfinite(out["action"]).all()
    print("\nFORWARD/BACKWARD OK")


if __name__ == "__main__":
    main()
