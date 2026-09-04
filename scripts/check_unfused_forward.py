#!/usr/bin/env python3
"""End-to-end forward pass on the unfused config, through real Hydra composition.

Builds the policy exactly as train.py would -- so the encoder auto-detection,
the ${oc.select:...} plumbing for num_groups_per_camera and the normalizer keys
are all exercised as they will be in the real run -- then pushes one real batch
through compute_loss and predict_action.

    python scripts/check_unfused_forward.py                    # Run B
    python scripts/check_unfused_forward.py --pose-source proprio   # A'
"""
import argparse
import pathlib
import sys

import numpy as np
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

sys.path.insert(0, "R3D")
if not OmegaConf.has_resolver("eval"):
    OmegaConf.register_new_resolver("eval", eval, replace=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="real_bimanual_unfused")
    ap.add_argument("--alg", default="r3d_robotwin2")
    ap.add_argument("--pose-source", default=None,
                    choices=["extrinsic", "proprio", "none"])
    ap.add_argument("--batch", type=int, default=2)
    args = ap.parse_args()

    cfg_dir = str(pathlib.Path("R3D/r3d/config").resolve())
    overrides = [f"task={args.task}", f"task_name={args.task}"]
    if args.pose_source:
        overrides.append(f"task.dataset.pose_source={args.pose_source}")
    with initialize_config_dir(config_dir=cfg_dir, version_base=None):
        cfg = compose(config_name=args.alg, overrides=overrides)

    ngpc = cfg.policy.get("num_groups_per_camera", None)
    print(f"num_groups_per_camera resolved to: "
          f"{OmegaConf.to_container(ngpc) if ngpc is not None else None}")
    print(f"pose_source: {cfg.task.dataset.get('pose_source', 'n/a (fused)')}")

    import hydra
    dataset = hydra.utils.instantiate(cfg.task.dataset)
    policy = hydra.utils.instantiate(cfg.policy)
    policy.set_normalizer(dataset.get_normalizer())

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    policy = policy.to(dev)

    loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch, shuffle=False, num_workers=0)
    batch = next(iter(loader))
    batch = {k: (v.to(dev) if torch.is_tensor(v)
                 else {kk: vv.to(dev) for kk, vv in v.items()})
             for k, v in batch.items()}

    print("\nbatch keys:")
    for k, v in batch["obs"].items():
        print(f"  obs/{k:20s} {tuple(v.shape)}")
    print(f"  action{'':15s} {tuple(batch['action'].shape)}")

    if dev == "cuda":
        torch.cuda.reset_peak_memory_stats()
    loss, loss_dict = policy.compute_loss(batch)
    print(f"\ncompute_loss -> {loss.item():.6f}  {loss_dict}")
    assert torch.isfinite(loss), "loss is not finite"
    loss.backward()
    if dev == "cuda":
        print(f"peak CUDA memory (fwd+bwd, batch {args.batch}): "
              f"{torch.cuda.max_memory_allocated() / 2**30:.2f} GiB")

    unused = [n for n, p in policy.named_parameters()
              if p.requires_grad and p.grad is None]
    pose_params = [n for n, _ in policy.named_parameters() if "pose_embed" in n]
    pose_no_grad = [n for n in pose_params
                    if dict(policy.named_parameters())[n].grad is None]
    print(f"parameters with no grad: {len(unused)}")
    print(f"pose_embed parameters  : {len(pose_params)}, of which no grad: "
          f"{len(pose_no_grad)} {pose_no_grad}")

    with torch.no_grad():
        out = policy.predict_action(batch["obs"])
    print(f"predict_action -> action {tuple(out['action'].shape)}, "
          f"action_pred {tuple(out['action_pred'].shape)}")
    assert torch.isfinite(out["action"]).all()
    print("\nFORWARD/BACKWARD OK")


if __name__ == "__main__":
    main()
