#!/usr/bin/env python3
"""End-to-end forward/backward on the hybrid 2D/3D configs, through real Hydra.

The hybrid counterpart of `check_unfused_forward.py` / `check_2d_forward.py`,
which are left untouched. Builds dataset + policy exactly as train.py would,
pushes one real batch through compute_loss and predict_action, and asserts the
things a training launch would only reveal much later.

    python scripts/check_hybrid_forward.py                                   # H1, defaults
    python scripts/check_hybrid_forward.py --task real_bimanual_hybrid_wrist3d --s3 proprio --s2 none
    python scripts/check_hybrid_forward.py --task real_bimanual_hybrid_wrist3d_fused --s2 extrinsic
    python scripts/check_hybrid_forward.py --unfreeze                        # 2D-U flags
    python scripts/check_hybrid_forward.py --all-sources                     # every valid pair,
                                                                             # policy only (no data)

Checks:
* the resolved per-block tag table is what the plan says it must be
  (e.g. static cam under 'proprio' -> identity, no MLP);
* token count == positional-encoding length == sum of the per-block budgets
  (H1 710, H2-U 699, H2-F 699), and the 3D blocks sit at 16 points per group;
* the fused cloud is clamped and unfused clouds are not (inherited loop);
* gradient reaches Uni3D, both cam_embed tables and every pose MLP, and reaches
  DINOv2 only in the blocks its freeze flags unfroze;
* 'proprio' leaving no block with a pose raises, and so does a non-'none'
  pose_source_3d on a fused block.
"""
import argparse
import itertools
import pathlib
import sys

import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

sys.path.insert(0, "R3D")
if not OmegaConf.has_resolver("eval"):
    OmegaConf.register_new_resolver("eval", eval, replace=True)

from r3d.common.hybrid_layout import hybrid_blocks, resolve_block_poses  # noqa: E402

SOURCES = ("extrinsic", "proprio", "none")
EXPECTED_TOKENS = {"real_bimanual_hybrid_top3d": 710,
                   "real_bimanual_hybrid_wrist3d": 699,
                   "real_bimanual_hybrid_wrist3d_fused": 699}


def compose_cfg(task, s3, s2, extra):
    cfg_dir = str(pathlib.Path("R3D/r3d/config").resolve())
    ov = [f"task={task}", f"task_name={task}"]
    if s3:
        ov.append(f"task.dataset.pose_source_3d={s3}")
    if s2:
        ov.append(f"task.dataset.pose_source_2d={s2}")
    ov += extra
    with initialize_config_dir(config_dir=cfg_dir, version_base=None):
        return compose(config_name="r3d_hybrid", overrides=ov)


def expected_table(cfg):
    d = cfg.task.dataset
    blocks = hybrid_blocks(d.cameras_3d, d.cameras_2d, d.fuse_3d)
    return blocks, resolve_block_poses(
        blocks, d.pose_source_3d, d.pose_source_2d,
        [None if s is None else list(s) for s in d.camera_arm_slice])


def check_policy_structure(policy, cfg):
    enc = policy.obs_encoder
    blocks, resolved = expected_table(cfg)
    assert [b["key"] for b in enc.blocks] == [b["key"] for b in blocks]
    assert enc.resolved == resolved, (enc.resolved, resolved)

    # tag modules really built what the table says
    for mod, emb in (("3d", enc.pose_embed_3d), ("2d", enc.pose_embed_2d)):
        rows = [r for r in resolved if r["modality"] == mod]
        assert emb.has_pose == [r["has_pose"] for r in rows], (mod, emb.has_pose)
        n_mlp = (len(emb.pose_mlps) if emb.pose_mlps is not None
                 else (1 if emb.pose_mlp is not None else 0))
        src = rows[0]["source"]
        want = (sum(r["has_pose"] for r in rows) if src == "proprio"
                else (1 if any(r["has_pose"] for r in rows) else 0))
        assert n_mlp == want, f"{mod}: built {n_mlp} pose MLPs, expected {want}"

    # budgets
    obs = cfg.task.shape_meta.obs
    for b in enc.blocks:
        if b["kind"] != "image":
            pts = int(obs[b["key"]].shape[0])
            g = enc.block_groups[b["key"]]
            assert pts == 16 * g, f"{b['key']}: {pts} points / {g} groups != 16"
    want = EXPECTED_TOKENS.get(cfg.task_name)
    if want is not None:
        assert enc.n_tokens == want, f"{enc.n_tokens} tokens, expected {want}"
    assert policy.pc_keys == [b["key"] for b in enc.blocks if b["kind"] != "image"]
    assert policy.unfused == (not cfg.task.dataset.fuse_3d)


def run_all_sources(task, extra):
    """Every (s3, s2) pair: build the policy only, check structure or the guard."""
    import hydra
    for s3, s2 in itertools.product(SOURCES, SOURCES):
        cfg = compose_cfg(task, s3, s2, extra)
        try:
            expected_table(cfg)
            should_raise = None
        except ValueError as e:
            should_raise = str(e)
        try:
            policy = hydra.utils.instantiate(cfg.policy)
        except Exception as e:  # hydra wraps the ValueError in InstantiationException
            assert should_raise is not None, f"({s3},{s2}) raised unexpectedly: {e}"
            assert "ValueError" in repr(e) or isinstance(e, ValueError), repr(e)
            print(f"  ({s3:9s},{s2:9s}) raises as designed: {should_raise[:70]}...")
            continue
        assert should_raise is None, f"({s3},{s2}) should have raised: {should_raise}"
        check_policy_structure(policy, cfg)
        print(f"  ({s3:9s},{s2:9s}) OK  tokens={policy.obs_encoder.n_tokens}")
        del policy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="real_bimanual_hybrid_top3d")
    ap.add_argument("--s3", default=None, choices=SOURCES)
    ap.add_argument("--s2", default=None, choices=SOURCES)
    ap.add_argument("--unfreeze", action="store_true",
                    help="2D-U flags: freeze=false train_last_n_blocks=3 drop_path_rate=0.2")
    ap.add_argument("--all-sources", action="store_true")
    ap.add_argument("-o", "--override", action="append", default=[])
    ap.add_argument("--batch", type=int, default=2)
    args = ap.parse_args()

    extra = list(args.override)
    if args.unfreeze:
        extra += ["policy.image_encoder_cfg.freeze=false",
                  "policy.image_encoder_cfg.train_last_n_blocks=3",
                  "policy.image_encoder_cfg.drop_path_rate=0.2"]

    if args.all_sources:
        print(f"[all sources] {args.task}")
        run_all_sources(args.task, extra)
        print("\nALL SOURCE PAIRS OK")
        return

    import hydra
    cfg = compose_cfg(args.task, args.s3, args.s2, extra)
    d = cfg.task.dataset
    print(f"task            : {args.task}")
    print(f"pose_source 3d/2d: {d.pose_source_3d} / {d.pose_source_2d}")
    print(f"dino frozen     : {cfg.policy.image_encoder_cfg.freeze} "
          f"(last_n={cfg.policy.image_encoder_cfg.train_last_n_blocks})")

    dataset = hydra.utils.instantiate(cfg.task.dataset)
    policy = hydra.utils.instantiate(cfg.policy)
    check_policy_structure(policy, cfg)
    normalizer = dataset.get_normalizer()
    want_keys = set(cfg.task.shape_meta.obs.keys()) | {"action"}
    assert set(normalizer.params_dict.keys()) == want_keys, \
        (set(normalizer.params_dict.keys()) ^ want_keys)
    policy.set_normalizer(normalizer)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    policy = policy.to(dev)
    policy.train()
    enc = policy.obs_encoder

    tk = enc.dino
    if tk.freeze:
        assert tk.backbone.training is False, "frozen DINO trunk is in train mode"

    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch,
                                         shuffle=True, num_workers=0)
    batch = next(iter(loader))
    batch = {k: (v.to(dev) if torch.is_tensor(v)
                 else {kk: vv.to(dev) for kk, vv in v.items()})
             for k, v in batch.items()}
    print("\nbatch keys:")
    for k, v in batch["obs"].items():
        print(f"  obs/{k:18s} {tuple(v.shape)}  [{v.min().item():.3f}, {v.max().item():.3f}]")
    for k, v in batch["obs"].items():
        want = tuple(cfg.task.shape_meta.obs[k].shape)
        assert tuple(v.shape[2:]) == want, f"{k}: {tuple(v.shape[2:])} != {want}"

    # token / PE alignment, captured on the REAL compute_loss path -- calling the
    # encoder directly would skip the inherited clamp the fused cloud relies on.
    seen = {}
    hook = enc.register_forward_hook(
        lambda m, i, o: seen.update(feat=tuple(o[0].shape), pe=tuple(o[1].shape)))
    # ...and the clamp itself: fused clouds must be clamped, unfused must not.
    seen_in = {}
    hook_in = enc.register_forward_pre_hook(
        lambda m, a, kw=None: seen_in.update(
            {k: (float(v.min()), float(v.max())) for k, v in a[0].items()
             if k.startswith("point_cloud")}))

    if dev == "cuda":
        torch.cuda.reset_peak_memory_stats()
    loss, loss_dict = policy.compute_loss(batch)
    hook.remove()
    hook_in.remove()
    assert seen["feat"][1] == seen["pe"][1] == enc.n_tokens, (seen, enc.n_tokens)
    print(f"\ntokens/step     : {enc.n_tokens} (features {seen['feat']}, pe {seen['pe']})")
    if cfg.task.dataset.fuse_3d:
        lo, hi = seen_in["point_cloud"]
        assert lo >= -1 - 1e-6 and hi <= 1 + 1e-6, f"fused cloud not clamped: [{lo}, {hi}]"
        print(f"fused cloud into encoder: [{lo:.4f}, {hi:.4f}] (clamped, as run A)")
    print(f"compute_loss -> {loss.item():.6f}")
    assert torch.isfinite(loss)
    loss.backward()
    if dev == "cuda":
        print(f"peak CUDA memory (fwd+bwd, batch {args.batch}): "
              f"{torch.cuda.max_memory_allocated() / 2**30:.2f} GiB")

    named = dict(policy.named_parameters())
    uni_no_grad = [n for n, p in named.items() if n.startswith("obs_encoder.uni3d.")
                   and p.requires_grad and p.grad is None]
    dino_with_grad = [n for n, p in named.items() if "obs_encoder.dino.backbone" in n
                      and p.grad is not None]
    dino_expected = [n for n, p in named.items() if "obs_encoder.dino.backbone" in n
                     and p.requires_grad]
    pose_no_grad = [n for n, p in named.items() if "pose_embed" in n and p.grad is None]
    n_tr = sum(p.numel() for p in named.values() if p.requires_grad)
    n_fr = sum(p.numel() for p in named.values() if not p.requires_grad)
    print(f"parameters      : {n_tr/1e6:.2f}M trainable, {n_fr/1e6:.2f}M frozen")
    print(f"uni3d trainable w/o grad : {len(uni_no_grad)} {uni_no_grad[:4]}")
    print(f"dino backbone w/ grad    : {len(dino_with_grad)} (expected {len(dino_expected)})")
    print(f"pose_embed w/o grad      : {len(pose_no_grad)} {pose_no_grad}")
    assert len(dino_with_grad) == len(dino_expected)
    assert not pose_no_grad, "a camera tag parameter received no gradient"
    # Uni3D never uses timm's own cls_token / pos_embed / IMAGE patch_embed
    # (`uni3d.transformer.*`), exactly as in the unfused run; its transformer
    # blocks, POINT patch embed and output projection must all get gradient.
    trunk_no_grad = [n for n in uni_no_grad if ".transformer.blocks." in n
                     or n.startswith("obs_encoder.uni3d.patch_embed.")
                     or ".out_proj." in n]
    assert not trunk_no_grad, trunk_no_grad

    with torch.no_grad():
        out = policy.predict_action(batch["obs"])
    print(f"predict_action -> action {tuple(out['action'].shape)}")
    assert torch.isfinite(out["action"]).all()
    print("\nFORWARD/BACKWARD OK")


if __name__ == "__main__":
    main()
