#!/usr/bin/env python3
"""Verify the fingertip-token + camera-one-hot run end to end.

    micromamba activate r3d && unset PYTHONPATH
    cd ~/R3D-Policy
    python scripts/check_fingertip_run.py            # all sections
    python scripts/check_fingertip_run.py --skip-dataset   # encoder/forward only

[1] fuse_cameras with the new arguments OFF is bit-identical to git HEAD
    (random and fps), and with the one-hot ON carries a valid one-hot.
[2] dataset with both flags OFF: __getitem__ bit-identical to git HEAD's
    dataset code on the same replay buffer, same seeds.
[3] dataset with the flags ON: shapes, one-hot validity, augmentation leaves
    the one-hot alone, anchors equal an independent recompute, normalizer
    widths / identity / shared xyz map.
[4] encoder unit tests on synthetic clouds: no point beyond the metric radius,
    empty and out-of-box anchors, duplicate filling is max-pool neutral, the
    9-channel encoder equals the 6-channel one at init (zero columns), the
    pretrained load widens conv1, zero tags are a no-op.
[5] full Hydra compose of task=real_bimanual_fingertip: compute_loss +
    backward, predict_action, attention mass; and the width guard fires.

One ReplayBuffer is ~12 GiB, so exactly ONE dataset is constructed; the
flags-off variant is a copy.copy sharing it.
"""
import argparse
import copy
import os
import pathlib
import subprocess
import sys
import types

import numpy as np
import torch

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "R3D"))
os.chdir(REPO)

from r3d.common import real_preprocess as rp  # noqa: E402

fails = []


def check(name, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}{'  ' + detail if detail else ''}", flush=True)
    if not ok:
        fails.append(name)


def head_module(relpath, name, patch=None):
    """Load a module's source as committed at git HEAD."""
    src = subprocess.check_output(["git", "show", f"HEAD:{relpath}"], cwd=REPO, text=True)
    mod = types.ModuleType(name)
    mod.__file__ = f"<HEAD>/{relpath}"
    exec(compile(src, mod.__file__, "exec"), mod.__dict__)
    if patch:
        mod.__dict__.update(patch)
    return mod


# ----------------------------------------------------------------------- [1]
def section1(root, cfg):
    print("\n[1] fuse_cameras: OFF == git HEAD, ON carries a valid one-hot")
    head_rp = head_module("R3D/r3d/common/real_preprocess.py", "head_rp")
    E0 = np.asarray(root["meta/extrinsics_cam0"][:], dtype=np.float64)
    for t in (0, 20000, 50000):
        clouds = [np.asarray(root[f"data/point_cloud_cam{i}"][t]) for i in range(3)]
        Ts = [E0] + [np.asarray(root[f"data/extrinsics_cam{i}"][t], dtype=np.float64)
                     for i in (1, 2)]
        for method in ("random", "fps"):
            a = head_rp.fuse_cameras(clouds, Ts, cfg, method=method,
                                     rng=np.random.default_rng(3), device="cuda")
            b = rp.fuse_cameras(clouds, Ts, cfg, method=method,
                                rng=np.random.default_rng(3), device="cuda")
            check(f"t={t} {method}: bit-identical to HEAD", a.dtype == b.dtype
                  and a.shape == b.shape and np.array_equal(a, b))
        c = rp.fuse_cameras(clouds, Ts, cfg, method="random", rng=np.random.default_rng(3),
                            camera_ids=[0, 1, 2], n_cameras=3)
        oh = c[:, 6:]
        check(f"t={t} one-hot ON: shape {c.shape}, rows one-hot",
              c.shape == (cfg.num_points, 9) and np.all(oh.sum(1) == 1)
              and set(np.unique(oh)) <= {0.0, 1.0})
        check(f"t={t} one-hot ON: first 6 columns == OFF draw (same rng)",
              np.array_equal(c[:, :6], rp.fuse_cameras(
                  clouds, Ts, cfg, method="random", rng=np.random.default_rng(3))))
        # each point's id must name the camera that actually produced it
        own = [clouds[i][:, :3] @ Ts[i][:3, :3].T + Ts[i][:3, 3] for i in range(3)]
        sample = np.random.default_rng(0).choice(len(c), 300, replace=False)
        ok = 0
        for j in sample:
            d = [np.min(np.linalg.norm(o - c[j, :3], axis=1)) for o in own]
            ok += int(np.argmin(d) == np.argmax(oh[j]) and min(d) < 1e-4)
        check(f"t={t} one-hot names the true source camera", ok == len(sample),
              f"{ok}/{len(sample)} points")


# ----------------------------------------------------------------------- [2,3]
def sections23(args):
    import hydra
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf
    if not OmegaConf.has_resolver("eval"):
        OmegaConf.register_new_resolver("eval", eval, replace=True)
    with initialize_config_dir(config_dir=str(REPO / "R3D/r3d/config"), version_base=None):
        cfg = compose(config_name="r3d_robotwin2",
                      overrides=["task=real_bimanual_fingertip",
                                 "task_name=real_bimanual_fingertip"])
    ds_on = hydra.utils.instantiate(cfg.task.dataset)

    # flags-off twin sharing the same replay buffer
    ds_off = copy.copy(ds_on)
    ds_off.camera_onehot, ds_off.hand_fk, ds_off.fingertips = False, None, None

    print("\n[2] dataset flags OFF == git HEAD dataset code (same buffer, same seeds)")
    head_rp = head_module("R3D/r3d/common/real_preprocess.py", "head_rp2")
    head_ds = head_module("R3D/r3d/dataset/real_multicam_dataset.py", "head_ds",
                          patch={"fuse_cameras": head_rp.fuse_cameras,
                                 "workspace_limits": head_rp.workspace_limits})
    H = head_ds.RealMultiCamDataset.__new__(head_ds.RealMultiCamDataset)
    H.__dict__.update(ds_off.__dict__)
    for idx in (0, len(ds_on) // 2, len(ds_on) - 1):
        np.random.seed(idx + 11)
        a = H[idx]
        np.random.seed(idx + 11)
        b = ds_off[idx]
        same = (a.keys() == b.keys() and a["obs"].keys() == b["obs"].keys()
                and all(torch.equal(a["obs"][k], b["obs"][k]) for k in a["obs"])
                and torch.equal(a["action"], b["action"]))
        check(f"idx {idx}: OFF sample bit-identical to HEAD", same)
    nh, no = H.get_normalizer(), ds_off.get_normalizer()
    check("OFF normalizer identical to HEAD",
          all(torch.equal(nh[k].params_dict["scale"], no[k].params_dict["scale"])
              and torch.equal(nh[k].params_dict["offset"], no[k].params_dict["offset"])
              for k in ("point_cloud", "agent_pos", "action")))

    print("\n[3] dataset flags ON")
    n_obs = cfg.n_obs_steps
    for idx in (0, len(ds_on) // 3, len(ds_on) - 1):
        np.random.seed(idx)
        s = ds_on[idx]
        pc, an = s["obs"]["point_cloud"], s["obs"]["fingertip_anchors"]
        oh = pc[..., 6:]
        check(f"idx {idx}: shapes pc {tuple(pc.shape)} anchors {tuple(an.shape)}",
              tuple(pc.shape) == (n_obs, 8192, 9) and tuple(an.shape) == (n_obs, 10, 3))
        check(f"idx {idx}: one-hot exact after augmentation (noise + jitter)",
              torch.all(oh.sum(-1) == 1).item() and torch.all((oh == 0) | (oh == 1)).item())
    # anchors vs an independent recompute straight from the zarr
    import yaml
    import zarr
    root = zarr.open(cfg.task.dataset.zarr_path, mode="r")
    raw = yaml.safe_load(open(cfg.task.dataset.preprocess_config))
    fk = rp.HandFK(cfg.task.dataset.fingertips.fk_config)
    he = {s: np.asarray(raw["cameras"][i]["extrinsics"]) for s, i in (("left", 1), ("right", 2))}
    idx = len(ds_on) // 2
    seq = ds_on.sampler.indices[idx]
    t0 = int(seq[0])  # buffer_start_idx; first obs frame when no pad_before
    sample = ds_on.sampler.sample_sequence(idx)
    got = ds_on._fingertip_anchors(sample, 0)
    link8 = {s: np.asarray(root[f"data/extrinsics_cam{i}"][t0]) @ np.linalg.inv(he[s])
             for s, i in (("left", 1), ("right", 2))}
    want = fk.anchors(link8, np.asarray(root["data/state"][t0]))
    check("anchors == independent recompute from the zarr", np.allclose(got, want, atol=1e-9),
          f"max {np.abs(got - want).max():.2e} m")
    norm = ds_on.get_normalizer()
    sc, of = norm["point_cloud"].params_dict["scale"], norm["point_cloud"].params_dict["offset"]
    check("point_cloud normalizer 9 wide, one-hot identity",
          sc.shape == (9,) and torch.all(sc[6:] == 1).item() and torch.all(of[6:] == 0).item())
    fa = norm["fingertip_anchors"].params_dict
    check("fingertip_anchors normalizer == point_cloud xyz map",
          torch.equal(fa["scale"], sc[:3]) and torch.equal(fa["offset"], of[:3]))
    return cfg, ds_on


# ----------------------------------------------------------------------- [4]
def section4():
    print("\n[4] encoder unit tests (synthetic)")
    from r3d.model.vision.pointnet_extractor import Uni3DPointcloudEncoder
    common = dict(pc_model="eva02_tiny_patch14_224", pc_feat_dim=192, embed_dim=256,
                  group_size=32, num_group=64, patch_dropout=0, drop_path_rate=0.0,
                  pretrained_pc=None, pc_encoder_dim=512, normalization_type="layer_norm",
                  feature_mode="pointsam",
                  fps_random_config={"use_random": False, "random_start": False,
                                     "random_noise_scale": 0, "shuffle_output": False})
    torch.manual_seed(0)
    e6 = Uni3DPointcloudEncoder(use_pretrained_weights=True,
                                pretrained_weights_path="R3D/pretrain_weight", **common).eval()
    torch.manual_seed(0)
    e9 = Uni3DPointcloudEncoder(use_pretrained_weights=True, point_channels=9,
                                pretrained_weights_path="R3D/pretrain_weight", **common).eval()
    w6 = e6.patch_embed.patch_encoder.conv1[0].weight
    w9 = e9.patch_embed.patch_encoder.conv1[0].weight
    check("pretrained conv1 widened: first 6 columns == pretrained, rest zero",
          torch.equal(w9[:, :6], w6) and torch.all(w9[:, 6:] == 0).item())
    e9.load_state_dict({k: v for k, v in e6.state_dict().items()
                        if k != Uni3DPointcloudEncoder._CONV1_KEY}, strict=False)
    g = torch.Generator().manual_seed(1)
    pc6 = torch.rand(2, 2048, 6, generator=g) * 2 - 1
    oh = torch.nn.functional.one_hot(torch.randint(0, 3, (2, 2048), generator=g), 3).float()
    pc9 = torch.cat([pc6, oh], -1)
    with torch.no_grad():
        x6, p6 = e6(pc6, eval=True)
        x9, p9 = e9(pc9, eval=True)
    check("9-channel encoder == 6-channel at init (zero columns)",
          torch.allclose(x6, x9, atol=1e-6) and torch.equal(p6, p9),
          f"max {(x6 - x9).abs().max():.2e}")
    try:
        e9(pc6, eval=True)
        check("width contract: 6-channel cloud into a 9-channel encoder raises", False)
    except ValueError:
        check("width contract: 6-channel cloud into a 9-channel encoder raises", True)

    torch.manual_seed(0)
    ef = Uni3DPointcloudEncoder(use_pretrained_weights=False, point_channels=9,
                                fingertip_tokens={"enabled": True, "radius_m": 0.03,
                                                  "tag_at": ["vit", "head"]}, **common).eval()
    hr = torch.tensor([0.325, 0.5, 0.3])  # the real workspace half-ranges, m per unit
    try:
        ef(pc9, eval=True, anchors=torch.zeros(2, 10, 3))
        check("unset xyz_half_range raises", False)
    except RuntimeError:
        check("unset xyz_half_range raises", True)
    ef.set_xyz_half_range(hr)

    # a cloud with known geometry: a dense blob around anchor 0, nothing near 1
    B, N = 1, 4096
    pts = (torch.rand(B, N, 3, generator=g) * 2 - 1)
    anchors = torch.zeros(B, 10, 3)
    anchors[0, 0] = torch.tensor([0.2, 0.1, -0.1])
    blob = anchors[0, 0] + (torch.rand(50, 3, generator=g) - 0.5) * 0.02 / hr  # +-1 cm metric
    pts[0, :50] = blob
    far = torch.tensor([0.9, 0.9, 0.9])
    pts = torch.where(((pts - far).abs() * hr).norm(dim=-1, keepdim=True) < 0.08,
                      torch.full_like(pts, -0.9), pts)   # carve a hole around `far`
    anchors[0, 1] = far
    anchors[0, 2] = torch.tensor([1.5, 0.0, 0.0])       # outside the workspace box
    # a SPARSE patch: carve a hole, then put exactly 8 points within 1 cm, so
    # the fill path (24 duplicated slots) is really exercised
    sparse = torch.tensor([-0.5, 0.5, 0.3])
    pts = torch.where(((pts - sparse).abs() * hr).norm(dim=-1, keepdim=True) < 0.08,
                      torch.full_like(pts, 0.95), pts)
    pts[0, 60:68] = sparse + (torch.rand(8, 3, generator=g) - 0.5) * 0.02 / hr
    anchors[0, 3] = sparse
    for k in range(4, 10):
        anchors[0, k] = pts[0, 100 + k]
    feats = torch.cat([torch.rand(B, N, 3, generator=g),
                       torch.nn.functional.one_hot(torch.randint(0, 3, (B, N), generator=g), 3).float()], -1)
    with torch.no_grad():
        emb, centers, empty = ef._fingertip_patches(pts, feats, anchors)
        dist, idx = rp_knn(anchors * hr, pts * hr)
    check("anchor with no point in radius -> empty", bool(empty[0, 1]))
    check("anchor outside the box -> empty, centre clamped",
          bool(empty[0, 2]) and torch.all(centers[0, 2].abs() <= 1).item())
    check("anchor on a dense blob -> not empty", not bool(empty[0, 0]))
    # recompute the chosen neighbours and check the radius on EVERY slot
    K = ef.group_size
    d, i = rp_knn(anchors * hr, pts * hr, K)
    valid = d <= 0.03
    nv = valid.sum(-1)
    fill = torch.arange(K).expand_as(i) % nv.clamp(min=1).unsqueeze(-1)
    chosen = torch.gather(i, 2, fill)
    chosen_d = torch.gather(d, 2, fill)
    ok = all(bool(torch.all(chosen_d[0, t] <= 0.03)) for t in range(10) if nv[0, t] > 0)
    check("no point beyond 3 cm (metric) in any non-empty fingertip patch", ok,
          f"max {max(float(chosen_d[0, t].max()) for t in range(10) if nv[0, t] > 0) * 100:.2f} cm")
    check("sparse patch has exactly 8 in-radius points (fill path active)", int(nv[0, 3]) == 8)
    # duplicate filling is max-pool neutral: encode only the valid prefix
    t = 3
    n = int(nv[0, t])
    sel = i[0, t, :n]
    rel = pts[0, sel] - centers[0, t]
    single = ef.patch_embed.patch_encoder(torch.cat([rel, feats[0, sel]], -1)[None, None])
    check("duplicate-filled patch == its unique in-radius points (max-pool)",
          torch.allclose(single[0, 0], emb[0, t], atol=1e-5), f"n_valid={n}")
    with torch.no_grad():
        x, pe = ef(torch.cat([pts, feats], -1), eval=True, anchors=anchors)
    check("token count 64 FPS + 10 fingertip per frame", x.shape[1] == 74 and pe.shape[1] == 74)
    # zero tags are a no-op; a non-zero head tag moves only the fingertip PEs
    with torch.no_grad():
        ef.ft_tip_head.fill_(0.5)
        x2, pe2 = ef(torch.cat([pts, feats], -1), eval=True, anchors=anchors)
        ef.ft_tip_head.zero_()
    check("head tag touches only the 10 fingertip PEs",
          torch.equal(pe2[:, :64], pe[:, :64]) and not torch.equal(pe2[:, 64:], pe[:, 64:])
          and torch.equal(x2, x))


def rp_knn(q, k, K=32):
    from r3d.model.vision.pointnet_extractor import knn_points
    return knn_points(q, k, K, sorted=True)


# ----------------------------------------------------------------------- [5]
def section5(cfg, dataset, batch_size):
    print("\n[5] full policy through Hydra: loss, backward, predict, attention mass")
    import hydra
    policy = hydra.utils.instantiate(cfg.policy)
    policy.set_normalizer(dataset.get_normalizer())
    dev = "cuda"
    policy = policy.to(dev)
    enc = policy.obs_encoder.extractor
    check("encoder built for 9 channels with fingertip tokens",
          enc.point_channels == 9 and enc.fingertip_cfg is not None)
    check("xyz_half_range filled from the normalizer (m per unit)",
          bool(enc.xyz_half_range_set) and torch.allclose(
              enc.xyz_half_range.cpu(), torch.tensor([0.325, 0.5, 0.3]), atol=1e-6),
          f"{enc.xyz_half_range.cpu().numpy().round(4)}")
    check("fingertip_anchors NOT a low-dim MLP input",
          "fingertip_anchors" not in policy.obs_encoder.low_dim_keys)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True,
                                         num_workers=0)
    batch = next(iter(loader))
    batch = {"obs": {k: v.to(dev) for k, v in batch["obs"].items()},
             "action": batch["action"].to(dev)}
    policy.train()
    loss, _ = policy.compute_loss(batch)
    loss.backward()
    grads = {n: p.grad for n, p in policy.named_parameters() if "ft_" in n or "conv1.0" in n}
    check(f"loss finite ({loss.item():.4f})", torch.isfinite(loss).item())
    check("gradient reaches every fingertip tag + empty token + conv1",
          all(g is not None for g in grads.values()), f"{len(grads)} params")
    policy.eval()
    with torch.no_grad():
        out = policy.predict_action(batch["obs"])
    check(f"predict_action {tuple(out['action'].shape)}", torch.isfinite(out["action"]).all().item())
    ft = policy.fingertip_attention_mass(batch["obs"])
    pooled = ft["mass"].sum(-1).mean(0)
    check("attention mass: one value per head layer, per tip",
          tuple(ft["mass"].shape) == (batch_size, 4, 10),
          f"pooled per layer {pooled.numpy().round(4)} vs uniform {ft['uniform'] * 10:.4f}")
    idx = policy.fingertip_key_indices()
    check("fingertip key indices sit at the end of each obs-step block",
          idx[:3] == [512, 513, 514] and idx[10] == 522 + 512)
    bad = dict(batch["obs"])
    bad["point_cloud"] = bad["point_cloud"][..., :6]
    try:
        policy.predict_action(bad)
        check("normalizer width guard: 6-channel obs into a 9-channel policy raises", False)
    except ValueError:
        check("normalizer width guard: 6-channel obs into a 9-channel policy raises", True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-dataset", action="store_true")
    ap.add_argument("--batch", type=int, default=2)
    args = ap.parse_args()
    import zarr
    cfg_pp = rp.PointCloudPreprocessConfig.from_yaml("R3D/r3d/config/preprocess/real_bimanual.yaml")
    root = zarr.open("R3D/data/real_bimanual_mixed150.zarr", mode="r")
    section1(root, cfg_pp)
    section4()
    if not args.skip_dataset:
        cfg, ds = sections23(args)
        section5(cfg, ds, args.batch)
    print(f"\n{'ALL PASS' if not fails else f'{len(fails)} FAILED: {fails}'}")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
