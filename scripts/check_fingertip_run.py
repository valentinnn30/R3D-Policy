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
[4b] anchor SETS (2026-09-24), synthetic: per-anchor radius honoured, the
    type tag moves only its own type's tokens, anchor-count guard.
[6] the six variant tasks real_bimanual_fingertip_{A,B,C,D,F,K}: Hydra compose,
    dataset anchors (shape + independent recompute), policy loss / backward /
    predict / attention mass, token layout; and the REFERENCE checkpoint
    (fingertip_150/300.ckpt) still loads strict into the reference policy.

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


def section4b():
    print("\n[4b] anchor-set encoder (synthetic)")
    from r3d.model.vision.pointnet_extractor import Uni3DPointcloudEncoder
    common = dict(pc_model="eva02_tiny_patch14_224", pc_feat_dim=192, embed_dim=256,
                  group_size=32, num_group=64, patch_dropout=0, drop_path_rate=0.0,
                  pretrained_pc=None, pc_encoder_dim=512, normalization_type="layer_norm",
                  feature_mode="pointsam", use_pretrained_weights=False, point_channels=9,
                  fps_random_config={"use_random": False, "random_start": False,
                                     "random_noise_scale": 0, "shuffle_output": False})
    spec = [{"type": "pad_mid", "radius_m": 0.015}, {"type": "tip", "radius_m": 0.03},
            {"type": "pinch", "fingers": ["index"], "radius_m": 0.03}]
    torch.manual_seed(0)
    ef = Uni3DPointcloudEncoder(fingertip_tokens={"enabled": True, "radius_m": 0.03,
                                                  "tag_at": ["vit", "head"],
                                                  "anchors": spec}, **common).eval()
    hr = torch.tensor([0.325, 0.5, 0.3])
    ef.set_xyz_half_range(hr)
    T = ef.n_tips
    check("anchor count 10 + 10 + 2 = 22", T == 22)
    check("type tags exist and are zero-init",
          ef.ft_type_vit.shape[0] == len(rp.ANCHOR_TYPES)
          and torch.all(ef.ft_type_vit == 0).item() and torch.all(ef.ft_type_head == 0).item())
    lay = rp.anchor_layout(rp.normalize_anchor_spec(spec))
    check("encoder per-anchor ids == anchor_layout",
          ef.ft_hand_idx.tolist() == [h for h, *_ in lay]
          and ef.ft_finger_idx.tolist() == [f for _, f, *_ in lay]
          and ef.ft_type_idx.tolist() == [t for _, _, t, _ in lay])
    g = torch.Generator().manual_seed(5)
    B, N = 1, 4096
    pts = torch.rand(B, N, 3, generator=g) * 2 - 1
    anchors = (torch.rand(B, T, 3, generator=g) - 0.5)
    # a shell of points at 2 cm (metric) around every anchor: inside a 3 cm
    # radius, outside a 1.5 cm one
    d = torch.randn(T, 20, 3, generator=g)
    d = d / d.norm(dim=-1, keepdim=True) * 0.02
    pts[0, :T * 20] = (anchors[0, :, None, :] + d / hr).reshape(-1, 3)
    near = [(((pts[0] - anchors[0, t]) * hr).norm(dim=-1) < 0.015).any().item() for t in range(T)]
    feats = torch.rand(B, N, 6, generator=g)
    with torch.no_grad():
        _, _, empty = ef._fingertip_patches(pts, feats, anchors)
    r = ef.ft_radius
    want = [(r[t] < 0.02).item() and not near[t] for t in range(T)]
    check("per-anchor radius: 1.5 cm anchors empty, 3 cm anchors hit the 2 cm shell",
          empty[0].tolist() == want, f"{sum(want)} of {T} expected empty")
    with torch.no_grad():
        x, pe = ef(torch.cat([pts, feats], -1), eval=True, anchors=anchors)
        pinch = rp.ANCHOR_TYPES.index("pinch")
        ef.ft_type_head[pinch].fill_(0.5)
        _, pe2 = ef(torch.cat([pts, feats], -1), eval=True, anchors=anchors)
        ef.ft_type_head.zero_()
    moved = [not torch.equal(pe2[0, 64 + t], pe[0, 64 + t]) for t in range(T)]
    check("pinch type tag moves only the pinch tokens",
          moved == [ti == pinch for ti in ef.ft_type_idx.tolist()]
          and torch.equal(pe2[:, :64], pe[:, :64]))
    try:
        ef(torch.cat([pts, feats], -1), eval=True, anchors=anchors[:, :10])
        check("anchor-count guard: 10 anchors into a 22-anchor encoder raises", False)
    except ValueError:
        check("anchor-count guard: 10 anchors into a 22-anchor encoder raises", True)

    # --- sampling: fps (2026-09-24) ------------------------------------
    sys.path.insert(0, str(REPO / "scripts"))
    from export_fingertip_replay import pick_patch
    torch.manual_seed(0)
    efn = Uni3DPointcloudEncoder(fingertip_tokens={"enabled": True, "radius_m": 0.03,
                                                   "tag_at": ["vit", "head"],
                                                   "anchors": [{"type": "tip"}]},
                                 **common).eval()
    torch.manual_seed(0)
    eff = Uni3DPointcloudEncoder(fingertip_tokens={"enabled": True, "radius_m": 0.03,
                                                   "tag_at": ["vit", "head"],
                                                   "anchors": [{"type": "tip"}],
                                                   "sampling": "fps"}, **common).eval()
    for e in (efn, eff):
        e.set_xyz_half_range(hr)
    g = torch.Generator().manual_seed(9)
    P = torch.rand(1, 4096, 3, generator=g) * 2 - 1
    A = (torch.rand(1, 10, 3, generator=g) - 0.5)
    # dense core (0.5 cm) + sparse shell (2-2.9 cm) around each anchor, and
    # anchor 9 with only 5 in-radius points
    for t in range(10):
        n_core, n_shell = (150, 60) if t < 9 else (5, 0)
        core = torch.randn(n_core, 3, generator=g) * 0.005
        sh = torch.randn(n_shell, 3, generator=g)
        sh = sh / sh.norm(dim=-1, keepdim=True) * (0.02 + 0.009 * torch.rand(n_shell, 1, generator=g))
        blob = A[0, t] + torch.cat([core, sh]) / hr
        P[0, t * 210:t * 210 + len(blob)] = blob
    F6 = torch.rand(1, 4096, 6, generator=g)
    hrP, hrA = (P * hr)[0].numpy(), (A * hr)[0].numpy()

    with torch.no_grad():
        embn, _, _ = efn._fingertip_patches(P, F6, A)
        embf, _, emf = eff._fingertip_patches(P, F6, A)
    check("fps: embeddings differ from nearest (a different draw)",
          not torch.allclose(embn, embf))
    # independent numpy draw -> encode -> must equal the encoder's fps embedding
    ok_same, ok_rad, spread = True, True, []
    for t in range(10):
        d = np.linalg.norm(hrP - hrA[t], axis=1)
        sel_f = pick_patch(d, 0.03, "fps", hrP)
        sel_n = pick_patch(d, 0.03, "nearest", hrP)
        ok_rad &= bool(np.all(d[sel_f] <= 0.03))
        spread.append((d[sel_f].mean(), d[sel_n].mean()))
        sel = torch.as_tensor(sel_f)
        reps = torch.arange(32) % len(sel)
        rel = P[0, sel[reps]] - A[0, t].clamp(-1, 1)
        one = eff.patch_embed.patch_encoder(torch.cat([rel, F6[0, sel[reps]]], -1)[None, None])
        ok_same &= torch.allclose(one[0, 0], embf[0, t], atol=1e-5)
    check("fps: encoder draw == independent numpy draw (replay) on all 10 patches", ok_same)
    check("fps: no point beyond the radius", ok_rad)
    sf, sn = np.mean([a for a, _ in spread]), np.mean([b for _, b in spread])
    check("fps spreads over the ball: mean distance to centre > nearest's",
          sf > 1.5 * sn, f"fps {sf * 100:.2f} cm vs nearest {sn * 100:.2f} cm")
    check("fps: short patch (5 in radius) not empty, refilled", not bool(emf[0, 9]))

    # fps_random_start: random only when training (eval=False)
    torch.manual_seed(0)
    efr = Uni3DPointcloudEncoder(fingertip_tokens={"enabled": True, "radius_m": 0.03,
                                                   "tag_at": ["vit", "head"],
                                                   "anchors": [{"type": "tip"}],
                                                   "sampling": "fps",
                                                   "fps_random_start": True}, **common).eval()
    efr.load_state_dict(eff.state_dict())
    efr.set_xyz_half_range(hr)
    with torch.no_grad():
        ev = [efr._fingertip_patches(P, F6, A, eval=True)[0] for _ in range(3)]
        tr = [efr._fingertip_patches(P, F6, A, eval=False)[0] for _ in range(3)]
    check("fps_random_start: eval draw deterministic and == the fixed-start draw",
          all(torch.equal(e, embf) for e in ev))
    check("fps_random_start: training draws differ from step to step",
          not torch.allclose(tr[0], tr[1]) and not torch.allclose(tr[1], tr[2]))
    check("fps_random_start: short patch still not empty in training",
          not bool(efr._fingertip_patches(P, F6, A, eval=False)[2][0, 9]))
    try:
        Uni3DPointcloudEncoder(fingertip_tokens={"enabled": True, "tag_at": ["vit"],
                                                 "fps_random_start": True}, **common)
        check("fps_random_start without sampling fps raises", False)
    except ValueError:
        check("fps_random_start without sampling fps raises", True)


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


# ----------------------------------------------------------------------- [6]
VARIANTS = {"A": 10, "B": 30, "C": 30, "D": 16, "F": 56, "K": 0}


def section6(ds_ref, batch_size):
    print("\n[6] variant tasks A B C D F K + the reference checkpoint")
    import hydra
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf
    import yaml
    import zarr
    dev = "cuda"
    with initialize_config_dir(config_dir=str(REPO / "R3D/r3d/config"), version_base=None):
        ref_cfg = compose(config_name="r3d_robotwin2",
                          overrides=["task=real_bimanual_fingertip",
                                     "task_name=real_bimanual_fingertip"])
        cfgs = {k: compose(config_name="r3d_robotwin2",
                           overrides=[f"task=real_bimanual_fingertip_{k}",
                                      f"task_name=real_bimanual_fingertip_{k}"])
                for k in VARIANTS}

    # reference checkpoint loads strict into the (refactored) reference policy
    ckpt = REPO / "R3D/data/outputs/fingertip_150/300.ckpt"
    if ckpt.exists():
        import dill
        payload = torch.load(ckpt.open("rb"), pickle_module=dill, map_location="cpu",
                             weights_only=False)
        pol = hydra.utils.instantiate(ref_cfg.policy)
        try:
            pol.load_state_dict(payload["state_dicts"]["model"], strict=True)
            check("reference checkpoint 300.ckpt loads strict", True)
        except RuntimeError as e:
            check("reference checkpoint 300.ckpt loads strict", False, str(e)[:300])
        del pol, payload
    else:
        check("reference checkpoint present", False, str(ckpt))

    root = zarr.open(ref_cfg.task.dataset.zarr_path, mode="r")
    raw = yaml.safe_load(open(ref_cfg.task.dataset.preprocess_config))
    he = {s: np.asarray(raw["cameras"][i]["extrinsics"]) for s, i in (("left", 1), ("right", 2))}
    n_obs = ref_cfg.n_obs_steps
    for k, n_anchor in VARIANTS.items():
        cfg = cfgs[k]
        dcfg = cfg.task.dataset
        onehot = bool(dcfg.camera_onehot)
        ft = dcfg.get("fingertips")
        # the variant dataset = the reference one with its flags swapped (one buffer)
        ds = copy.copy(ds_ref)
        ds.camera_onehot = onehot
        if ft is None:
            ds.hand_fk, ds.fingertips = None, None
        else:
            spec = ft.get("anchors")
            ds.anchor_kwargs = {} if spec is None else {
                "spec": OmegaConf.to_container(spec, resolve=True),
                "pad_offset_m": float(ft.pad_offset_m)}
        n_ch = 9 if onehot else 6
        meta = cfg.task.shape_meta.obs
        np.random.seed(7)
        smp = ds[len(ds) // 2]
        pc = smp["obs"]["point_cloud"]
        ok = tuple(pc.shape) == (n_obs, 8192, n_ch) and tuple(meta.point_cloud.shape) == (8192, n_ch)
        if n_anchor:
            an = smp["obs"]["fingertip_anchors"]
            ok = ok and tuple(an.shape) == (n_obs, n_anchor, 3) \
                and tuple(meta.fingertip_anchors.shape) == (n_anchor, 3)
        else:
            ok = ok and "fingertip_anchors" not in smp["obs"] and "fingertip_anchors" not in meta
        check(f"{k}: dataset + shape_meta shapes (cloud {n_ch} ch, {n_anchor} anchors)", ok)
        if n_anchor and ds.anchor_kwargs:
            idx = len(ds) // 2
            t0 = int(ds.sampler.indices[idx][0])
            got = ds._fingertip_anchors(ds.sampler.sample_sequence(idx), 0)
            link8 = {s: np.asarray(root[f"data/extrinsics_cam{i}"][t0]) @ np.linalg.inv(he[s])
                     for s, i in (("left", 1), ("right", 2))}
            want = rp.HandFK(ft.fk_config).anchors(
                link8, np.asarray(root["data/state"][t0]), **ds.anchor_kwargs)
            check(f"{k}: anchors == independent recompute from the zarr",
                  np.allclose(got, want, atol=1e-9))

        policy = hydra.utils.instantiate(cfg.policy)
        policy.set_normalizer(ds.get_normalizer())
        policy = policy.to(dev)
        enc = policy.obs_encoder.extractor
        want_g = int(cfg.task.get("num_group", 512))
        check(f"{k}: encoder {n_ch} ch, num_group {enc.num_group}, anchors {getattr(enc, 'n_tips', 0)}",
              enc.point_channels == n_ch and enc.num_group == want_g
              and (enc.fingertip_cfg is not None) == bool(n_anchor)
              and (not n_anchor or enc.n_tips == n_anchor))
        loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0)
        batch = next(iter(loader))
        batch = {"obs": {kk: v.to(dev) for kk, v in batch["obs"].items()},
                 "action": batch["action"].to(dev)}
        policy.train()
        loss, _ = policy.compute_loss(batch)
        loss.backward()
        grads = [n for n, p in policy.named_parameters() if "ft_" in n and p.grad is None]
        check(f"{k}: loss finite ({loss.item():.4f}), grad on every ft_ param",
              torch.isfinite(loss).item() and not grads, f"no grad: {grads}" if grads else "")
        policy.eval()
        with torch.no_grad():
            out = policy.predict_action(batch["obs"])
        check(f"{k}: predict_action finite", torch.isfinite(out["action"]).all().item())
        if n_anchor:
            ftm = policy.fingertip_attention_mass(batch["obs"])
            idx = policy.fingertip_key_indices()
            per = 512 + n_anchor
            check(f"{k}: attention mass {tuple(ftm['mass'].shape)}, keys at the end of each block",
                  tuple(ftm["mass"].shape) == (batch_size, 4, n_anchor)
                  and idx[0] == 512 and idx[n_anchor] == per + 512 and len(idx) == n_obs * n_anchor
                  and abs(ftm["uniform"] - n_obs / (n_obs * per)) < 1e-9)
            if k != "A":
                check(f"{k}: per-type ids present", "type_idx" in ftm)
        del policy
        torch.cuda.empty_cache()


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
    section4b()
    if not args.skip_dataset:
        cfg, ds = sections23(args)
        section5(cfg, ds, args.batch)
        section6(ds, args.batch)
    print(f"\n{'ALL PASS' if not fails else f'{len(fails)} FAILED: {fails}'}")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
