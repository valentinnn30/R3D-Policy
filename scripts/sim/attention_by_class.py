#!/usr/bin/env python
"""Where does the action head look? Cross-attention mass per point class.

For each checkpoint: run predict_action on stored frames (n_obs_steps windows),
read the action head's cross-attention (queries = action tokens, keys = point
tokens) of the LAST denoising step in every OneWayTransformer layer, average
over heads and queries, and sum the attention per token class. A token's class
is the class of its FPS centre point (point_labels.py). Reported against the
class's share of tokens, so ratio > 1 = over-attended.

Also: attention on the tokens whose centre lies within 3 cm of the mug handle /
rack hook (critical_points recorded by check_head_view.py).

    python scripts/sim/attention_by_class.py <ckpt> [<ckpt> ...] [--frames 300]

Uses the 8192-point replay of the hanging_mug screen (sim_view_dense: clouds,
labels, joint states AND critical points); each policy sees the first
num_points points (= its FPS cloud) and, if it was trained with a cut, the same
oracle cut. Mode: inference only, GPU-light, safe to run next to training.
"""
import argparse
import copy
import glob
import json
import os
import sys

import h5py
import numpy as np
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "R3D"))
DENSE = os.path.join(ROOT, "R3D", "data", "outputs", "sim_view_dense", "hanging_mug_labeled")
CLASSES = {4: "left_arm", 5: "right_arm", 6: "left_gripper", 7: "right_gripper", 8: "mug", 9: "rack"}


def load_frames(n_frames, seed=0):
    files = sorted(glob.glob(f"{DENSE}/data/episode*.hdf5"), key=lambda p: int(p.split("episode")[1][:-5]))
    names = json.load(open(f"{DENSE}/critical_names.json"))
    items = []
    for f in files:
        h = h5py.File(f)
        pc, lab, vec, crit = (h["pointcloud"][()], h["pointcloud_label"][()],
                              h["joint_action/vector"][()], h["view_stats/crit_xyz"][()])
        for t in range(1, len(pc)):
            items.append((pc[t - 1:t + 1], lab[t - 1:t + 1], vec[t - 1:t + 1], crit[t]))
    rng = np.random.default_rng(seed)
    pick = rng.choice(len(items), size=min(n_frames, len(items)), replace=False)
    return [items[i] for i in sorted(pick)], names


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ckpts", nargs="+")
    ap.add_argument("--frames", type=int, default=300)
    ap.add_argument("--batch", type=int, default=16)
    a = ap.parse_args()

    import dill
    import hydra
    from omegaconf import OmegaConf
    from r3d.common.label_cut import cut_points_by_label
    from r3d.model.vision.pointnet_extractor import KNNGrouper
    OmegaConf.register_new_resolver("eval", eval, replace=True)
    os.chdir(ROOT)

    frames, crit_names = load_frames(a.frames)
    stash = {}
    orig_fwd = KNNGrouper.forward

    def spy(self, xyz, features, *args, **kw):
        out = orig_fwd(self, xyz, features, *args, **kw)
        # index of each centre in this grouper call's input (exact coordinates)
        d = torch.cdist(out["centers"].float(), xyz.float())
        stash["centre_idx"] = d.argmin(-1)              # [B*To, G]
        return out
    KNNGrouper.forward = spy

    results = {}
    for ck in a.ckpts:
        p = torch.load(open(ck, "rb"), pickle_module=dill, map_location="cpu", weights_only=False)
        cfg = p["cfg"]
        N = int(cfg.task.get("num_points", 1024) or 1024)
        drop = cfg.task.get("drop_point_labels", None)
        drop = list(drop) if drop else None
        pol = hydra.utils.instantiate(cfg.policy)
        pol.load_state_dict(p["state_dicts"]["ema_model"], strict=True)
        pol.eval().cuda()
        mods = [l.cross_attn_token_to_image for l in pol.model.one_way_transformer.layers]
        To, G = pol.n_obs_steps, int(cfg.policy.pointcloud_encoder_cfg.num_group)
        n_layers = len(mods)
        acc_mass = {c: np.zeros(n_layers) for c in list(CLASSES.values()) + crit_names + ["other_keys"]}
        acc_share = {c: 0.0 for c in list(CLASSES.values()) + crit_names}
        H = mods[0].num_heads
        acc_head = {c: np.zeros((n_layers, H)) for c in list(CLASSES.values()) + crit_names}
        acc_ent = np.zeros((n_layers, H))
        n_win = 0
        for s in range(0, len(frames), a.batch):
            chunk = frames[s:s + a.batch]
            pcs, labs, lens = [], [], []
            for pc, lab, _, _ in chunk:
                fp, fl, fn = [], [], []
                for t in range(To):
                    c, l = pc[t, :N], lab[t, :N]
                    if drop:
                        c2, n = cut_points_by_label(c, l, drop)
                        keep = ~np.isin(l, drop)
                        l2 = np.concatenate([l[keep], np.full(N - n, l[keep][0])])
                        c, l = c2, l2
                    else:
                        n = N
                    fp.append(c); fl.append(l); fn.append(n)
                pcs.append(fp); labs.append(fl); lens.append(fn)
            obs = {"point_cloud": torch.tensor(np.asarray(pcs), dtype=torch.float32).cuda(),
                   "agent_pos": torch.tensor(np.stack([v for _, _, v, _ in chunk]), dtype=torch.float32).cuda()}
            if drop:
                obs["point_cloud_len"] = torch.tensor(np.asarray(lens), dtype=torch.int64).cuda()
            for m in mods:
                m.store_attn = True
            with torch.no_grad():
                pol.predict_action(obs)
            B = len(chunk)
            cidx = stash["centre_idx"].reshape(B, To, G).cpu().numpy()
            labs = np.asarray(labs)                                  # [B, To, N]
            xyz = np.asarray(pcs)[..., :3]
            tok_lab = np.take_along_axis(labs, cidx, axis=2)         # [B, To, G]
            tok_xyz = np.take_along_axis(xyz, cidx[..., None].repeat(3, -1), axis=2)
            crit = np.stack([c for _, _, _, c in chunk])             # [B, K, 3]
            for li, m in enumerate(mods):
                full = m.last_attn.float()                                  # [B, H, Q, n_keys]
                ent = -(full.clamp_min(1e-12) * full.clamp_min(1e-12).log()).sum(-1) / np.log(full.shape[-1])
                acc_ent[li] += ent.mean(dim=2).sum(0).cpu().numpy()        # normalized entropy, 1 = uniform
                att_h = full.mean(dim=2).cpu().numpy()[:, :, :To * G].reshape(B, H, To, G)
                for cl, name in CLASSES.items():
                    acc_head[name][li] += (att_h * (tok_lab == cl)[:, None]).sum((0, 2, 3))
                for ki, name in enumerate(crit_names):
                    near_h = ((tok_lab == 8) | (tok_lab == 9)) & \
                        (np.linalg.norm(tok_xyz - crit[:, ki][:, None, None], axis=-1) < 0.03)
                    acc_head[name][li] += (att_h * near_h[:, None]).sum((0, 2, 3))
                att = full.mean(dim=(1, 2)).cpu().numpy()                   # [B, n_keys]
                n_keys = att.shape[1]
                pc_att = att[:, :To * G].reshape(B, To, G)
                acc_mass["other_keys"][li] += att[:, To * G:].sum(1).sum() if n_keys > To * G else 0.0
                for cl, name in CLASSES.items():
                    acc_mass[name][li] += (pc_att * (tok_lab == cl)).sum()
                for ki, name in enumerate(crit_names):
                    near = ((tok_lab == 8) | (tok_lab == 9)) & \
                        (np.linalg.norm(tok_xyz - crit[:, ki][:, None, None], axis=-1) < 0.03)
                    acc_mass[name][li] += (pc_att * near).sum()
                    if li == 0:
                        acc_share[name] += near.sum() / (To * G)
            for cl, name in CLASSES.items():
                acc_share[name] += (tok_lab == cl).sum() / (To * G)
            n_win += B
            for m in mods:
                m.store_attn = False
                m.last_attn = None
        run = os.path.basename(os.path.dirname(os.path.dirname(ck))).replace("robotwin2_sim-hanging_mug_labeled-", "")
        res = {"n_windows": n_win, "n_keys": int(n_keys), "pc_keys": To * G,
               "classes": {c: {"token_share": acc_share[c] / n_win,
                               "attention_per_layer": (acc_mass[c] / n_win).tolist(),
                               "attention_mean": float((acc_mass[c] / n_win).mean())}
                           for c in acc_share},
               "other_keys_attention_mean": float((acc_mass["other_keys"] / n_win).mean()),
               "per_head": {"entropy_norm": (acc_ent / n_win).tolist(),
                            "attention": {c: (acc_head[c] / n_win).tolist() for c in acc_head}}}
        results[run] = res
        print(f"\n=== {run} ({N} pts, {G} tokens{', cut ' + str(drop) if drop else ''}) -- {n_win} windows, "
              f"{n_keys} keys ({To * G} point tokens)")
        print(f"  {'class':<14}{'token share':>12}{'attention':>11}{'ratio':>8}   per layer")
        for c, d in res["classes"].items():
            r = d["attention_mean"] / d["token_share"] if d["token_share"] > 0 else float("nan")
            print(f"  {c:<14}{d['token_share']:>12.1%}{d['attention_mean']:>11.1%}{r:>8.2f}   "
                  + " ".join(f"{v:.1%}" for v in d["attention_per_layer"]))
        # per head: attention share / token share per class, spread over the L x H heads
        print(f"  per head ({n_layers} layers x {H} heads): ratio = head attention share / token share; "
              f"entropy 1.0 = uniform over {n_keys} keys")
        ent = acc_ent / n_win
        print(f"    entropy: min {ent.min():.2f}  median {np.median(ent):.2f}  max {ent.max():.2f}  "
              f"(most focused: L{np.unravel_index(ent.argmin(), ent.shape)[0]}H{np.unravel_index(ent.argmin(), ent.shape)[1]})")
        for c in acc_head:
            share = acc_share[c] / n_win
            if share <= 0:
                continue
            ratio = (acc_head[c] / n_win) / share
            top = np.argsort(ratio.ravel())[::-1][:3]
            print(f"    {c:<14} ratio min {ratio.min():.2f}  median {np.median(ratio):.2f}  max {ratio.max():.2f}  "
                  f"| top heads " + ", ".join(f"L{i // H}H{i % H} {ratio.ravel()[i]:.2f} ({(acc_head[c] / n_win).ravel()[i]:.0%})"
                                             for i in top))
        if res["other_keys_attention_mean"] > 0:
            print(f"  non-point keys: {res['other_keys_attention_mean']:.1%}")
        del pol, p
        torch.cuda.empty_cache()
    out = os.path.join(DENSE, "attention_by_class_per_head.json")
    json.dump(results, open(out, "w"), indent=1)
    print(f"\nwritten: {out}")


if __name__ == "__main__":
    main()
