#!/usr/bin/env python
"""Round 7, part 1: does a policy react to the OBJECTS or to the ROBOT points?  Run in r3d_sim.

    python scripts/sim/sensitivity_shortcut.py [--stride 1] [--out R3D/data/outputs/sim/mimicgen/round7/sensitivity.json]

Offline, no training. Held-out windows = every frame of the 4 validation demos (the training
split's own val mask, seed 42), observation = 2 frames as in training, no augmentation.
Perturbations, each a horizontal translation by 2 or 5 cm in a per-window random direction
(the same direction for every policy and both magnitudes), applied to BOTH obs frames of the raw
cloud BEFORE each policy's own cut / remover:
  objects : nut + peg points (point_label 8, 9) only
  arm57   : robot points of link cells link5, link6, link7 only
  gripper : hand + finger cells only (extra; the arm cut keeps them)
Measure: change of the EXECUTED action (8 steps) position, mean over steps of |d pos| in mm.
`--task threading` / `--task mug` (2026-11-09) run the same test on MimicGen Threading (narrowed) and RoboTwin
hanging_mug. hanging_mug has class labels only (no link cells) and JOINT-space actions: kinds `arms` (classes 4, 5),
`grippers` (6, 7), `objects` (mug 8, rack 9), and the measure is |d action| over all 14 joint dims (rad) -- compare
tasks by the ratio to each policy's own floor, not by the raw number.
  floor  = two calls on the identical observation (fresh diffusion noise / FPS start each);
  effect = perturbed observation vs base call, fresh noise (ratio ~1 = the policy ignores it,
           as the cube study in ~/R3D-Policy/CLAUDE.md);
  crn    = perturbed observation with the base call's own seed: the perturbation's pure effect
           (exactly 0 if the perturbed points never reach the policy).
"""

import argparse
import json
import os
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "R3D"))
sys.path.insert(0, str(ROOT / "scripts" / "sim"))
OUT = ROOT / "R3D/data/outputs/sim"
POLICIES = {
    "uncut": "mimicgen_sim-square_d2-p4096-g64-42",
    "arm cut": "mimicgen_sim-square_d2-p4096-g64-cut5-42",
    "SPEC (link5+6 removed)": "mimicgen_sim_remover-square_d2-p4096-g64-rm-SPEC-42",
    "SPEC06 (link0-6 removed)": "mimicgen_sim_remover-square_d2-p4096-g64-rm-SPEC06-42",
    "DROP": "mimicgen_sim_remover-square_d2-p4096-g64-rm-DROP-42",
    "FTcont": "mimicgen_sim_remover-square_d2-p4096-g64-rm-FTcont-42",
    "GObal (learned mask)": "mimicgen_sim_remover-square_d2-p4096-g64-rm-GObal-42",
    "frozen robot": "mimicgen_sim-square_d2-p4096-g64-frozen-42",
    "1024 uncut s42": "mimicgen_sim-square_d2-p1024-g64-42",
    "1024 uncut s43": "mimicgen_sim-square_d2-p1024-g64-43",
    "1024 arm cut s42": "mimicgen_sim-square_d2-p1024-g64-cut5-42",
    "1024 arm cut s43": "mimicgen_sim-square_d2-p1024-g64-cut5-43",
}
PERTS = [("objects", 0.02), ("objects", 0.05), ("arm57", 0.02), ("arm57", 0.05), ("gripper", 0.02), ("gripper", 0.05)]
TASKS = {
    "square": dict(zarr="R3D/data/square_d2-agentview_p4096-200.zarr", ckpt="250", policies=POLICIES, perts=PERTS,
                   out="mimicgen/round7/sensitivity.json", joint=False),
    "threading": dict(zarr="R3D/data/threading_d1n-agentview_p4096-200.zarr", ckpt="250", perts=PERTS, joint=False,
                      out="mimicgen/threading/sensitivity.json", policies={
                          "uncut": "mimicgen_sim-threading_d1n-p4096-g64-42",
                          "arm cut": "mimicgen_sim-threading_d1n-p4096-g64-cut5-42",
                          "GObalAll": "mimicgen_sim_remover-threading_d1n-p4096-g64-rm-GObalAll-42"}),
    "mug": dict(zarr="R3D/data/hanging_mug_labeled-sim_stock_p4096-50.zarr", ckpt="600", joint=True,
                out="mug_drop/sensitivity.json",
                perts=[("objects", 0.02), ("objects", 0.05), ("arms", 0.02), ("arms", 0.05), ("grippers", 0.02), ("grippers", 0.05)],
                policies={"uncut": "robotwin2_sim-hanging_mug_labeled-p4096-g64-42",
                          "arms cut": "robotwin2_sim-hanging_mug_labeled-p4096-g64-cut45-42",
                          "dropout": "robotwin2_sim-hanging_mug_labeled-p4096-g64-drop4567-42"}),
}


def load_windows(zarr_path, stride):
    import zarr
    from r3d.common.sampler import get_val_mask
    z = zarr.open(str(zarr_path), "r")
    ends = z["meta/episode_ends"][:]
    starts = np.concatenate([[0], ends[:-1]])
    val = np.flatnonzero(get_val_mask(n_episodes=len(ends), val_ratio=0.02, seed=42))
    W = {k: [] for k in ("pc", "label", "link", "state", "phase", "ep")}
    for ep in val:
        s, e = int(starts[ep]), int(ends[ep])
        pc, lab = z["data/point_cloud"][s:e], z["data/point_label"][s:e]
        link = z["data/point_link"][s:e] if "point_link" in z["data"] else np.full(lab.shape, -1, np.int8)
        st = z["data/state"][s:e]
        for t in range(0, e - s, stride):
            f = [max(t - 1, 0), t]                   # n_obs_steps 2, pad_before 1: frame 0 twice
            W["pc"].append(pc[f]); W["label"].append(lab[f]); W["link"].append(link[f])
            W["state"].append(st[f]); W["phase"].append(t / (e - s)); W["ep"].append(int(ep))
    return {k: np.asarray(v) for k, v in W.items()}, val


def perturb(W, kind, mag, rng_dirs):
    pc = W["pc"].copy()
    if kind == "objects":
        sel = np.isin(W["label"], (8, 9))
    elif kind == "arm57":
        sel = np.isin(W["link"], (5, 6, 7))
    elif kind == "gripper":
        sel = W["link"] >= 8
    elif kind == "arms":
        sel = np.isin(W["label"], (4, 5))
    elif kind == "grippers":
        sel = np.isin(W["label"], (6, 7))
    else:
        raise ValueError(kind)
    d = np.zeros((len(pc), 3), np.float32)
    d[:, :2] = mag * rng_dirs
    pc[..., :3] += np.where(sel[..., None], d[:, None, None, :], 0.0).astype(np.float32)
    return pc, sel


_REF = {}


def frozen_cloud(cfg, pc, W, idx, move=None):
    """round 7 policies: freeze as in training/eval (r3d.common.robot_freeze); `move` (B, 3)
    then translates the FROZEN robot points."""
    from r3d.common.robot_freeze import freeze_robot_points, reference_robot_index
    lab_set = [int(c) for c in cfg.task.freeze_robot_labels]
    if "ref" not in _REF:
        ref = np.load(ROOT / cfg.task.freeze_reference)
        _REF["ref"] = ref["cloud"]
        _REF["idx"] = [reference_robot_index(l, lab_set, pc.shape[2]) for l in ref["label"]]
    out = np.empty_like(pc[idx])
    for b, i in enumerate(idx):
        ep = W["ep"][i]
        for t in range(pc.shape[1]):
            out[b, t] = freeze_robot_points(pc[i, t], W["label"][i, t], _REF["ref"][ep], _REF["idx"][ep], lab_set)
            if move is not None:
                r = np.isin(W["label"][i, t], lab_set)
                out[b, t, r, :3] += move[b]
    return out


def build_obs(policy, cfg, pc, W, idx, move_frozen=None):
    import torch
    from r3d.common.label_cut import cut_points_by_label
    obs = {"agent_pos": torch.from_numpy(W["state"][idx].astype(np.float32))}
    drop = cfg.task.get("drop_point_labels", None)
    if cfg.task.get("freeze_robot_labels", None):
        p = frozen_cloud(cfg, pc, W, idx, None if move_frozen is None else move_frozen[idx])
    else:
        p = pc[idx]
    # num_points < stored: the first N points = the FPS-N cloud (as RobotwinSimDataset)
    N = int(cfg.task.num_points)
    if N > p.shape[2]:
        raise ValueError(f"policy wants {N} points, stored {p.shape[2]}")
    W = {**W, "label": W["label"][:, :, :N], "link": W["link"][:, :, :N]}
    p = p[:, :, :N]
    links = cfg.task.get("drop_point_links", None)
    thin = cfg.task.get("thin_point_set", None)
    if sum(map(bool, (drop, links, thin))) > 1:
        raise ValueError("more than one of drop_point_labels / drop_point_links / thin_point_set")
    if drop or links or thin:
        # 2026-11-14 (yaw test): drop_point_links (keyed on the link cell) and prefix thinning (stored index; the
        # perturbations move points in place, so the stored order is intact) as in training / eval
        if links:
            from r3d.common.robosuite_pcd import CELL_NAMES
            ids = [CELL_NAMES.index(str(c)) for c in links]
            fn = lambda b, t: cut_points_by_label(p[b, t], W["link"][idx][b, t], ids)
        elif thin:
            from r3d.common.prefix_thin import thin_cloud
            fn = lambda b, t: thin_cloud(p[b, t], W["link"][idx][b, t], thin, int(cfg.task.thin_prefix))
        else:
            fn = lambda b, t: cut_points_by_label(p[b, t], W["label"][idx][b, t], list(drop))
        cut = [[fn(b, t) for t in range(p.shape[1])] for b in range(len(p))]
        p = np.stack([[c[0] for c in row] for row in cut])
        obs["point_cloud_len"] = torch.tensor([[c[1] for c in row] for row in cut], dtype=torch.int64)
    obs["point_cloud"] = torch.from_numpy(np.ascontiguousarray(p))
    if getattr(policy, "link_remover", None) is not None:
        obs["point_link"] = torch.from_numpy(W["link"][idx].astype(np.int64))
    return {k: v.cuda() for k, v in obs.items()}


def predict(policy, cfg, pc, W, seed, bs, move_frozen=None):
    import torch
    out = []
    for b0 in range(0, len(pc), bs):
        idx = np.arange(b0, min(b0 + bs, len(pc)))
        torch.manual_seed(seed * 100003 + b0)
        with torch.no_grad():
            out.append(policy.predict_action(build_obs(policy, cfg, pc, W, idx, move_frozen))["action"]
                       .float().cpu().numpy())
    return np.concatenate(out)                         # (B, n_action_steps, 10)


JOINT = [False]


def dpos_mm(a, b):
    if JOINT[0]:   # joint-space actions (RoboTwin): |d action| over all dims, rad -- read as a ratio to the floor
        return np.linalg.norm(a - b, axis=-1).mean(-1)
    return 1000 * np.linalg.norm(a[..., :3] - b[..., :3], axis=-1).mean(-1)   # per window


def boot_ci(x, n=2000, seed=0):
    """95% bootstrap CI of the mean over WINDOWS (windows of one demo are correlated: optimistic)."""
    rng = np.random.default_rng(seed)
    m = x[rng.integers(0, len(x), (n, len(x)))].mean(1)
    return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", default="square", choices=sorted(TASKS))
    ap.add_argument("--ckpt", default=None, help="default: the task's (square / threading 250, mug 600)")
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--only", nargs="*", default=None, help="subset of policy names")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    T = TASKS[a.task]
    a.ckpt = a.ckpt or T["ckpt"]
    a.out = a.out or str(OUT / T["out"])
    JOINT[0] = T["joint"]
    os.chdir(ROOT)
    from eval_mimicgen_ckpt import load_policy
    W, val = load_windows(ROOT / T["zarr"], a.stride)
    rng = np.random.default_rng(7)
    ang = rng.uniform(0, 2 * np.pi, len(W["pc"]))
    dirs = np.stack([np.cos(ang), np.sin(ang)], 1)
    print(f"val demos {val.tolist()}: {len(W['pc'])} windows", flush=True)
    res = {"val_episodes": val.tolist(), "windows": int(len(W["pc"])), "ckpt": a.ckpt, "policies": {}}
    if os.path.exists(a.out):
        res["policies"] = json.load(open(a.out)).get("policies", {})
    early = W["phase"] < 0.5
    for name, run in T["policies"].items():
        if a.only and name not in a.only:
            continue
        ck = OUT / run / "checkpoints" / f"{a.ckpt}.ckpt"
        if not ck.exists():
            print("missing", ck)
            continue
        policy, cfg, _ = load_policy(str(ck))
        base = predict(policy, cfg, W["pc"], W, 1, a.bs)
        base2 = predict(policy, cfg, W["pc"], W, 2, a.bs)
        floor = dpos_mm(base, base2)
        per = {"floor": floor}
        r = {"run": run, "floor_mm": float(floor.mean()), "floor_mm_early": float(floor[early].mean()),
             "floor_mm_late": float(floor[~early].mean()), "perts": {}}
        perts = list(T["perts"])
        if cfg.task.get("freeze_robot_labels", None):
            perts += [("frozenrobot", 0.02), ("frozenrobot", 0.05)]
        for k, (kind, mag) in enumerate(perts):
            move = None
            if kind == "frozenrobot":      # move the frozen robot points (after the swap)
                pc, sel = W["pc"], np.isin(W["label"], [int(c) for c in cfg.task.freeze_robot_labels])
                move = np.zeros((len(pc), 3), np.float32)
                move[:, :2] = mag * dirs
                fresh = dpos_mm(predict(policy, cfg, pc, W, 10 + k, a.bs, move), base)
                crn = dpos_mm(predict(policy, cfg, pc, W, 1, a.bs, move), base)
            else:
                pc, sel = perturb(W, kind, mag, dirs)
                fresh = dpos_mm(predict(policy, cfg, pc, W, 10 + k, a.bs), base)
                crn = dpos_mm(predict(policy, cfg, pc, W, 1, a.bs), base)
            per[f"{kind}_{int(mag * 100)}cm"] = crn
            r["perts"][f"{kind}_{int(mag * 100)}cm"] = {
                "effect_mm": float(fresh.mean()), "ratio": float(fresh.mean() / floor.mean()),
                "crn_mm": float(crn.mean()), "crn_mm_ci": boot_ci(crn), "crn_ratio": float(crn.mean() / floor.mean()),
                "ratio_early": float(fresh[early].mean() / floor[early].mean()),
                "ratio_late": float(fresh[~early].mean() / floor[~early].mean()),
                "crn_ratio_early": float(crn[early].mean() / floor[early].mean()),
                "crn_ratio_late": float(crn[~early].mean() / floor[~early].mean()),
                "points_moved_mean": float(sel.sum((1, 2)).mean() / 2)}
        res["policies"][name] = r
        np.savez(a.out.replace(".json", f"_{run}.npz"), **per)
        print(name, json.dumps({"floor_mm": round(r["floor_mm"], 2),
                                **{p: (round(v["ratio"], 2), round(v["crn_ratio"], 2)) for p, v in r["perts"].items()}}),
              flush=True)
        del policy
        import torch
        torch.cuda.empty_cache()
        os.makedirs(os.path.dirname(a.out), exist_ok=True)
        json.dump(res, open(a.out, "w"), indent=1)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
