#!/usr/bin/env python
"""Nut-YAW sensitivity (user, 2026-11-14): does a policy read the nut's ORIENTATION?  Run in r3d_sim.

    python scripts/sim/sensitivity_yaw.py [--stride 1] [--only NAME ...]

Hypothesis under test (shortcut learning): with fine object patches the nut's shape / orientation must be assembled
across tokens and is hard to learn, so a policy that can lean on the arm learns it poorly; coarse patches (1024,
scenesparse, k128) make it easy, and the arm cut / dropout force it. Prediction: uncut 4096 (and armsparse) react
LESS to a nut rotation than the cut / dropout / coarse-patch policies. The round-7 translation test could not see this
(position is the easy part).

Same machinery as sensitivity_shortcut.py (round 7): held-out windows = every frame of the 4 val demos, 2 obs frames,
no augmentation; each policy's own cloud (num_points prefix, cut / link cut / prefix thinning / remover) built AFTER
the perturbation; crn = perturbed observation with the base call's own seed (the pure effect), floor = two seeds on
the identical observation. Perturbations, on both obs frames, nut points (label 8) only:
  nut_yaw15 / nut_yaw45 : rotation about the vertical axis through the nut's centroid (latest obs frame), by +-15 /
                          +-45 deg (sign random per window, the same for every policy)
  nut_shift2            : horizontal translation of the nut by 2 cm (position reference, random direction)
Measures on the EXECUTED 8 action steps: d_pos (mm, mean |d goal pos|) and d_rot (deg, mean geodesic angle between the
goal rotations). Phases: all, early (< 0.3, before the grasp: the nut yaw decides the approach), mid / late.
Writes R3D/data/outputs/sim/mimicgen/yaw_sensitivity/sensitivity_yaw.json (+ a markdown table).
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
OUT = ROOT / "R3D/data/outputs/sim/mimicgen/yaw_sensitivity"
P = "mimicgen_sim-square_d2-p4096-g64"
RM = "mimicgen_sim_remover-square_d2"
POLICIES = {   # name -> run dir (250.ckpt); missing checkpoints are skipped
    "uncut 4096 s42": f"{P}-42", "uncut 4096 s43": f"{P}-43",
    "arm cut 4096 s42": f"{P}-cut5-42", "arm cut 4096 s43": f"{P}-cut5-43", "arm cut 4096 s44": f"{P}-cutL07-44",
    "DROP s42": f"{RM}-p4096-g64-rm-DROP-42", "FTcont s42": f"{RM}-p4096-g64-rm-FTcont-42", "DROP s43": f"{RM}-p4096-g64-rm-DROP-43",
    "armsparse s42": f"{P}-armsparse-42", "scenesparse s42": f"{P}-scenesparse-42",
    "1024 uncut s42": "mimicgen_sim-square_d2-p1024-g64-42", "1024 uncut s43": "mimicgen_sim-square_d2-p1024-g64-43",
    "DROP 1024 s42": f"{RM}-p1024-g64-rm-DROP-42",
    "4096/128 uncut": "mimicgen_sim-square_d2-p4096-g128-42", "4096/256 uncut": "mimicgen_sim-square_d2-p4096-g256-42",
    "4096/64 k128": f"{P}-k128-42",
}
PERTS = [("nut_yaw15", 15.0), ("nut_yaw45", 45.0), ("nut_shift2", 0.02)]
PHASES = {"all": (0.0, 1.01), "early <0.3": (0.0, 0.3), "mid 0.3-0.7": (0.3, 0.7), "late >0.7": (0.7, 1.01)}


def perturb_nut(W, kind, val, signs, dirs):
    pc = W["pc"].copy()
    sel = W["label"] == 8                                       # (B, T, N) nut points
    if kind == "nut_shift2":
        d = np.zeros((len(pc), 3), np.float32)
        d[:, :2] = val * dirs
        pc[..., :3] += np.where(sel[..., None], d[:, None, None, :], 0.0).astype(np.float32)
        return pc, sel
    for b in range(len(pc)):
        s_last = sel[b, -1]
        if not s_last.any():
            continue
        c = pc[b, -1, s_last, :2].mean(0)                       # centroid in the latest obs frame
        a = np.deg2rad(val) * signs[b]
        R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]], np.float32)
        for t in range(pc.shape[1]):
            m = sel[b, t]
            pc[b, t, m, :2] = (pc[b, t, m, :2] - c) @ R.T + c
    return pc, sel


def drot_deg(a, b):
    from r3d.common.robosuite_actions import rot6d_to_mat
    Ra, Rb = rot6d_to_mat(a[..., 3:9]), rot6d_to_mat(b[..., 3:9])
    tr = np.einsum("...ij,...ij->...", Ra, Rb)                  # trace(Ra^T Rb)
    return np.degrees(np.arccos(np.clip((tr - 1) / 2, -1, 1))).mean(-1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--only", nargs="*", default=None)
    a = ap.parse_args()
    os.chdir(ROOT)
    import torch
    from eval_mimicgen_ckpt import load_policy
    from sensitivity_shortcut import load_windows, predict, dpos_mm, boot_ci
    W, val = load_windows(ROOT / "R3D/data/square_d2-agentview_p4096-200.zarr", a.stride)
    rng = np.random.default_rng(11)
    signs = rng.choice([-1.0, 1.0], len(W["pc"]))
    ang = rng.uniform(0, 2 * np.pi, len(W["pc"]))
    dirs = np.stack([np.cos(ang), np.sin(ang)], 1)
    ph = W["phase"]
    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / "sensitivity_yaw.json"
    res = json.load(open(out)) if out.exists() else {}
    res.update({"val_episodes": val.tolist(), "windows": int(len(W["pc"])),
                "nut_points_mean": float((W["label"][:, -1] == 8).sum(1).mean())})
    res.setdefault("policies", {})
    print(f"val demos {val.tolist()}: {len(W['pc'])} windows", flush=True)
    for name, run in POLICIES.items():
        if a.only and name not in a.only:
            continue
        ck = ROOT / "R3D/data/outputs/sim" / run / "checkpoints/250.ckpt"
        if not ck.exists():
            print("missing", ck, flush=True)
            continue
        policy, cfg, _ = load_policy(str(ck))
        base = predict(policy, cfg, W["pc"], W, 1, a.bs)
        base2 = predict(policy, cfg, W["pc"], W, 2, a.bs)
        r = {"run": run, "floor": {}, "perts": {}}
        for pn, (lo, hi) in PHASES.items():
            m = (ph >= lo) & (ph < hi)
            r["floor"][pn] = {"pos_mm": float(dpos_mm(base, base2)[m].mean()), "rot_deg": float(drot_deg(base, base2)[m].mean())}
        for kind, v in PERTS:
            pc, sel = perturb_nut(W, kind, v, signs, dirs)
            pert = predict(policy, cfg, pc, W, 1, a.bs)              # base call's own seed: pure effect
            dp, dr = dpos_mm(pert, base), drot_deg(pert, base)
            r["perts"][kind] = {pn: {"pos_mm": float(dp[(ph >= lo) & (ph < hi)].mean()),
                                     "rot_deg": float(dr[(ph >= lo) & (ph < hi)].mean()),
                                     "rot_deg_ci": boot_ci(dr[(ph >= lo) & (ph < hi)])}
                                for pn, (lo, hi) in PHASES.items()}
        res["policies"][name] = r
        print(name, {k: (round(v["all"]["rot_deg"], 2), round(v["early <0.3"]["rot_deg"], 2)) for k, v in r["perts"].items()},
              "floor", round(r["floor"]["all"]["rot_deg"], 2), flush=True)
        json.dump(res, open(out, "w"), indent=1)
        del policy
        torch.cuda.empty_cache()
    # table
    L = ["Nut-yaw sensitivity: change of the executed action (8 steps), pure effect (same seed), held-out val windows", "",
         "| policy | floor rot (deg) | yaw 15: rot deg (early) | yaw 45: rot deg (early) | yaw 45: pos mm | shift 2 cm: pos mm |",
         "|---|---|---|---|---|---|"]
    for name, r in res["policies"].items():
        p = r["perts"]
        L.append(f"| {name} | {r['floor']['all']['rot_deg']:.2f} | {p['nut_yaw15']['all']['rot_deg']:.2f} "
                 f"({p['nut_yaw15']['early <0.3']['rot_deg']:.2f}) | {p['nut_yaw45']['all']['rot_deg']:.2f} "
                 f"({p['nut_yaw45']['early <0.3']['rot_deg']:.2f}) | {p['nut_yaw45']['all']['pos_mm']:.1f} | "
                 f"{p['nut_shift2']['all']['pos_mm']:.1f} |")
    (OUT / "sensitivity_yaw.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
