#!/usr/bin/env python
"""Score sim-baseline checkpoints by RoboTwin2 rollouts.  Run in the r3d_sim env.

    python scripts/eval_robotwin2_ckpt.py R3D/data/outputs/sim/<run>/checkpoints/1000.ckpt
    python scripts/eval_robotwin2_ckpt.py <run>/checkpoints/{300,600,1000}.ckpt --episodes 100
    python scripts/eval_robotwin2_ckpt.py <ckpt> --episodes 3 --render-freq 5   # live SAPIEN window
    python scripts/eval_robotwin2_ckpt.py <ckpt> --noise-offset 1               # same scenes, new policy noise

Replaces `scripts/robotwin2_eval.sh`, which calls a `RoboTwin2.0_3D_policy/test.py`
that is not in this repo. Everything (policy, env_runner, RoboTwin task + task
config) comes from the checkpoint's own cfg, so evaluation uses the identical
crop -> FPS pipeline the data was collected with. EMA weights when the run used
EMA, as train.py does.

Eval seeds start at 100000 * (1 + seed) (RoboTwin2EnvManager), disjoint from the
collection seeds. Results: <ckpt>.eval_<n>ep_seed<s>[_noise<k>].json next to each ckpt.

Determinism: every episode re-seeds numpy AND torch from its scene seed
(_base_task._init_task_env_), so a checkpoint sees the SAME policy noise
(diffusion init noise, FPS random start, token shuffle) on a scene every time it
is evaluated -- re-running an eval reproduces it exactly. `--noise-offset k`
re-seeds torch with (scene seed + k * 1000003) right AFTER the scene is built:
identical scenes, different policy draws. That isolates sampling noise from
scene variation. Videos are always written (RoboTwin hard-codes it; it costs ~
nothing: the rgb is rendered for the cloud colours anyway).
"""

import argparse
import copy
import json
import os
import pathlib
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
R3D_DIR = ROOT / "R3D"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ckpts", nargs="+")
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0, help="eval seed block (not the training seed)")
    ap.add_argument("--noise-offset", type=int, default=0,
                    help="k > 0: re-seed torch per episode with scene seed + k*1000003 after the scene is "
                         "built -> same scenes, different policy noise")
    ap.add_argument("--render-freq", type=int, default=0,
                    help="open the SAPIEN viewer and redraw every N sim steps (needs a display); "
                         "only the policy's episodes, not the expert runs that validate eval seeds")
    a = ap.parse_args()

    ckpts = [pathlib.Path(c).resolve() for c in a.ckpts]  # before the chdir
    # repo root, as train.py: relative cfg paths (pretrained_weights_path
    # "R3D/pretrain_weight", zarr paths) resolve against it. The trained ckpt
    # overwrites the pretrained weights anyway; this only keeps the log honest.
    os.chdir(ROOT)
    sys.path.insert(0, str(R3D_DIR))
    import dill
    import hydra
    import torch
    from omegaconf import OmegaConf, open_dict
    OmegaConf.register_new_resolver("eval", eval, replace=True)

    runner, runner_key = None, None
    for ck in ckpts:
        payload = torch.load(ck.open("rb"), pickle_module=dill, map_location="cpu", weights_only=False)
        cfg = copy.deepcopy(payload["cfg"])
        # RoboTwin names the video dir "<epoch>-<basename(output_dir)>-<task_config>"
        # (robotwin2/eval_video/), so the basename must be the RUN, not the epoch
        tag = (ck.parent.parent.name + (f"_seed{a.seed}" if a.seed else "")
               + (f"_noise{a.noise_offset}" if a.noise_offset else ""))
        out_dir = ck.parent.parent / "eval" / tag   # basename names the video dir
        with open_dict(cfg):
            cfg.task.env_runner.output_dir = str(out_dir)
            cfg.task.env_runner.eval_episodes = a.episodes
            cfg.task.env_runner.seed = a.seed

        policy = hydra.utils.instantiate(cfg.policy)
        key = "ema_model" if cfg.training.use_ema else "model"
        policy.load_state_dict(payload["state_dicts"][key], strict=True)
        policy.eval().cuda()
        del payload

        # one runner per (task, config): building it imports and initialises RoboTwin
        rk = (cfg.task_name, cfg.setting, a.episodes, a.seed)
        if runner is None or rk != runner_key:
            runner = hydra.utils.instantiate(cfg.task.env_runner)
            runner_key = rk
        runner.output_dir = str(out_dir)
        if a.noise_offset:
            em = runner.env_manager
            if not getattr(em, "_noise_patched", False):
                orig_init_n = em.Init_task_env

                def init_with_noise(seed, *args, **kwargs):
                    out = orig_init_n(seed, *args, **kwargs)  # builds the scene (seeds np + torch)
                    torch.manual_seed(int(seed) + a.noise_offset * 1000003)
                    return out
                em.Init_task_env = init_with_noise
                em._noise_patched = True
        if a.render_freq > 0:
            em = runner.env_manager
            if not getattr(em, "_render_patched", False):
                orig_init = em.Init_task_env

                def init_with_viewer(*args, **kwargs):
                    em.args["render_freq"] = a.render_freq
                    return orig_init(*args, **kwargs)
                em.Init_task_env = init_with_viewer
                em._render_patched = True

        t0 = time.time()
        # NOT under torch.no_grad(): the runner validates eval seeds by running the
        # expert, and curobo's trajectory optimizer needs autograd. The runner
        # already wraps predict_action in no_grad itself.
        log = runner.run(policy, epoch=ck.stem, task_config=cfg.setting)
        res = {
            "ckpt": str(ck), "weights": key, "task_name": cfg.task_name, "task_config": cfg.setting,
            "episodes": a.episodes, "eval_seed": a.seed, "noise_offset": a.noise_offset, "minutes": round((time.time() - t0) / 60, 1),
            "success_rate": float(log["test_mean_score"]), "runner_log": log,
        }
        suffix = f"_noise{a.noise_offset}" if a.noise_offset else ""
        path = ck.with_name(f"{ck.stem}.eval_{a.episodes}ep_seed{a.seed}{suffix}.json")
        path.write_text(json.dumps(res, indent=2))
        print(f"{ck.name}: success {res['success_rate']:.1%} over {a.episodes} episodes -> {path}")
        del policy
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
