#!/usr/bin/env python
"""Score sim-baseline checkpoints by RoboTwin2 rollouts.  Run in the r3d_sim env.

    python scripts/eval_robotwin2_ckpt.py R3D/data/outputs/sim/<run>/checkpoints/1000.ckpt
    python scripts/eval_robotwin2_ckpt.py <run>/checkpoints/{300,600,1000}.ckpt --episodes 100

Replaces `scripts/robotwin2_eval.sh`, which calls a `RoboTwin2.0_3D_policy/test.py`
that is not in this repo. Everything (policy, env_runner, RoboTwin task + task
config) comes from the checkpoint's own cfg, so evaluation uses the identical
crop -> FPS pipeline the data was collected with. EMA weights when the run used
EMA, as train.py does.

Eval seeds start at 100000 * (1 + seed) (RoboTwin2EnvManager), disjoint from the
collection seeds. Results: <ckpt>.eval_<n>ep_seed<s>.json next to each ckpt.
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
    ap.add_argument("--no-video", action="store_true")
    a = ap.parse_args()

    ckpts = [pathlib.Path(c).resolve() for c in a.ckpts]  # before the chdir
    os.chdir(R3D_DIR)
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
        out_dir = ck.parent.parent / "eval" / ck.parent.parent.name
        with open_dict(cfg):
            cfg.task.env_runner.output_dir = str(out_dir)
            cfg.task.env_runner.eval_episodes = a.episodes
            cfg.task.env_runner.seed = a.seed
            if a.no_video:
                cfg.task.env_runner.save_video = False

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

        t0 = time.time()
        # NOT under torch.no_grad(): the runner validates eval seeds by running the
        # expert, and curobo's trajectory optimizer needs autograd. The runner
        # already wraps predict_action in no_grad itself.
        log = runner.run(policy, epoch=ck.stem, task_config=cfg.setting)
        res = {
            "ckpt": str(ck), "weights": key, "task_name": cfg.task_name, "task_config": cfg.setting,
            "episodes": a.episodes, "eval_seed": a.seed, "minutes": round((time.time() - t0) / 60, 1),
            "success_rate": float(log["test_mean_score"]), "runner_log": log,
        }
        path = ck.with_name(f"{ck.stem}.eval_{a.episodes}ep_seed{a.seed}.json")
        path.write_text(json.dumps(res, indent=2))
        print(f"{ck.name}: success {res['success_rate']:.1%} over {a.episodes} episodes -> {path}")
        del policy
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
