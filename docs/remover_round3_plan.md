# Learned remover, round 3: seeds and the specialist reference (planned 2026-11-01)

Handoff for a new chat. Background: `docs/sim_token_allocation_learnings.md` ("Learned link
remover v0", rounds 1–2), design and knobs: `docs/learned_remover_design.md`. Nothing from
rounds 1–3 is committed yet (user decides when).

## Where we are
Round 2 (4096 points / 64 tokens, seed 42, 200 paired scenes) found the first working learned
removal. **GObal** trains θ on gates drawn at a fixed keep 0.5 (`theta_keep=0.5`), decoupled
from φ; φ starts at 0.5 and learns. It learned "remove link5 + link6, keep link7 / hand /
fingers": 22.0% at 200 and 19.5% at 250, the same as the oracle arm cut (20.0%, p = 1.0) and
better than uncut (8.5%, p < 0.001). **GOmix** (structured masks, init 0.8, θ coupled to φ)
reached the same mask: 18.5 / 16.5%; vs GObal p = 0.38 / 0.47. Everything is one seed, and seed
noise is the open problem: the arm cut is 20.0% at seed 42 and 16.5% at seed 43.

## Goals (user, 2026-11-01)
1. **Second seed of GObal**: is "learned removal ≈ oracle" robust?
2. **Specialist reference**: a policy trained from scratch, all run long, on the FIXED mask
   GObal learned (link5 + link6 removed, everything else kept). How much does GObal lose by
   training θ on balanced random masks instead of the final mask, if anything?
3. Seeds over more eval scenes: more scenes do not remove seed noise.
4. **Open, decide at the start:** GOmix with init_keep 0.5, or a second seed of GOmix (0.8).
   - GOmix-0.5 vs GObal is still not one variable (coupling AND structured masks).
   - Recommendation: run neither in the main queue (GObal is the method, GOmix's main value
     was the per-mask diagnostic, which round 2 already gave); queue GOmix-0.5 last only if
     time is left.

**Decided (user, 2026-11-01): run ONLY 1–3 now.** Runs 4 and 5 are deferred; the user
continues once 1–3 are in.

## Queue (in this order; ~3.9 h train + ~1.4 h per evaluated checkpoint each)
| # | tag / run | what | evals |
|---|---|---|---|
| 1 | GObal seed 43 | same overrides as GObal: `learn_density=false init_keep=0.5 theta_keep=0.5` | 250, 200 |
| 2 | SPEC seed 42 | fixed mask {link5, link6} removed, no φ (needs code, below) | 250, 200 |
| 3 | SPEC seed 43 | same | 250, 200 |
| 4 | uncut 4096 seed 43 | plain baseline. Only seed 42 exists for uncut 4096 (the arm cut has 42 and 43), so this gives the baseline's own seed pair | 250, 125 (as seed 42 has) |
| 5 | (optional) GOmix-0.5 seed 42 | `learn_density=false init_keep=0.5 mix_eps=0.25 mix_masks=[[],[link5,link6,link7],[link5,link6,link7,hand_f1side,hand_centre,hand_f2side],all]` | 250 |

Pass 1 (train + eval 250 for every run) before pass 2 (the second checkpoints), as
`scripts/sim/run_remover_sweep.sh` already does. Total for 1–4 with both evals: ~27 h; with
the 250 evals only: ~21 h.

Commands:
- Runs 1, 2, 3, 5: `scripts/sim/run_remover_sweep.sh` with a new queue file
  `R3D/data/outputs/sim/mimicgen/remover_tune3_queue.txt`, lines `<points> <seed> <tag> [overrides]`.
  Env: `KEEP_ONLY_EVALUATED=1 SWEEP_OUT=R3D/data/outputs/sim/mimicgen/remover_tune3 MID_CKPTS="200"`.
  Run dir `mimicgen_sim_remover-square_d2-p4096-g64-rm-<tag>-<seed>`.
- Run 4 is not a remover run, so it goes through the baseline launcher:
  ```bash
  TASK_CFG=mimicgen_sim SETTING=agentview_p4096-200 EPOCHS=251 CHECKPOINT_EVERY=125 BATCH_SIZE=64 GRAD_ACCUM=2 \
      TAG=p4096-g64 bash scripts/train_robotwin2_sim.sh square_d2 43 0 task.num_group=64 task.num_points=4096
  ```
  Then evaluate with `scripts/sim/eval_mimicgen_ckpt.py <ckpt> --episodes 200 --workers 5`
  (env `r3d_robosuite`).
  - Run dir: `mimicgen_sim-square_d2-p4096-g64-43`. `analyze_remover_run.py`'s
    `baselines()` picks it up automatically.
  - Check that the hydra overrides match seed 42's
    (`mimicgen_sim-square_d2-p4096-g64-42/.hydra/overrides.yaml`).
- Run the GPU jobs one after another (one 4090).

## Code needed: the specialist (fixed mask, no φ)
Default off; existing configs and checkpoints must behave bit-for-bit as before.
- **Config** `policy.link_remover.fixed_removed: [link5, link6]` (null = off). LinkRemover:
  - gate logits set to −gate_max for the listed regions and +gate_max for the rest, so the
    deterministic mask is exactly "listed removed, rest kept";
  - `learn_density=false`; unknown region names raise;
  - nothing in φ is learned (`phi_parameters()` empty).
- **Policy:** with `fixed_removed`, `compute_loss` always uses the deterministic mask
  (`remover_mode` forced "deterministic"), in training and evaluation alike.
- **`RemoverController`** (`R3D/r3d/common/link_remover_training.py`): with a fixed mask,
  - no φ optimizer, no φ steps;
  - phase always "fixed" (it currently raises when there is no φ phase);
  - keep the epoch logging and `mechanism_check`; skip the held-out and B diagnostics (they
    compare masks).
- **Checks** to add to `scripts/sim/check_link_remover.py`:
  - the training and eval clouds have exactly link5 + link6 removed and every other point kept;
  - no φ parameter changes over a controller step;
  - the remover-off == DP3 bit-identity still holds.
- **Smoke run** (10 epochs × 40 batches, `checkpoint.save_ckpt=False`, as in round 2) before
  the queue starts. Delete the smoke folders after.

## Analysis / report
- `scripts/sim/analyze_remover_run.py` per run, and the sweep report over GO, GObal s42/s43,
  SPEC s42/s43, (GOmix runs).
- For SPEC there are no φ tables; it needs the mechanism table and success.
- **Report:**
  - Success at 200 / 250 per seed.
  - Paired McNemar within a seed (GObal vs SPEC vs arm cut vs uncut, same seed) and the mean
    over the two seeds.
  - The seed spread of each method, which is the noise scale for every comparison.
  - The questions to answer:
    - Is GObal ≈ SPEC? Then balanced θ training costs nothing.
    - Is GObal ≈ the arm cut on both seeds?
    - Did GObal seed 43 learn the same mask? Which gates, and when did they cross 0.5?
- Add the results to the learnings doc ("Round 3").

## Constraints (standing)
- No commit or push unless asked; never a Claude co-author trailer.
- Confirm deletions of the user's data; deleting your own smoke folders is fine.
- Never `pkill -f` / `pgrep -f` from a tool shell: the pattern matches the shell itself. Find
  PIDs with `ps -eo pid,cmd` and an exact match.
- Present results as tables, verify numbers before claiming them.
- Keep only evaluated checkpoints (`KEEP_ONLY_EVALUATED=1`).
- Run `check_link_remover.py` before launching; it must pass.
