# Sim token-allocation study — learnings and reasoning (2026-10-17 … 2026-10-24)

Branch `sim-token-alloc`, worktree `~/R3D-Policy-sim`, env `r3d_sim`. Detailed
operational notes (setup, gotchas, commands) are in the "Sim baseline" section of
`~/R3D-Policy/CLAUDE.md`; this file keeps the **reasoning and results** so a new
session can pick up the scientific thread.

## The research idea

Paper idea (user): the point-cloud encoder (R3D / Uni3D) spends its tokens where
farthest-point sampling (FPS) puts them — by visible surface, not by relevance.
Removing irrelevant points (first: robot points, whose pose proprioception already
knows) **after preprocessing** frees tokens for the task-relevant parts.
Final method (user's plan): a **learned** removal — start from a reasonably dense
cloud, the policy learns which proprio-known points to drop (no manual cut, can
also choose *not* to drop). The sim study exists to understand the mechanism
before building it.

## Principles agreed with the user (don't re-litigate without asking)

- **Scene design:** static, never-relevant geometry (table, coaster, left arm in
  a right-arm task, torso) is cut by the workspace box. "Cropping static clutter
  helps" is obvious and **not a claim**. Clutter is NOT added: freed tokens would
  partly flow into it and dilute the method.
- **Difficulty via token budget, not fewer demos.** Target a baseline of
  **30–40% success**. Train to convergence (a baseline weak only from
  undertraining invites "your method just converges faster").
- **Point density:** more points *to resolve details the task needs* is
  legitimate; more points *so coverage collapses and the baseline gets worse* is
  a hack.
- **Not going below 16 tokens** as a claim ("method helps when the encoder is
  starved" is weak); 512 tokens considered too many for the sim tasks.
- **Grippers stay** in the cloud (cutting them hurt, see below).
- Head camera only for now; wrist cams / multi-view later.
- Evaluate on **≥ 200 scenes, paired** (same scenes for every policy; McNemar on
  discordant scenes). 50 scenes swung results by ~10 points from scene choice alone.
- Training-seed repeats and longer schedules are for the *clean paper runs*
  later, once the task is settled.
- Present results as **tables**; verify claims before stating them.

## Task history — why each task was (not) usable

| task | setup | result | verdict |
|---|---|---|---|
| `place_empty_cup_right` | right arm, front camera, pinned + cropped coaster, 1024 pts | 100% success at 128/64/32/16 tokens; blinding test 20/20 → 0/20 (policy uses the cloud) | too easy: ~4 tokens localise one rotationally symmetric cup, target fixed |
| `stack_blocks_two_fov` | stock task, both arms, stock head cam, spawn \|x\| ≤ 0.24 (FoV), 1024 pts | 80–94% for every token count; on 200 new scenes g64 91.5% vs g64 arms-cut 92.0% | too easy / token budget never binds |
| `hanging_mug_labeled` | stock task (hand-over, hang mug by handle on rack hook), both arms, stock head cam | 42–61% depending on setting | **current task**: not saturated, detail-dependent |

Hard facts learned on the way: eval is **deterministic per (checkpoint, scene)**
(RoboTwin re-seeds np + torch from the scene seed every episode;
`--noise-offset` re-draws only the policy noise: ±2 pts). Differences between
checkpoints of one run are larger than that noise. 600 epochs is **not converged**
on hanging_mug (all runs +4…17 pts from epoch 300 to 600).

## hanging_mug results (600 epochs, seed 42, 200 paired scenes)

Sweep 1 (4096-point data; cut = oracle label cut after downsampling):

| policy | success | key comparison |
|---|---|---|
| 4096 / 128 | 50.5% | vs 4096/64: 35 vs 18 discordant, p = 0.027 |
| 4096 / 64 | 42.0% | reference |
| 4096 / 64 arms cut [4,5] | **59.0%** | vs 4096/64: 51 vs 17, p < 0.001; vs 4096/128: 37 vs 20, p = 0.03 |
| 4096 / 64 arms+grippers cut | 48.0% | vs 4096/64 n.s. (p = 0.16) — grippers carry information |
| 2048 / 64 | 51.0% | vs 4096/64 n.s. on common scenes (p = 0.09) |
| 1024 / 64 | **61.0%** | vs 4096/64 p = 0.0001; vs 4096/128 p = 0.03 (common scenes) |

Sweep 2 (1024 points; same 200 scenes as 1024/64):

| policy | success | vs same tokens, no cut |
|---|---|---|
| 1024 / 64 arms cut | 44.5% | −16.5 pts, p = 0.0003 (cut **hurts**) |
| 1024 / 32 | 53.5% | (vs 1024/64: p = 0.049) |
| 1024 / 32 arms cut | 49.5% | −4, n.s. |
| 1024 / 16 | 51.5% | (vs 1024/32: n.s.) |
| 1024 / 16 arms cut | 44.0% | −7.5, p = 0.058 |

Caveat: the 4096 / 2048 / 1024 eval configs originally validated their own scene
lists (only ~180 of 200 shared) — cross-density p-values above are recomputed on
the common scenes. Fixed for the future (`scene_seed_group`, one shared list).

## Memorization check on hanging_mug (2026-10-24)

Question (user): is ~50% "near training layouts succeed, the rest fail"? Initial
poses of all 50 training + 229 eval scenes rebuilt with the real `setup_demo(seed)`
(`scripts/sim/dump_scene_layouts.py`, checked against stored mug ids and frame-0
label centroids), joined with per-episode outcomes of the 11 600-epoch / 200-scene
evals (`scripts/sim/scene_distance_analysis.py`; report, json, png in
`R3D/data/outputs/sim/scene_layouts/`).

| finding | number |
|---|---|
| layout space is densely covered | eval → nearest training layout: median 4.1 cm joint, 1.7 cm mug; training LOO spacing 4.4 cm |
| distance does NOT predict failure | AUC(fail farther) joint 0.41–0.57, n.s. in all 11 runs (mug model 5 excluded); nearest 10% of scenes 30–75% success ≈ overall |
| unseen mug mesh fails | mug model 5 has 0 training scenes → 0% in 10/11 runs (10 eval scenes) |
| outcomes are scene-specific | 170 common scenes: var of solve count 12.1 vs 2.7 if independent; 17 never / 9 always solved |
| mug shape matters | mug model explains 20% of per-scene solve variance (models 1/6 hardest); handle angle AUC 0.60–0.64 in 7/11 runs |
| arms cut helps on hard shapes | 4096/64 → cut45: mug 1 8 → 44%, mug 6 20 → 55% (small n) |

Reading: no evidence of local layout memorization, but the test has little power
because the layout space is tiny (every eval scene is within a few cm of training).
What is hard is mug SHAPE (and handle angle), not position. Supports moving to a
benchmark with wide layouts (proposed: MimicGen D1/D2 over LIBERO -- LIBERO's
regions are ~±2.5 cm and LIBERO-PRO shows VLAs collapse to ~0% under position
shifts); run the same distance analysis there.

## Next task: MimicGen Square D2 (set up 2026-10-24, NOT trained yet)

Why: hanging_mug's layouts are tiny; Square D2 randomizes peg AND nut over 0.5 x 0.5 m
each (nut yaw random), demos are generated (1000, `core/square_d2.hdf5`, 153k frames,
mean 153/episode), and DP3 at 1024 pts is near the floor there (EquiDiff: 7/6/19% at
100/200/1000 demos vs voxel 39/48/63) -- headroom where the 3D representation binds.
Decisions (user, 2026-10-24): **200 demos** (demo_0..199, 30,646 frames) -- with 1000
the data alone may generalize and shrink the token effect. **Training budget follows
EquiDiff** (diffusion, same horizon 16 / obs 2 / act 8, same tasks): 50000 / n_demo
epochs = 250 epochs at batch 128 (= 64 x 2 accum here) ~ 60k optimizer steps, 7.7M
samples. **Report the FINAL checkpoint only** -- no max-over-training (EquiDiff and
MimicGen report the peak, so their numbers are not comparable to ours); a mid
checkpoint is evaluated only to check convergence. Budgets in the papers' code:
robomimic/MimicGen low-dim 200k steps x 100 = 20M samples, MimicGen image 300k x 16 =
4.8M; both fixed regardless of dataset size.

Setup: env `r3d_robosuite` (cp -a of r3d, paths rewritten; MuJoCo 2.3.2, robosuite
b9d8d3d, robomimic d0b37cf, mimicgen c0fbff2, all `--no-deps` so torch stays 2.4.1).
Sources in `third_party/` (git-excluded). Cloud: `r3d/common/robosuite_pcd.py`
(shared by conversion and eval): agentview 512 px, depth + per-geom segmentation,
box x[-0.40,0.40] y[-0.55,0.55] z[table+3 mm, +0.5 m], FPS deterministic start
(prefix = smaller budgets). Labels in the RoboTwin scheme (arm=5, gripper=7, nut=8,
peg=9) so `drop_point_labels=[5]` is the arm cut.

View check (`scripts/sim/mimicgen_view_check.py`, 5 demos, 800 frames):

| | arm | gripper | nut | peg |
|---|---|---|---|---|
| tokens @ 4096/128 | 69.7 | 25.1 | 15.3 | 16.8 |
| tokens @ 4096/64 (coverage) | 35.2 (0.46) | 12.9 (0.56) | 7.5 (0.45) | 7.6 (0.58) |
| tokens @ 1024/64 (coverage) | 35.4 (0.99) | 12.9 (1.0) | 7.5 (1.0) | 7.5 (1.0) |

Robot = ~75% of tokens (RoboTwin hanging_mug: 76%). At 4096/64 coverage is the
bottleneck (the regime where the arms cut gave +17 on hanging_mug). 18.7k points in
the box per frame (min 4.6k). Gotchas found: robosuite renders OpenGL-flipped
(flipud before unprojecting; nut centroid within 3-20 mm of the body); widening x
toward the robot lets the pedestal in (12/128 tokens) -> x stays at the table edge;
silhouette pixels carry foreground depth + background label -> in-box TABLE /
BACKGROUND points dropped (97-100% sit on > 2 cm depth jumps); the end effector
leaves agentview's frustum in 0.09% of frames (17/1000 episodes) -- accepted.
~147 ms/frame single process -> full conversion needs parallel workers.

### Sweep 1 (launched 2026-10-24 16:28, driver `scripts/sim/run_mimicgen_sweep.sh`)

User decisions: **1024 points is the default** (DP3 / EquiDiff's point-cloud baseline used
1024; "the fewer points, the stronger the claim" -- raise only if 1024 is the limit).
Grid at **32 / 16 / 8 points per token** for every density: 1024: 32/64/128 tokens,
2048: 64/128/256, 4096: 128/256/512 (same coverage ratio per row -> density varies patch
size, not coverage). Batch 128 (EquiDiff), 251 epochs, ckpts 125 + 250; final evaluated
on 200 scenes (seeds 100000..100199), mid on the first 100 (convergence only). Arm cut
(`EXTRA="task.drop_point_labels=[5]" TAGSUFFIX=-cut5`) decided after the 1024 results.

Data check (`scripts/sim/plot_mimicgen_layouts.py`, `sim_view/square_d2/layouts.png`):
200 demos and 200 eval scenes both fill nut/peg x,y in [-0.25, 0.25] m and nut yaw over
360 deg; eval scene -> nearest demo (nut + peg position) median 8.6 cm (p10 5.1, p90 12.8);
the position-nearest demo's nut yaw differs by 84 deg (median). hanging_mug: 4.1 cm.
Conversion check: absolute actions replayed open loop succeed 19/20 (original delta
actions 18/20 -- the misses are contact sensitivity, not the conversion), eef tracking
0.5-1.6 mm. Eval seeding verified (identical layouts on repeat). Eval: ~28 s per full
400-step episode per worker.

Measured on the 4090 (micro-batch it/s, 449 micro-batches of 64 per epoch): 1024/128 at
64: 11.0 it/s (2.9 h); 2048/256 at 64: 6.5 it/s, 20.4 GB (4.8 h); 4096/256 at 64:
5.9 it/s, 19.3 GB (5.3 h); 4096/512 OOMs at 64 -> 32 x 4: 4.9 it/s, 20.4 GB (12.7 h).
Whole sweep ~41 h training + ~4 h eval.
**1024 results (2026-10-25, final ckpt, 200 scenes):** g128 20.5% | g64 23.0% | g32 19.0%;
token count n.s. (g64 vs g128 27 vs 22 discordant p=0.57; vs g32 27 vs 19 p=0.30).
Converged: 125 -> 250 equal on the first 100 scenes (gained = lost). 120/200 scenes solved
by none, 10 by all. Distance to nearest demo MATTERS here (unlike hanging_mug): pooled
solve fraction 0.28 / 0.23 / 0.11 for nearest-demo terciles (< 7.3 / 7.3-10 / > 10 cm);
AUC 0.70 (g128), 0.57 (g64, p=0.08), 0.67 (g32). Baseline below the 30-40% target.
Eval: ~30 min per 200 scenes with 5 workers.

**2048 results (2026-10-25):** g256 16.5% | g128 15.5% | g64 14.0% (within 2048 n.s.).
2048 < 1024 everywhere: same points/token 1024/64 vs 2048/128 28 vs 13 discordant p=0.028,
1024/32 vs 2048/64 p=0.14, 1024/128 vs 2048/256 p=0.31; same tokens 64: 23.0 vs 14.0%,
30 vs 12, p=0.008. Pooled 20.8% vs 15.3%. Replicates hanging_mug: fewer points win;
1024 is not the bottleneck -- the baseline (~20%) is below the 30-40% band and tokens do
not lift it. Eval ~45 min per 200 scenes at 2048 (30 at 1024).

**Decision 2026-10-25 13:20 (user): 4096/512 etc. dropped; TOKENS FIXED AT 64** (as k is
fixed at 32 -- the token sweep only served to pick the count). Axes from now on: point
count {1024, 2048, 4096} x arm cut {none, [5]}, seed 42, no seed repeats until the clean
paper runs. Storyline: the learned method is compared against (a) the arm cut = hand-made
ALLOCATION baseline, (b) the uniform densities over the same area = hand-made DENSITY
baseline. Note: at fixed 64 x 32 = 2048 patch slots the point count moves patch size AND
coverage together (2x / 1x / 0.5x) -- the trade-off the learned method has to navigate.
Queue (`scripts/sim/run_mimicgen_queue.sh`, file `outputs/sim/mimicgen/queue.txt`, re-read
before every run): 1024/64 cut, 4096/64, 4096/64 cut, 2048/64 cut, then 1024/64 and
1024/64 cut at TRAINING seed 43. Only the 250 ckpt is evaluated (200 scenes) from now on.
2048/64 mid ckpt: 17%. `training.seed` (train.py seeds torch / numpy / random BEFORE the
model is built) covers all training randomness: init of the non-pretrained layers,
shuffling, augmentation, diffusion noise/timesteps, the encoder's random FPS start. The
train/val split (`task.dataset.seed: 42`, 4 of 200 demos held out) and the eval scenes are
fixed on purpose, so seeds pair on identical data and scenes.

**64-token grid results (2026-10-26, seed 42, final ckpt, 200 paired scenes):**

| | 1024 | 2048 | 4096 |
|---|---|---|---|
| no cut | 23.0% | 14.0% | 8.5% |
| arm cut [5] | 17.5% (31 vs 20, p=0.16) | 19.0% (+5, 16 vs 26, p=0.16) | **20.0% (+11.5, 13 vs 36, p=0.001)** |

Seed 43, 1024/64 no cut: 19.0% (vs 23.0% at seed 42: 29 vs 21, p=0.32) -> training-seed
noise ~4 points. Replicates hanging_mug's regime finding on a second task: the cut HELPS
where coverage is the bottleneck (4096/64, nut coverage ~46%) and does not help (trend:
hurts) at full coverage (1024). With the cut, density no longer matters (17.5 / 19.0 /
20.0%); the best policy is still 1024/64 no cut (23.0%), and 4096 cut vs 1024 no cut is
n.s. (27 vs 21, p=0.47) -- the "why not just 1024" question remains for the learned method.

Seed 43, 1024/64 cut: 20.5% (vs no cut seed 43 19.0%: 27 vs 24, p=0.78). Over 2 seeds at
1024/64: no cut 21.0% vs cut 19.0% (55 vs 47 scene-seed pairs, p=0.49) -> at 1024 the cut
is NEUTRAL, not harmful (seed 42's -5.5 was noise). Same scene-level spread between seeds
of one config (cut 17.5 vs 20.5%: 23 vs 29 discordant) -> runs differ on ~25% of scenes.

Seed 43, 4096/64 cut: 16.5% (seed 42: 20.0%; 29 vs 22, p=0.40) -> mean 18.2%. vs 4096 no
cut (8.5%, seed 42 only): 26 vs 10, p=0.011 -> the cut gain at 4096 holds on both seeds
(+11.5 / +8.0). 2 seeds each: 1024 no cut 21.0% vs 4096 cut 18.2% (54 vs 43, p=0.31) --
4096 + cut does not beat plain 1024; roughly level, slightly below. Queue finished
2026-10-26 15:00; GPU free.

Working model (user + me, 2026-10-26): at fixed tokens, more points only lower coverage;
reallocation (cut) raises coverage of the relevant parts until it matches an uncut policy
of similar coverage; at ~full coverage reallocation and patch size do not matter (on
these tasks, where generalization not detail limits success -- a fine-geometry task may
differ). Open: 2048/128 has ~99% nut / 100% peg coverage like 1024 cut but 15.5%
(single seed) -- coverage does not explain it; noise or "removal > adding tokens".

Coverage after the cut (64 tokens, `token_allocation` on stored frames): 4096 cut keeps
1710 pts, nut cov 83% / peg 93%, patches 2.2-2.5 cm, 14.6 / 16.1 tokens on nut / peg
(vs 6.7 uncut); 2048 cut 99-100%, 3.1 cm; 1024 cut 100%, 4.3 cm, 14.9 / 16.0 tokens.
1024 uncut: 100%, 4.5 cm, 6.8 / 6.6 tokens.

Conclusion discussed with the user (2026-10-26): (1) coverage of task-relevant parts
matters most (4096: 48 -> 83% coverage via cut: 8.5 -> 20%); (2) once coverage is full,
more / overlapping tokens add nothing (1024 token sweep; 1024 cut doubles object tokens,
no gain); (3) patch size shows no benefit on this task at 200 demos (4.5 cm ~ 2.2 cm);
(4) NOT "only coverage": reaching ~99% coverage by MORE TOKENS (2048/128: 15.5%) did
less than by REMOVING the arm (2048/64 cut: 19.0%), and 2048 uncut (85%, 14%) vs 4096 cut
(83%, 20%) differ at similar coverage -- same direction as hanging_mug's open puzzle
(4096/64 cut 59% > 4096/128 50.5% despite less handle coverage). Each comparison within
seed noise (~4 pts); worth a clean multi-seed test (matched coverage: cut vs extra tokens).

**4096 runs were provisional** (user, 2026-10-24): may be dropped after the 1024 / 2048
results -- only on the user's explicit word. To drop: stop the driver after the 2048 evals
(4096/512 starts ~Sun 13:30); editing the script does not affect the running loop.

## Mechanism — what we learned

1. **FPS allocates tokens by surface extent, not density.** Same surface ⇒ same
   token centres for any point count (handle tokens at 64: 2.15/2.11/2.07/2.09 for
   1024/2048/4096/8192 pts). Only removing surface changes allocation.
2. **Density sets the token size** (patch = 32 nearest points): hook patch radius
   7.7 / 6.1 / 3.5 / 1.5 cm for 1024 / 2048 / 4096 / 8192 pts. With fixed k, more
   density ⇒ smaller patches ⇒ lower coverage (64 tokens: 99 / 80 / 47 / 25%).
   Scaling k with density keeps patch size (4096 pts, k = 128 ≈ 1024 pts, k = 32:
   4.5 vs 4.3 cm) — a clean future experiment to isolate *detail inside a token*.
3. **Reallocation pays off when patches are small and coverage is the
   bottleneck** (4096/64: +17), **not with large patches** (1024: cut neutral or
   harmful — coverage matters little there, 16 ≈ 32 tokens, and the arm points
   carry information). ⇒ a fixed cut fails in one regime or the other ⇒
   motivates the *learned* removal.
4. Open puzzle: 4096/64 arms cut (59%) beats 4096/128 no cut (50.5%) despite
   less handle coverage (78 vs 94%) and fewer handle tokens (2.9 vs 4.6) — so
   neither coverage nor tokens-on-part alone explain the 4096 gain. Single seed.
5. **Attention** (`scripts/sim/attention_by_class.py`): averaged over heads,
   the action head's cross-attention follows token counts (~26% on arm tokens in
   baselines). Per head: dedicated arm heads exist (e.g. L2H0 36–38% on left arm,
   2.6× its token share); with the cut those heads move to gripper/mug (63% on
   the grasping gripper). Sharp hook focus did NOT predict success.
6. The cut changes **allocation only** — density on kept parts (patch size) is
   unchanged. A learned method affects effective k only if it **thins** regions
   (bigger patches) or acts **before** the final downsampling (denser kept parts).

## Real robot (R1 / K data, measured on stored clouds)

Fused 8192 pts, 512 tokens: ~half the tokens on robot points (top cam 73% robot).
Wrist views near the contact are dense (22–24 pts/token, **1.2–1.3 cm patches**)
with the lowest coverage (85–88%; top view 99%). R1's cut raises wrist coverage
to 96–97% and wrist-scene tokens +60%; K's +56 tokens add only ~15 there. The
user reports R1 ≈ K on the robot ⇒ above ~90% coverage more tokens mostly add
overlap (saturation). Supports "small patches + imperfect coverage" as the
regime, but coverage was not "bad".

## Learned link remover v0 (built 2026-10-26)

Design and tunable knobs: `docs/learned_remover_design.md`; brief: `docs/learned_remover_v0.md`.

**Labels.** Every Square D2 frame was re-rendered from its hdf5 state with per-point link cells
(22 cells; the gripper is split geometrically in its mesh frames). The cloud and labels are
bit-identical to the stored zarr (200/200 demos). Kinematics-based labels (nearest mesh
surface, sampled at 1.5 mm) vs segmentation: 99.9% of robot points within 5 mm, 98.7% the
same cell; 0.7% of nut/peg points come within 5 mm of a robot mesh.

**Per-link statistics** (mean points per frame, N = 1024 / 4096): link5 207 / 825, link6
261 / 1083, link7 114 / 457, hand f1side / centre / f2side 67 / 46 / 77 (×4 at 4096),
finger 9.5 / 33 each. link0–4 total ~9 / 29, mostly silhouette mislabels or the elbow in ~5%
of frames. Robot share 78%. On 1.2% / 0.24% of frames (1024 / 4096) fewer than 64 non-robot
points remain.

**FPS thinning does not reallocate tokens** (64 tokens, link5–7 thinned to ρ, 511 frames):

| ρ arm (N = 4096) | 1 | 0.5 | 0.2 | 0.05 | 0.02 | 0 |
|---|---|---|---|---|---|---|
| FPS thinning: arm tokens / nut coverage | 34.6 / 0.49 | 34.2 / 0.49 | 33.3 / 0.50 | 30.8 / 0.53 | 28.3 / 0.55 | 0 / 0.81 |
| nearest-to-EE thinning: arm tokens / nut coverage | 34.5 / 0.48 | 27.0 / 0.58 | 17.2 / 0.69 | 8.9 / 0.76 | 6.2 / 0.78 | 0 / 0.81 |

Under FPS thinning the arm becomes a few huge patches (2 → 14 cm) and holds its tokens; only
complete removal frees them. **Decision (user): removal gate + FPS density** (not
nearest-to-EE, which would bake in a "near the gripper first" prior).

**Toy REINFORCE** (synthetic reward with the measured cliff; region sizes from the stats):
- The brief's lr 1e-5 does not move φ.
- Adam 3e-2 learns "remove link6, keep the fingers" within ~150 steps even at 10× noise,
  but closes 0–3 neutral gates by noise over a run.
- **3e-3 with 128 windows × 4 masks** learns it by step ~1350–2000 of ~9.8k with no neutral
  gate closed and drift ≤ 0.9 logit.
- Key insight: under Adam a zero-signal parameter drifts by ~lr per step **whatever the noise
  level**, so less noise buys learning speed and only a lower lr buys less drift.

**Cost** (4090, bf16): the θ step with the remover is +2% (1024) / +20% (4096). The reward
pass at 128 × 4 is 184 / 259 ms per φ step, so a run takes ~2.5 h at 1024 and ~4.4 h at 4096
(vs 1.9 / 3.3 h).

**First sweep (launched 2026-10-26):** N ∈ {4096, 1024, 2048}, 64 tokens, seed 42, defaults
(G), checkpoints 50–250 all evaluated on the 200 scenes. Driver
`scripts/sim/run_remover_sweep.sh`, queue `R3D/data/outputs/sim/mimicgen/remover_queue.txt`,
reports `<run>/analysis/report.md` and `R3D/data/outputs/sim/mimicgen/remover_sweep/`.
Purpose: as many diagnostics as possible to set up the tuning sweep.

## Method plan and positioning

**Implementation brief for the learned remover: `docs/learned_remover_v0.md`** (user's plan,
2026-10-26: per-link logit-normal density, REINFORCE with leave-one-out baseline, joint
θ/φ training, step-0 toy test before any GPU run). It supersedes the sketch below where
they differ.

1. **First learned version:** removal restricted to **proprio-known points**,
   applied **after** the final downsampling — the learned counterpart of the
   oracle cut (oracle = upper bound). Needs a way around non-differentiable
   selection (straight-through / Gumbel relaxation, score-function gradient, or
   soft weighting in the patch pooling).
2. Then **a task where fine detail genuinely needs a dense cloud** (hanging_mug
   cannot show it: fewer points + full coverage won). Also answers the reviewer
   question "why not just use 1024 points?" (1024/64 no cut is the best policy).
3. Later: density learning over the whole scene (start ~8k pts, learn effective
   density) — but that conflicts with "only proprio points removed".
4. **Baselines to include:** density-weighted FPS (strong on the real rig, where
   density and relevance correlate near the wrist cams); tuned / learnable k or
   multi-scale grouping; the oracle cut.
5. Related work to position against (check properly): Learning to Sample
   (Dovrat 2019), SampleNet (Lang 2020), PointASNL (Yan 2020), APES (Wu 2023),
   token pruning/merging (DynamicViT 2021, ToMe 2023). Differentiators: trained
   on the action loss; physically grounded prior (proprio-known points); shown
   that a fixed rule fails across regimes.
6. Rack base-plate crop (floor 0.75 m removes 47% of rack points, 5% of the mug)
   or a slim custom hook — options to make the reallocation effect more visible.

## Engineering pitfalls worth remembering

- Run trainings **sequentially**: two runs sharing the 4090 are slower in total
  (5.5 vs 16.6 it/s per run).
- `EPOCHS=601` (not 600): train.py saves at epoch % interval over 0..N-1.
- Each checkpoint is 2.15 GB — keep `CHECKPOINT_EVERY=300` on long sweeps.
- `pkill -f <pattern>` from a tool shell matches its own command line.
- YAML: `shape: [${task.num_points}, 6]` does not parse — quote the interpolation.
- RoboTwin collection needs `envs/` on PYTHONPATH (`scripts/sim/collect.sh`).
- `seeds_list/` (validated eval scenes) is gitignored — copy it to other machines.

## Where things are

| what | path |
|---|---|
| driver logs + results | `R3D/data/outputs/sim/mug_sweep_driver.log`, `mug_sweep2_driver.log` |
| per-checkpoint evals | `<run>/checkpoints/600.eval_200ep_seed1.json` |
| eval videos | `R3D/r3d/env/robotwin2/eval_video/` |
| cloud viewers (HTML) | `R3D/data/outputs/sim_view/clouds/` |
| density / attention studies | `R3D/data/outputs/sim_view_dense/hanging_mug_labeled/*.json` |
| real-data local coverage | `~/R3D-Policy/R3D/data/outputs/robot_cut/local_coverage_s20.json` |
