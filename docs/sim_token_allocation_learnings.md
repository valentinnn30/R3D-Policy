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

**Sweep 1 result (2026-10-27; 200 paired scenes, seed 42).** No gate closed at any N
(keep prob 0.97–0.99 at the end); ρ drifted from 0.8 to 0.6–0.78. At inference the cloud was
therefore uncut except for thinner robot regions, and thinning moves no tokens
(4096: link5/6/7 11.6/17.8/6.6 tokens uncut vs 11.6/17.7/6.4).

| N | 50 | 100 | 150 | 200 | 250 | uncut s42 | arm cut s42 |
|---|---|---|---|---|---|---|---|
| 4096 | 8.5 | 12.0 | 13.5 | 11.5 | 12.5 | 8.5 (p = 0.19) | 20.0 (p = 0.05) |
| 1024 | 16.0 | 23.5 | – | 26.5 | 24.0 | 23.0 (p = 0.88) | 17.5 |
| 2048 | – | – | – | 19.0 | 19.5 | 14.0 | 19.0 |

- **The removal effect is positive (removal raises the loss) for every region**, from the
  first φ steps on. It grows ~3× as removals get rarer (4096 link6: 1.3 → 3.7e-4 while closed
  draws per step fall 58 → 3.5). That is the co-adaptation loop: θ sees fewer removals, gets
  worse at them, and the gates rise further.
- **The density slope is weak but has a consistent sign** (22 of 24 region × N positive,
  t ≈ 2–3): slightly sparser is preferred, at ~2–3% of the size of a removal effect. Success
  shows no "sparser is better" (1024 remover = uncut).
- The fine-tune phase could not help: with every gate open, θ was fine-tuned on an
  unreallocated token layout (user's reading, confirmed by the mechanism table).

**Tuning round 1: gates only (2026-10-28; `learn_density: false`, 4096, seed 42).**

| run | gates at the end | 200 | 250 | vs uncut at 250 |
|---|---|---|---|---|
| GO (gates only) | 0.98–0.99, all open | 12.0 | 8.5 | 15 vs 15, p = 1.0 |
| GOgm2 (gates only, gate_max 2) | 0.88 = the bound, all open | 13.0 | 10.0 | 14 vs 11, p = 0.69 |

With θ still seeing each region removed in ≥ 12% of draws (GOgm2), the reward says removal
hurts for every region, now very clearly (t = 12–22, link6 +1.2e-4 vs +1.75e-4 in GO).
**Conclusion: the reward of one shared θ is the obstacle**, not the density variable and not
too little exploration. User's framing: removal is out of distribution for the shared
network, and training one network on very different input types is worse than specialising.

**Validation loss of networks trained on their own fixed cloud** (last 25 epochs, ×1e-3):
the sign flips relative to the shared-θ reward. 4096 uncut 14.51 vs arm cut 11.66 (seed 43:
11.59); success 8.5 vs 20.0. Over 12 Square runs, Spearman(success, −val loss) = 0.63. Loss
gets the large 4096 effect right and is mixed at 1024 (favours the cut in both seeds; success
−5.5 / +1.5 points). Seed-to-seed noise in the loss (1024 uncut 11.5 vs 12.9) is far larger
than its epoch-to-epoch SE. Both 4096 baselines overfit in the second half (uncut 11.9 → 14.5,
cut 10.6 → 11.7, from epoch 125 to 250).

**Branch test (2026-10-28, `scripts/sim/run_branch_test.sh`):** the uncut 4096 run resumed
from its 125.ckpt with the arm cut on the same schedule.

| epoch | 150 | 175 | 200 | 250 | success 175 / 250 |
|---|---|---|---|---|---|
| uncut | 12.47 | 13.20 | 13.99 | 14.51 | – / 8.5 |
| arm cut from 125 | 11.30 | 11.75 | 12.84 | 13.75 | 8.5 / 13.0 |
| arm cut from scratch | 10.44 | 10.94 | 11.47 | 11.66 | – / 20.0 |

The ranking is right from the first loss window on; the level is not. After a late branch
the network keeps overfitting like the uncut run and recovers about half of the benefit
(vs uncut 16 vs 7, p = 0.09; vs the full cut 16 vs 30, p = 0.05), and 50 epochs are not
visible in success.

**Early branch (2026-10-29, `scripts/sim/run_early_branch.sh`, report
`R3D/data/outputs/sim/mimicgen/early_branch/`):** the same with the branch at epoch 50. The
uncut trunk reproduced the baseline's val loss for epochs 0–50 bit for bit (training is
deterministic), so the baseline is the matched control.

| val loss ×1e-3 (last 25 ep) | 75 | 100 | 150 | 200 | 250 | success |
|---|---|---|---|---|---|---|
| uncut | 12.23 | 11.87 | 12.47 | 13.99 | 14.51 | 250: 8.5 |
| arm cut from scratch | 10.95 | 10.29 | 10.44 | 11.47 | 11.66 | 250: 20.0 |
| branch at 125 | – | – | 11.30 | 12.84 | 13.75 | 175: 8.5, 250: 13.0 |
| **branch at 50** | 10.92 | 10.91 | 11.18 | 12.43 | 13.28 | 150: 11.0, 250: 10.0 |

25 epochs after the branch the loss equals the cut from scratch; after that it rises faster and
ends between the two. Success at 250: vs uncut 15 vs 12 (p = 0.70), vs the cut from scratch
13 vs 33 (p = 0.005), vs the branch at 125 15 vs 21 (p = 0.41). The early branch did NOT
recover the cut's benefit, and did no better than the late one.

**The resume is not exact, but shows no bias.** Trunk 50.ckpt → uncut continuation, 25 epochs:
val loss differs from the baseline from epoch 51 on (0/25 identical, per-epoch differences up
to 2.4e-3), with equal means (12.27 vs 12.23 ×1e-3). So a resumed run is a different noise
realisation of the remaining training, comparable to a seed change, and every branch result
carries seed-level variance. Square success varies a lot between seeds (arm cut 20.0 / 16.5;
1024 uncut 23.0 / 19.0), and the 20.0% reference is a single realisation. **Open:** is "branches
fall short of the cut from scratch" real, or seed noise? It needs seed repeats (e.g. uncut 4096
seed 43; a second resume of each branch) before branch-and-prune is built on it.
The arm + gripper cut cannot be a candidate at 4096: it leaves as few as 38 points (< 64 tokens).

**Tuning round 2 (launched 2026-10-31, queue `R3D/data/outputs/sim/mimicgen/remover_tune2_queue.txt`,
report `.../mimicgen/remover_tune2/`).** Four runs at 4096 / 64, seed 42, lr 3e-3, schedule 20 / 70 / 10:

| tag | what | question |
|---|---|---|
| GOclosed | gates only, init_keep 0.2 (mirror of GO) | does the start decide? a gate opening against it = the region is needed |
| GObal | gates only, policy pass at a fixed keep 0.5 (`theta_keep`), φ learns | is the bias only how often θ sees each mask? |
| DOscene | density only, 8 robot regions + a label-free scene region (floor ρ 0.1), init ρ 0.9 | can the remover learn "fewer points" (uncut 1024 23.0% vs uncut 4096 8.5%)? |
| GOmix | GO + ε = 0.25 structured masks (none, arm, arm + hand, all) | does the reward rate whole-arm removal better when it is actually sampled? |

- **Fill-up is not an issue for the remover.** With all 8 regions closed, a frame keeps at
  least 56 points (p1 282, median 891); only 3 of 30,646 frames (0.01%) fall below 64. The
  remover never thins link0–4, unlike the label-5 cut (38).
- **Why the scene region:** robot-only density cannot reproduce the 1024 effect, which is
  coverage of the objects. Their density stays at 4096 when only the robot is thinned.

**Round 2 result (finished 2026-11-01; 200 paired scenes, seed 42; report
`R3D/data/outputs/sim/mimicgen/remover_tune2/sweep_report.md`):**

| run | learned mask at inference | 200 | 250 | vs uncut 8.5 | vs arm cut 20.0 | vs uncut 1024 23.0 |
|---|---|---|---|---|---|---|
| GO (round 1) | all kept | 12.0 | 8.5 | p = 1.0 | 13 vs 36, p < 0.001 | – |
| GOclosed | whole robot removed | 18.5 | 14.0 | 18 vs 7, p = 0.04 | 19 vs 31, p = 0.12 | p = 0.015 |
| **GObal** | **link5 + link6 removed**; link7, hand, fingers kept | **22.0** | **19.5** | **30 vs 8, p < 0.001** | **25 vs 26, p = 1.0** | p = 0.40 |
| DOscene | scene ρ 0.25, robot ρ 0.5–0.88 | 16.0 | 13.0 | p = 0.11 | p = 0.054 | 11 vs 31, p = 0.003 |
| GOmix | link5 + link6 removed (same mask as GObal) | 18.5 | 16.5 | 26 vs 10, p = 0.011 | p = 0.43 | p = 0.12 |

- **GOclosed** is the mirror of GO, as the user predicted. Every gate closed further
  (0.007–0.04) and the reward said adding hurts for every region (t −6 to −18). The start
  decides; no region opened against it.
- **GObal: the bias was how often θ saw each mask.** With θ's training pass fixed at keep 0.5,
  the shared reward separates the regions:
  - remove link5 / link6: t −9.7 / −9.2;
  - keep link7: t +8.6;
  - keep the hand regions: t +7.6 to +8.6;
  - keep the fingers: t +5.7 / +7.8.

  The gates move from epoch 75 on. The learned mask refines the oracle: it keeps link7, the
  wrist link next to the gripper. It matches the oracle's success, and the freed tokens go to
  the task (nut 5.8 → 9.5 tokens, coverage 0.58 → 0.80; peg 6.8 → 14.6, coverage 0.47 → 0.85).
- **GOmix reached the same mask independently.**
  - **Per-mask verdict** (mean advantage against the rest of the window):
    - arm removed (link5–7) −9.5e-5, t −24: better;
    - nothing removed: ~0 in the second half;
    - arm + hand removed +2.0e-4, t +31: worse;
    - everything removed +7.7e-4, t +79: much worse.
  - **Its per-region effects are confounded** by the bad structured masks: they all read
    positive, yet link5 and link6 closed. Read only the per-mask table for a mix run.
- **GObal vs GOmix is within noise** (27 vs 20 at 200, p = 0.38; 27 vs 21 at 250, p = 0.47).
  It is NOT fine-tune length: both fine-tune epochs 226–250. GOmix's θ saw the exact final
  mask more often (32.8% of its training clouds at epoch 200 vs 0.4% for GObal), and its val
  loss on the final mask is lower (11.41 vs 12.12 ×1e-3). Possible but untested: GObal's 50%
  robot-part dropout makes the policy more robust in closed loop.
- **DOscene learned the "fewer points" density, but not the success.**
  - The scene ρ fell 0.9 → 0.25 and was still falling when φ froze (0.30 at 175, 0.29 at 200,
    0.25 at 224).
  - The reward favoured sparser everywhere (scene density slope t 8.4, fingers 6.2 / 4.4).
  - Object coverage reached the 1024 regime (nut 0.99, peg 1.00, patches 4.5 cm).
  - Success stayed at 13.0%, against 23.0% for uncut 1024 (p = 0.003). The robot stayed dense
    (still ~50 of 64 tokens).
- **Fill-up is negligible:** at most 0.01% of clouds in any run.

**Next:** round 3, spec in `docs/remover_round3_plan.md`. Seeds over more eval scenes (user,
2026-11-01): seed noise dominates (the arm cut alone is 20.0 vs 16.5% between seeds).

**Round 3 (finished 2026-11-02; 4096 / 64 tokens, 200 paired scenes; queue
`R3D/data/outputs/sim/mimicgen/remover_tune3_queue.txt`, report `.../mimicgen/remover_tune3/`).**
Runs: GObal seed 43, and **SPEC** seeds 42 / 43 -- a specialist trained from scratch on the FIXED mask GObal
learned (link5 + link6 removed, everything else kept, nothing learned; `policy.link_remover.fixed_removed`,
see the design doc). Afterwards (user, same day) GObal's 250.ckpt was re-evaluated with OTHER masks at
inference (`eval_mimicgen_ckpt.py --remover-removed`, `scripts/sim/run_mask_override_evals.sh`).

| method | s42 @200 | s43 @200 | s42 @250 | s43 @250 | mean @250 | spread @250 |
|---|---|---|---|---|---|---|
| GObal (learned mask) | 22.0 | 20.5 | 19.5 | 18.0 | 18.8 | 1.5 |
| SPEC (fixed mask, from scratch) | 14.5 | 8.0 | 11.0 | 11.5 | 11.3 | 0.5 |
| arm cut (oracle) | – | – | 20.0 | 16.5 | 18.3 | 3.5 |
| uncut | – | – | 8.5 | – | – | – |
| GObal 250.ckpt, **uncut cloud** at inference | | | 16.0 | 18.0 | 17.0 | |
| GObal 250.ckpt, link5 + 6 + 7 removed at inference | | | 18.0 | 18.0 | 18.0 | |

Paired McNemar (a only / b only; pooled = discordant pairs summed over both seeds, 400 scene-runs):

| a vs b | s42 | s43 | pooled |
|---|---|---|---|
| GObal vs SPEC @200 | 31 / 16, p = 0.040 | 34 / 9, p = 0.0002 | 65 / 25, p = 3e-5 |
| GObal vs SPEC @250 | 32 / 15, p = 0.019 | 28 / 15, p = 0.066 | 60 / 30, p = 0.002 |
| GObal vs arm cut @250 | 25 / 26, p = 1.0 | 25 / 22, p = 0.77 | 50 / 48, p = 0.92 |
| SPEC vs arm cut @250 | 14 / 32, p = 0.011 | 14 / 24, p = 0.14 | 28 / 56, p = 0.003 |
| GObal vs uncut @250 | 30 / 8, p < 0.001 | – | – |
| GObal uncut cloud vs GObal learned mask | 17 / 24, p = 0.35 | 15 / 15, p = 1.0 | 32 / 39, p = 0.48 |
| GObal uncut cloud vs uncut baseline | 26 / 11, p = 0.020 | – | – |
| GObal uncut cloud vs SPEC | 25 / 15, p = 0.15 | 28 / 15, p = 0.066 | 53 / 30, p = 0.015 |
| GObal link5-7 removed vs GObal learned mask | 16 / 19, p = 0.74 | 10 / 10, p = 1.0 | 26 / 29, p = 0.79 |
| GObal link5-7 removed vs arm cut | 28 / 32, p = 0.70 | 22 / 19, p = 0.76 | 50 / 51, p = 1.0 |

- **GObal reproduces.** Seed 43 learned the same mask: keep prob link5 0.11, link6 0.07; link7 0.80, hand
  0.89-0.93, fingers 0.82-0.84 (seed 42: 0.07 / 0.08 vs 0.82-0.91). The reward's per-region verdict agrees
  in sign on every region (removal effect t: link5 -7.4 / link6 -9.7; link7 +5.0, hand +8.4..+10.4,
  fingers +6.2 / +6.4; seed 42 in round 2). Gates (init 0.5, phi from epoch 50) leave 0.4-0.6 at
  epochs 55-98 (link5 / link6: 76 / 86 at s42, 98 / 92 at s43) and stay on their final side from
  epoch 50-70. Mechanism identical on both seeds and in SPEC: nut 5.8 -> 9.5 tokens (coverage 0.58 -> 0.80),
  peg 6.8 -> 14.6 (0.47 -> 0.85).
- **GObal ≈ the oracle arm cut on both seeds** (pooled p = 0.92) and > uncut (s42). GObal is the most
  seed-stable run (1.5 points at both checkpoints).
- **GObal ≈ SPEC is FALSE: the specialist is clearly worse** (~7-10 points, both seeds, both checkpoints;
  pooled p = 3e-5 at 200, 0.002 at 250). SPEC sits at the uncut level (s42: 11.0 vs 8.5, p = 0.49),
  although its token allocation equals GObal's. Its val loss (last 25 epochs, own mask) does not explain
  it: 11.18 (s42, the lowest of any 4096 run) vs 13.66 (s43); GObal 12.12 / 11.89 (its last 25 epochs =
  the fine-tune on the final mask), arm cut 11.66 / 11.59.
- **The gain sits in the dropout-trained NETWORK, not in the mask.** GObal's 250.ckpt on the UNCUT cloud
  keeps its success (16.0 / 18.0 vs 19.5 / 18.0 with its mask, pooled p = 0.48) and beats the uncut
  baseline on the identical input (s42 16.0 vs 8.5, p = 0.02); it also matches the arm cut with nothing
  removed. Removing link7 as well changes nothing (18.0 / 18.0), so link7 does NOT explain SPEC either.
  Together: training theta on random robot-part dropout (keep 0.5 per region) makes the policy better on
  every one of these clouds; the learned mask adds no measurable inference-time gain. Note GObal also had
  the 25-epoch fine-tune on its mask, and seed 43 has no uncut baseline.
- Diagnostics (phi phase mean): held-out mask-rank agreement ~0 (0.03 / -0.07) against a same-window floor
  of 0.27 / 0.30, diagnostic B 0.73 / 0.68 -- the training-window reward ranks masks unlike held-out demos.
- SPEC trains in 3 h 24 min (no reward passes) vs ~3 h 54 min for GObal.

**Consequence for the method (user, 2026-11-02).** "Learned removal" currently works as a training-time
augmentation, not as an inference-time selection. Round 4 tests it (queue `remover_tune4_queue.txt`,
report `.../mimicgen/remover_tune4/`, both seed 42, launched 2026-11-02 23:42):
- **DROP**: GObal's theta pass (keep 0.5 per region) for the whole run, phi frozen (lr 0, no phi steps),
  no fine-tune -> inference on the uncut cloud. DROP ≈ GObal = dropout alone carries the gain; DROP ≈ uncut
  = the learned part matters.
- **DObal**: the decoupling fix for density -- DOscene + `theta_rho: [0.1, 1.0]` (theta's densities
  ~ U[0.1, 1] per region and window, independent of phi; phi learns from its own draws as before).
  User decision: density only, so it differs from DOscene in the decoupling alone. Diagnose by phi's own
  movement, its logged density slopes, and success -- not by hand mechanism analyses (user: that would be
  reverse-engineering the concept).

**Round 4 (2026-11-03/04; seed 42 unless noted; PRELIMINARY until DROP seed 43 and SPECarm are in).**
Drivers: `remover_tune4_queue.txt` (DROP, DObal), `scripts/sim/run_finetune_branches.sh` (round 4b),
`remover_tune4c_queue.txt` (SPECarm). New knob: `training.link_remover.finetune_mask` (fine-tune on a given
mask instead of phi's gates).

*Density, decoupled (DObal = DOscene + theta_rho [0.1, 1]).* phi now gets a clear signal on the upper arm:

| learned rho | link5 | link6 | link7 | hand f1 / ctr / f2 | fingers | scene | success 200 / 250 |
|---|---|---|---|---|---|---|---|
| DObal | 0.60 | 0.55 | 0.92 | 0.84 / 0.85 / 0.76 | 0.55 / 0.53 | 0.37 | 11.0 / 14.5 |
| DOscene (round 2) | 0.84 | 0.76 | 0.85 | 0.87 / 0.88 / 0.85 | 0.51 / 0.67 | 0.25 | 16.0 / 13.0 |

Density slope t (> 0 = sparser preferred): link5 +6.1, link6 +5.9, link7 -0.6, hand +1.2..+2.9, fingers
+5.9 / +5.7, scene +6.6 (second half 0.2: it stopped). Same structure as GObal's gates (thin the upper arm,
keep link7 + hand), found by density alone. Success: vs uncut 4096 20 / 8, p = 0.036; vs uncut 1024 15 / 32,
p = 0.019; vs DOscene p = 0.74. Density moves no tokens, so the learned density barely reaches success.

*Is GObal's gain only the dropout? (DROP = GObal's theta pass, keep 0.5 per region, for all 251 epochs; phi
frozen; no fine-tune; inference on the uncut cloud.)* DROP: 18.0 at 200, 13.0 at 250. At 200 GObal and DROP
are the same procedure (phi never touches theta's masks before the fine-tune): GObal 200 on the uncut cloud
20.5 vs DROP 200 18.0 (22 / 17, p = 0.52) -- equal.

*Fine-tune branches (round 4b).* DROP-42's 200.ckpt resumed with the identical config; the branches are bit-identical
up to epoch 225 (val loss checked; they differ from the ORIGINAL DROP from epoch 201 on -- a resume re-draws the
random stream) and differ only in the fine-tune 226-250:

| epoch 250 | success | vs FTcont | vs uncut 8.5 | vs SPEC 11.0 |
|---|---|---|---|---|
| FTcont: no fine-tune (dropout to 250) | **18.5** | – | 28 / 8, p = 0.001 | 27 / 12, p = 0.024 |
| FTlearned: fine-tune on link5 + 6 removed | 17.5 | 11 / 13, p = 0.84 | 24 / 6, p = 0.001 | 26 / 13, p = 0.053 |
| FTlearned evaluated on the uncut cloud | 18.5 | | | |
| FTnone: fine-tune on the full cloud | 19.0 | 14 / 13, p = 1.0 | 31 / 10, p = 0.002 | 30 / 14, p = 0.023 |
| FTarm: fine-tune on link5 + 6 + 7 removed | 18.5 | 17 / 17, p = 1.0 | 27 / 7, p = 0.001 | 26 / 11, p = 0.020 |

FTcont vs GObal 19.5: 15 / 17, p = 0.86; vs arm cut 20.0: 24 / 27, p = 0.78; vs the original DROP 13.0:
20 / 9, p = 0.061.

- **GObal's result is the dropout training.** Random keep-0.5 dropout of whole robot regions, with no learned
  phi, no fine-tune and the uncut cloud at inference, matches GObal and the oracle arm cut and beats the uncut
  baseline (p = 0.001). The fine-tune mask (none / learned / full / arm) and the inference mask change nothing
  measurable.
- **The original DROP 13.0 was an unlucky realisation.** Re-running only its last 50 epochs from the same
  200.ckpt gave 18.5. Run-to-run noise at a FIXED seed is therefore ~5 points on 200 scenes -- the McNemar
  p-values cover scenes, not this; single comparisons below ~5-6 points are not interpretable.
- **What the learned phi contributes at 4096 / 64: an interpretable mask, no success.**
- **Open puzzle -- fixed clouds:** the fixed arm cut (all arm links removed) reaches 20.0 / 16.5, SPEC (link5 + 6
  removed, link7 kept) only 11.0 / 11.5, the uncut cloud 8.5. Hypothesis: a visible, always-present link7
  (or any robot part) lets a policy trained without dropout shortcut through proprio-redundant points;
  dropout makes every robot part unreliable. Test queued: SPECarm (fixed link5 + 6 + 7 removed, from scratch).

*DROP seed 43:* 17.0 at 250 -- vs GObal-43 18.0 (14 / 16, p = 0.86), vs GObal-43 on the uncut cloud
18.0 (p = 0.85), vs arm cut 16.5 (p = 1.0), vs SPEC 11.5 (29 / 18, p = 0.14). Dropout alone replicates on
the second seed (18.5 / 17.0 for FTcont / DROP-43 against GObal 19.5 / 18.0).

**Round 5 (2026-11-04, user): link0-4 were never removable.** SPECarm was cancelled. Label 5 (the arm cut) is
exactly cells link0..link7. link0-4 average 27 points per training frame (median 4), but link3 / link4 come into view
in ~4-5% of frames with up to ~460 / ~980 of 4096 points, and no remover / dropout ever touched them.
- New: `region_groups: all_links` (13 regions, gates for link0..link4 as well; closing link0..7 == the label-5
  arm cut bit for bit, checked); `eval_mimicgen_ckpt.py --remover-removed` also removes link cells that are in
  no region of an old checkpoint. Chain: `scripts/sim/run_round5_chain.sh`, queue `remover_tune5_queue.txt`.
- **GObal-42 250.ckpt with link0..6 removed at inference** (link7 + gripper + scene kept): 15.0 -- vs its learned
  mask 19.5 (11 / 20, p = 0.15), vs uncut cloud 16.0 (p = 0.87), vs uncut baseline 8.5 (23 / 10, p = 0.035). Not
  better (the network never saw link0-4 removed in training).
- In ROLLOUTS link0-4 are far more visible than in the demos: 142 points per frame on average over the 200 eval
  scenes, and 83 / 200 scenes have a frame with > 100 (max 1111). Those scenes fail for every policy, the arm cut
  included (success with / without such a frame: arm cut 8.4 / 28.2, GObal 4.8 / 29.9, uncut 2.4 / 12.8). Most
  likely a symptom of failing rollouts (the arm wanders), not a cause.
- **GObalAll** (GObal + all_links, seed 42): learned mask again link5 0.10 / link6 0.05 removed, link7 0.82, hand
  0.88-0.93, fingers 0.82 / 0.87 kept (removal effect t -8.1 / -11.2; +6.2 .. +10.0). link0-4: NO signal
  (t -0.7 .. +0.7, second half within +-1.7), keep probs 0.46-0.56 -> coin flips at inference (link0, link4 just
  below 0.5). Success 17.0 at 250: vs GObal 19.5 p = 0.50, vs dropout-only 18.5 p = 0.74, vs arm cut p = 0.47,
  vs uncut 24 / 7 p = 0.003. Removing link0-4 is neither learned (the prediction-error reward does not see it) nor
  needed for the dropout effect.

**Rounds 3-5 summary (all evaluations finished 2026-11-04 15:54; 200 paired scenes):**

| run | s42 @200 | s43 @200 | s42 @250 | s43 @250 |
|---|---|---|---|---|
| GObal, learned mask | 22.0 | 20.5 | 19.5 | 18.0 |
| GObal, uncut cloud at inference | 20.5 | 16.5 | 16.0 | 18.0 |
| DROP (dropout only, no phi, no fine-tune) | 18.0 | 17.5 | 13.0 | 17.0 |
| DROP-42 tail re-run from 200 (FTcont) | | | 18.5 | |
| GObalAll (13 gates incl. link0-4) | 19.0 | | 17.0 | |
| SPEC (fixed learned mask, from scratch) | 14.5 | 8.0 | 11.0 | 11.5 |
| arm cut / uncut (baselines, 250 only) | | | 20.0 / 8.5 | 16.5 / – |

At 200, where GObal and DROP are the same procedure: GObal on the uncut cloud vs DROP 22 / 17 (s42, p = 0.52),
16 / 18 (s43, p = 0.86), pooled 38 / 35, p = 0.82. **Conclusion: on Square at 4096 / 64, the whole gain of the
learned remover comes from training theta with random whole-part robot dropout (keep 0.5 per region).** Learned
phi, its mask, the fine-tune mask and the inference mask (none, learned, link5-7, link0-6) all change nothing
measurable; a fixed cloud without dropout is clearly worse unless it is the full arm cut (SPEC 11 vs arm cut 18).
Run-to-run noise at one seed is up to ~5 points (DROP-42 13.0 vs its tail re-run 18.5). Open: why the fixed arm cut
works without dropout but SPEC (link7 + link0-4 kept) does not -- the shortcut hypothesis (visible robot points
duplicate proprio; dropout or a full cut removes the shortcut) is untested.
Running (launched 2026-11-04 16:05, user): **SPEC06** -- the specialist with link0..6 removed (only link7 +
gripper + scene kept; `region_groups: all_links`, fixed mask, seed 42), queue `remover_tune6_queue.txt`, report
`.../mimicgen/remover_tune6/`. SPEC (link5 + 6 only) got 11.0 / 11.5, the full arm cut (link0..7) 20.0 / 16.5.
**SPEC06 result (finished 2026-11-04 22:24):** 11.0 at 200, 13.0 at 250 (val loss 11.15, again among the lowest).
At 250: vs SPEC 20 / 16, p = 0.62; vs arm cut 18 / 32, p = 0.065; vs dropout-only (FTcont) 17 / 28, p = 0.14; vs
GObal 22 / 35, p = 0.11; vs uncut 22 / 13, p = 0.18. At 200 vs GObal 13 / 35, p = 0.002. **Removing link0-4 does
not fix the specialist.** The only remaining difference to the arm cut is link7 (the remover reproduces the label
cut bit for bit), so a fixed cloud without dropout reaches the arm-cut level only with EVERY arm link gone --
consistent with link7 (rigid with the gripper, encodes the EE pose) carrying a proprio shortcut. Single seed and
p = 0.065: suggestive, not shown. Next: the shortcut tests (sensitivity test; frozen robot points), watching link7.
Correction (2026-11-05): SPEC06 and the arm cut run are equivalent EXCEPT link7 and the random stream (the remover
draws extra random numbers), and a changed stream alone moved results by up to ~5 points (DROP 13.0 vs re-run 18.5). So
the 5-7 point gap to the arm cut is within noise. Control queued: **SPEC07** = link0..7 removed through the specialist
pipeline (the arm cut's cloud), seed 42, 125 + 250 ckpts, starts when the GPU is free (queue `remover_tune7_queue.txt`).
Disk (2026-11-05, user): deleted the checkpoints of rm-G-42 (4096 / 1024 / 2048), rm-GO / GOgm2 / GOclosed / GOmix /
DOscene, cut5from50, cut5from125, trunk-42 (25 files, 56 GB). Their eval JSONs, logs and analyses are kept.

**Round 7: shortcut hypothesis (2026-11-04/05; 4096 / 64, seed 42, 200 paired scenes; outputs
`R3D/data/outputs/sim/mimicgen/round7/`).** Hypothesis: visible robot points duplicate proprio and encode past actions,
so a policy trained on them without dropout leans on them instead of the nut / peg; dropout or a full arm cut removes
the shortcut.

*Part 1 -- sensitivity test* (`scripts/sim/sensitivity_shortcut.py`, offline on the 250.ckpts, no training).
594 held-out windows (every frame of the 4 val demos 17 / 87 / 130 / 153), 2 obs frames, no augmentation. A
horizontal translation by 2 or 5 cm in a per-window random direction (the same for every policy), on both obs frames of
the raw cloud BEFORE each policy's own cut / remover. Points moved per frame: nut + peg 795, link5-7 2470, gripper 816.
Value: change of the executed action's position (8 steps, mean |d pos|), mm, with the base call's own seed (pure
effect; 0 = the points never reach the policy). Noise floor = two seeds on the identical observation. 95% CI =
bootstrap over windows (windows of one demo are correlated, so optimistic). The fresh-noise effect / floor ratios
(the cube-study convention) are in `sensitivity_v2.json`; same ordering.

| policy | success 200 / 250 | floor | nut + peg 5 cm | link5-7 5 cm | gripper 5 cm |
|---|---|---|---|---|---|
| uncut | – / 8.5 | 4.6 | 19.9 [18.9, 20.9] | 12.4 [11.8, 13.1] | 12.1 [11.3, 12.9] |
| arm cut | – / 20.0 | 3.7 | 22.2 [21.3, 23.2] | 0 (cut) | 16.6 [15.7, 17.5] |
| SPEC (link5 + 6 removed) | 14.5 / 11.0 | 3.9 | 20.1 [19.1, 21.1] | 7.9 [7.5, 8.3] (= link7) | 13.8 [12.9, 14.8] |
| SPEC06 (link0-6 removed) | 11.0 / 13.0 | 3.8 | 20.0 [19.0, 21.0] | 8.4 [8.0, 8.8] (= link7) | 12.8 [12.0, 13.6] |
| DROP | 18.0 / 13.0 | 5.4 | 20.1 [19.2, 21.1] | 7.2 [6.8, 7.5] | 8.5 [8.0, 9.0] |
| FTcont | – / 18.5 | 5.4 | 20.1 [19.1, 21.0] | 7.2 [6.9, 7.6] | 8.5 [8.0, 9.0] |
| GObal (learned mask) | 22.0 / 19.5 | 3.4 | 20.9 [19.9, 21.9] | 4.5 [4.2, 4.7] | 7.9 [7.4, 8.4] |
| frozen robot (part 2) | 11.5 / 14.5 | 4.1 | 18.7 [17.9, 19.5] | 0 (live robot unused) | 0 |

At 2 cm every column is roughly half the 5 cm value (objects 8.1-9.7 mm for every policy). The frozen policy moved by
its FROZEN robot points: 11.1 [10.4, 11.8] mm at 5 cm.

- **No policy ignores the objects.** Every policy moves 20-22 mm for a 5 cm nut + peg shift, the uncut one included
  (arm cut +2.3 [1.4, 3.2] mm over uncut). The predicted "uncut / SPEC weak on objects" is NOT there.
- **Reliance on the ARM points ranks success** (with FTcont as the dropout run; user, 2026-11-05: DROP 13.0 was the
  unlucky realisation). Order by link5-7 reaction: arm cut 0, GObal 4.5, FTcont 7.2, SPEC 7.9, SPEC06 8.4, uncut 12.4.
  Success order: 20.0, 19.5, 18.5, 11.0, 13.0, 8.5 (Spearman ~0.94). The paired differences are resolved (FTcont −
  SPEC −0.66 [−1.15, −0.18] mm, FTcont − SPEC06 −1.20 [−1.66, −0.76]), though SPEC / SPEC06 see only link7 here.
- **The gripper does NOT rank success:** the arm cut (best) reacts most to it (16.6 mm). If there is a shortcut it
  runs through the arm, not robot points in general -- consistent with GObal keeping link7 + gripper.
- **DROP and FTcont are nearly the same function** (per-window correlation 0.98, mean |diff| 0.37 mm) and still
  succeed 13.0 vs 18.5%. The ~5-point run-to-run noise is closed-loop evaluation noise for almost identical weights, so
  offline sensitivity can only describe a training procedure, never one run's success.

*Part 2 -- frozen robot points* (one run: uncut 4096 / 64, seed 42, 251 epochs, 64 x 2; run dir
`mimicgen_sim-square_d2-p4096-g64-frozen-42`, 200.ckpt + 250.ckpt kept). Every frame's robot points (labels 5 + 7 =
link0-7 + gripper cells, checked identical to `point_link >= 0`) are replaced by the episode's FIRST frame's, in
training and at inference (the rollout's reset frame); the scene points stay live, agent_pos untouched.
- **Count kept at 4096 without padding:** frame t keeps its own scene points at their positions; its R_t robot
  positions get the first R_t robot points (FPS order) of an FPS-8192 render of frame 0, whose first 4096 points ARE the
  stored frame 0 (FPS prefix; asserted bit for bit on 200 / 200 demos and at every eval reset). Robot point count =
  uncut's per frame, so the token allocation is uncut's; frame 0 maps to itself. Naive "frame-t scene + stored frame-0
  robot" would have been short in 91.5% of frames (median −354; frame-0 robot 2864 vs 3228 points). References: min
  5038 robot points (demos), 4887 (eval scenes) >= 4096, so no frame can run out.
- **Train = eval:** one function (`r3d/common/robot_freeze.py`), in `RobotwinSimDataset._sample_to_data` BEFORE
  augmentation (noise / jitter draws identical to the uncut run's -- checked) and in `eval_mimicgen_ckpt.py`.
  Residual leak: occlusion holes of the current arm in the nut / peg points.

| frozen vs (a only / b only, p) | frozen @200 (11.5) | frozen @250 (14.5) |
|---|---|---|
| uncut @250 (8.5) | 17 / 11, p = 0.35 | 21 / 9, p = 0.043 |
| arm cut @250 (20.0) | 16 / 33, p = 0.021 | 18 / 29, p = 0.14 |
| SPEC, same epoch (14.5 / 11.0) | 13 / 19, p = 0.38 | 21 / 14, p = 0.31 |
| SPEC06, same epoch (11.0 / 13.0) | 18 / 17, p = 1.0 | 24 / 21, p = 0.77 |
| FTcont @250 (18.5) | 12 / 26, p = 0.034 | 17 / 25, p = 0.28 |
| GObal, same epoch (22.0 / 19.5) | 12 / 33, p = 0.002 | 18 / 28, p = 0.18 |

**Interpretation.** Frozen sits at the SPEC / SPEC06 level (~11-14%): above uncut at 250 only (p = 0.04), below the
arm cut / dropout level at 200 (p = 0.02-0.03), not separable at 250. Neither clean outcome: robot points with no
current information give at most part of the arm cut's gain. Splitting it, frozen − uncut = the information part
(+3 / +6) and arm cut − frozen = the token part (+5.5 / +8.5); each is within the ~5-point run-to-run noise, single
seed. The frozen policy still uses its constant robot points (11.1 mm per 5 cm), as uncut uses the live arm.
Together with part 1: the policies at ~18-20% are those with no arm points (arm cut) or trained with arm dropout;
every fixed cloud that keeps arm points (uncut, frozen, SPEC, SPEC06) is below them. Open: seed repeats of frozen and
SPEC07 (arm cut through the specialist pipeline) to separate "information" from "tokens" from noise.

**SPEC07 result (finished 2026-11-05 10:09; link0..7 removed through the specialist pipeline, seed 42, 200 paired
scenes): 13.0% at 250 (26/200).** Its cloud is the arm cut's cloud (closing link0..7 == the label-5 cut, bit for bit,
`check_link_remover.py`); the training configs differ only in the policy class, the remover block and the dataset cut
flag. Paired at 250:

| SPEC07 vs | other | SPEC07 only / other only | p |
|---|---|---|---|
| SPEC06 (link0-6) | 13.0 | 19 / 19 | 1.00 |
| SPEC (link5+6) | 11.0 | 16 / 12 | 0.57 |
| arm cut s42 | 20.0 | 15 / 29 | 0.049 |
| arm cut s43 | 16.5 | 16 / 23 | 0.34 |
| GObal s42 | 19.5 | 17 / 30 | 0.08 |
| FTcont (dropout to 250) | 18.5 | 12 / 23 | 0.09 |
| uncut | 8.5 | 20 / 11 | 0.15 |

**link7 is NOT the difference:** with the same cloud as the arm cut, SPEC07 lands exactly on SPEC06 (19 / 19). This
retracts the link7 reading of SPEC06 above, and the round-7 pattern "no arm points -> ~18-20%" (SPEC07 has no arm
points and sits at 13). All four fixed-mask runs of the remover pipeline cluster at 11-13% (SPEC 11.0 / 11.5,
SPEC06 13.0, SPEC07 13.0); the two arm-cut runs of the plain DP3 pipeline sit at 20.0 / 16.5. What still differs
between SPEC07 and the arm cut is only the random stream: `thin_window` draws extra `torch.rand` (the
within-region FPS start, inert at rho 0 / 1), the token FPS is the same encoder FPS in both, and `fill_to_min_keep`
never fired (filled fraction 0.0 in every epoch, mean 1715 points kept). So either the fixed-mask runs of the remover
pipeline are consistently unlucky draws, or the arm cut s42 (20.0) is a high one -- vs the arm cut s43 the gap is not
significant (p = 0.34).
User (2026-11-05): the arm-cut runs are trusted (two seeds); the link7 question is re-run in the ARM-CUT pipeline, not
the specialist one. **cutL06** = the arm cut run with link7 kept: plain DP3, `task.drop_point_links=[link0..link6]`
(new dataset / eval option: the same `cut_points_by_label` after augmentation, keyed on the link cell; link0..7
reproduces `drop_point_labels=[5]` bit for bit in training and live in eval, `scripts/sim/check_link_cut.py`), seed
42, 251 epochs, 64 x 2, ckpts 200 + 250. link7 holds ~420-470 of the 4096 points per training frame (~620 at the
eval resets). Chain `scripts/sim/run_cut_keep7.sh`, log `mimicgen/cut_keep7/chain.log`.

**cutL06 result (finished 2026-11-06 00:07; checks and smoke passed): 11.5% at 200, 14.5% at 250.** Paired at 250:
vs arm cut s42 (20.0) 18 / 29, p = 0.14; vs arm cut s43 (16.5) 16 / 20, p = 0.62; vs SPEC06 (13.0, the SAME cloud
through the specialist pipeline) 21 / 18, p = 0.75; vs SPEC07 (13.0) 20 / 17, p = 0.74; vs GObal (19.5) 20 / 30,
p = 0.20; vs FTcont (18.5) 16 / 24, p = 0.27; vs uncut (8.5) 22 / 10, p = 0.05. At 200 vs GObal (22.0) 13 / 34,
p = 0.003. (Same percentages as frozen-42 by coincidence: 15 / 15 discordant at 250.)
- Keeping link7 costs 2-5.5 points against the two arm-cut seeds, not significant: link7 is not shown to matter.
- The two pipelines agree on the same cloud (cutL06 14.5 vs SPEC06 13.0, p = 0.75): the specialist runs were not
  broken.
- Spread between runs that differ only in seed or random stream: arm cut 20.0 / 16.5, DROP 13.0 vs its re-run 18.5,
  DROP s42 / s43 13.0 / 17.0 -- 3.5-5.5 points. Every fixed-cloud removal run (14.5, 13.0, 13.0, 11.0, 11.5, 16.5,
  20.0) lies within ~one such spread of ~14.5; the dropout runs sit at 17-19.5; uncut at 8.5 is the only fixed-cloud
  run clearly below. The scene-level McNemar p-values do not include this training-run variance.
Queued 2026-11-06 (user): seed 44 pair through ONE process, `scripts/sim/run_cut_pair.sh` (SEED=44):
**cutL07-44** (`drop_point_links=[link0..link7]` == the arm cut) and **cutL06-44** (link7 kept); identical otherwise
(251 epochs, 64 x 2, ckpts 200 + 250, eval 250 then 200). Neither removes the mount (class 3, ~8 pts/frame); both
remove link0-4 (~30 pts/frame on average, in view in 3-58% of frames). A re-run of arm cut s42 was NOT done: training
at a fixed seed is deterministic here, so it would reproduce 20.0. Log `mimicgen/cut_pair_s44/chain.log`. Separating them
needs a seed repeat (a second SPEC07 seed, or a third arm-cut seed), not a new mask.

*Part 3 -- the 1024-point policies (2026-11-05, same test, first 1024 points of each cloud).* Prediction (from
"shortcut x coverage": at full object coverage the policy needs the arm less): 1024 uncut relies on the arm much
less than 4096 uncut. **Falsified.**

| policy | success 250 | floor | nut + peg 5 cm | link5-7 5 cm | gripper 5 cm |
|---|---|---|---|---|---|
| 4096 uncut s42 | 8.5 | 4.6 | 19.9 [18.9, 20.9] | 12.4 [11.8, 13.1] | 12.1 [11.3, 12.9] |
| 1024 uncut s42 | 23.0 | 4.3 | 21.6 [20.5, 22.8] | 13.4 [12.6, 14.2] | 12.5 [11.7, 13.4] |
| 1024 uncut s43 | 19.0 | 4.3 | 20.7 [19.8, 21.7] | 11.9 [11.2, 12.6] | 12.9 [12.1, 13.8] |
| 1024 arm cut s42 | 17.5 | 3.0 | 22.3 [21.3, 23.4] | 0 | 17.1 [16.2, 18.2] |
| 1024 arm cut s43 | 20.5 | 2.9 | 22.0 [21.0, 23.1] | 0 | 16.2 [15.4, 17.1] |

1024 uncut leans on the arm exactly as much as 4096 uncut (s42 +0.9 [+0.2, +1.6] mm, s43 −0.5 [−1.3, +0.2]) and
succeeds 2-3x as often. **Reliance on the arm points is not harmful in itself**, so the within-4096 ranking (arm
reliance vs success, Spearman ~0.94) does not carry over across point counts and cannot be the mechanism by itself.
The 1024 policies react to the objects as much as everything else (+0.8 / +1.7 mm over 4096 uncut). Whatever separates
4096 uncut from the rest is invisible to a 5 cm rigid translation of nut + peg -- candidates: object ORIENTATION / shape
detail (nut yaw is random over 360 deg and coverage of the nut is ~50% at 4096 / 64), which a translation does not
probe.

Files: `r3d/common/robot_freeze.py`; task keys `freeze_robot_labels` (null = off) + `freeze_reference`
(`mimicgen_sim.yaml`); sidecar `R3D/data/square_d2-agentview_p4096-200.frame0_p8192.npz`
(`scripts/sim/make_frame0_reference.py`); `+training.checkpoint_epochs=[200,250]` (train.py, absent = unchanged);
checks `scripts/sim/check_frozen_robot.py` (29 checks: default off bit-identical to git HEAD for uncut + arm cut, train
+ val, normalizer); chain `scripts/sim/run_round7_chain.sh`; report `scripts/sim/round7_report.py`.

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

## Next task candidate: MimicGen Threading (checked 2026-11-06, no training)

User constraint: failures on Square D2 come from generalization (50 x 50 cm nut and peg ranges), not from the
observation; the next task should be harder than cup / blocks but with small init ranges -- AND (user) the target
must move, or everything after the grasp can be memorized from proprio. `scripts/sim/threading_task_check.py`
(r3d_robosuite, CPU, simulator states only) on the 1000-demo MimicGen datasets, output
`R3D/data/outputs/sim/mimicgen/threading_check/threading_task_check.json`. Ring radius (success tolerance) 12 mm.
Held-out (5-fold by demo) RMSE of the EE position at the insertion frame, gradient-boosted trees:

| dataset | n | tripod range | mean only | proprio at grasp | + tripod pose | + in-hand offset |
|---|---|---|---|---|---|---|
| D0 (tripod fixed) | 1000 | fixed | 12.6 mm | 7.5 | 7.5 | 5.9 |
| D1 | 1000 | 25 x 10 cm, 120 deg | 80.0 | 87.8 | 12.3 | 10.5 |
| D1, x + yaw middle half | 313 | 12 x 10 cm, 60 deg | 52.0 | 59.5 | 12.7 | 10.4 |

- D0 is memorizable after the grasp: a constant insertion pose is already at the ring radius, proprio at the grasp
  gets 7.5 mm. User's concern confirmed; all D0 variants (fixed target) are out.
- D1 needs the tripod: proprio alone 80-88 mm, with the tripod pose 12 mm. Holds in the narrowed window (52-60 ->
  12.7 mm), which keeps 313 demos (more than the 250 expected from uniform sampling).
- The in-hand needle offset varies 4-5 mm per axis at the grasp and adds ~2 mm (12.5 -> 10.5): a fine-detail cue.
- All 1000 demos parse (grasp >= 5 frames, success; 4-6% reach success only with the final action). Median 211
  frames per demo, 144 after the grasp.
Not yet measured: whether the agentview cloud resolves the ring opening and the needle at 4096 / 64 tokens (view
check), and whether a baseline's failures are near misses (precision) or gross (generalization).

**Seed-44 pair + val loss (2026-11-06).** Success at 250: arm cut s44 (cutL07-44) 16.5 (200: 14.5), link7 kept s44
(cutL06-44) 13.0. Paired arm cut vs link7 kept: s42 29 / 18 (p = 0.14), s44 23 / 16 (p = 0.34), pooled 52 / 34
(p = 0.07): removing link7 helps by ~4.5 points on both seeds, at the edge of resolution. Val loss (e-3, mean of the last
25 epochs before 250, each run on its own val clouds; `scripts/sim/wandb_val_loss.py`):

| cloud | run | val loss @250 | success @250 |
|---|---|---|---|
| link0-7 removed | arm cut s42 / s43 / s44 (cutL07) | 11.66 / 11.59 / 12.44 | 20.0 / 16.5 / 16.5 |
| link0-7 removed | SPEC07 s42 | 12.28 | 13.0 |
| link0-6 removed | cutL06 s42 / s44 | 12.93 / 12.68 | 14.5 / 13.0 |
| link0-6 removed | SPEC06 s42 | 11.15 | 13.0 |
| link5+6 removed | SPEC s42 / s43 | 11.18 / 13.66 | 11.0 / 11.5 |
| none | uncut s42 / frozen s42 | 14.51 / 16.50 | 8.5 / 14.5 |

- Matched pairs (same pipeline, same seed): the arm cut has the lower val loss on both seeds at 250, as in success --
  clearly at s42 (-1.27), marginally at s44 (-0.24, ~2 SEM; reversed at 200: 12.12 vs 11.94).
- Across runs val loss does not track success: Spearman -0.17 over the 11 fixed-cloud runs (p = 0.63; -0.30 without
  frozen). The same cloud at the same seed through the other pipeline (another random stream) moves val loss by
  0.6-1.8 (SPEC07 vs arm cut s42, SPEC06 vs cutL06 s42) -- more than the s44 gap.
- Conclusion (user): the learner does not work as intended. Reasons: (1) loss and success are only loosely aligned;
  (2) one link more or less is a marginal effect, below run noise in both loss and success; (3) phi's reward scores
  masks for ONE shared network that has learned to use link7 (removing it = information that network loses), not what a
  network trained without link7 would reach -- a from-scratch run can reallocate the tokens and lean on agent_pos.

**Threading D1 narrowed: runs queued (2026-11-06, user).** Window `threading_d1n` (r3d/common/robosuite_tasks.py):
tripod x [-0.0375, 0.0875], y [-0.2, -0.1], yaw [60, 120] deg (middle half of D1's x and yaw), needle as D1. Training
data: the FIRST 200 demos in file order inside the window (317 of 1000 qualify; ids 1 ... 612, stored in the zarr attrs
`demos`) -- as for Square, where the zarr holds demos 0-199. Eval: `Threading_D1_Narrow` resets from exactly the
window (asserted per scene), 200 scenes of seed block 1, MAX_STEPS 500 (demos: median 215, max 261 steps). 4096
points / 64 tokens, 251 epochs, 64 x 2, seed 42: uncut -> arm cut [5] -> GObalAll (GObal with gates for every
arm link: `region_groups=all_links`, 13 regions incl. link0-4; user); eval 250 first, then 200. One test
demo: needle 332 points per frame (min 162), tripod 1116 (min 572), arm 1405, gripper 1234 of 4096. Chain
`scripts/sim/run_threading_chain.sh`, log `mimicgen/threading/chain.log`; zarr `R3D/data/threading_d1n-agentview_p4096-200.zarr`.

**Threading results at 250 (2026-11-07; seed 42, 200 paired scenes) -- the Square effect does not transfer.**

| run | success @250 | vs uncut (only this / only uncut, p) |
|---|---|---|
| uncut | 37.5 | - |
| arm cut [5] | 35.0 | 36 / 41, p = 0.65 |
| GObalAll (dropout keep 0.5, gates for all 13 regions) | 36.5 | 41 / 43, p = 0.91 |

- Coverage (`scripts/sim/coverage_cut_vs_uncut.py`, training clouds, 64 tokens): Threading is in the Square-4096 regime
  (patches ~1.8 cm, needle 56% / tripod 48% covered uncut), NOT the 1024 one (100%). The arm cut raises needle coverage
  to 76% and tripod to 64% (the large tripod absorbs most freed tokens: 21 -> 31; needle 5.7 -> 8.4) -- and success does
  not move. Square 4096 for reference: nut 49 -> 83%, peg 57 -> 93%.
- Neither the arm cut nor the dropout helps here, so "removing proprio-redundant robot points" (reallocation) and
  "robot-part dropout" (reliance) are both Square-specific so far. User decision rule (agreed before the GObalAll
  result): GObalAll ~ uncut -> the hypothesis is dead for the paper. It is.
- GObalAll's learned mask: removes link1, link6, link7, hand_centre; keeps link0, link2-5, hand sides, fingers. Reward
  signal (removal effect, t over all phi steps): link6 -8.6 (remove), hand_f1side +6.3 / hand_f2side +5.9 / finger1 +3.3
  (keep); link7 -1.2 (2nd half -2.6), link1 -1.6, link0-4 |t| < 2 (noise again), hand_centre +0.7 (2nd half -2.8).
  On Square phi removed link5 + link6 and kept link7: the learned masks differ between tasks, and none helps success.
- Single seed per run; the scene-level interval for the arm cut is about -11 to +6 points, so a Square-sized gain (+9)
  is unlikely, small effects are not excluded. The 200 evals follow (chain).

**Last test (2026-11-07, user): is the Square gain about generalization (out-of-distribution scenes), not tokens?**
(a) Square, from existing evals (`scripts/sim/ood_gain_square.py`, epoch 250, 200 scenes): U = uncut s42, C = arm cut
(cut5-42, cut5-43, cutL07-44), D = dropout-trained (GObal-42/43, FTcont-42, DROP-43, GObalAll-42); gains per tertile of
three "unusualness" measures. Gain C-U near / mid / far: nn distance to the training layouts +10.4 / +9.6 / +7.5,
reach from the robot base +10.4 / +11.6 / +5.5, link0-4 points in rollout +10.9 / +8.6 / +8.0; D-U alike. Far - near
CIs all include 0 and lean NEGATIVE. **Not supported:** the gain is spread evenly (roughly a doubling of success in every
tertile), not concentrated in unusual scenes. (b) Threading policies evaluated on stock Threading D1 (tripod also outside
the training window), split inside / outside: queued, `scripts/sim/run_threading_ood.sh` -> `mimicgen/threading/ood_report.txt`.
(b) result (2026-11-08): the three Threading 250 checkpoints on stock Threading D1, 200 scenes, split by the training
window -- inside (44): uncut 40.9 / arm cut 36.4 / GObalAll 40.9 (8/10, 9/9 vs uncut); outside (156): uncut 10.9 / arm
cut 6.4 / GObalAll 7.1 (9/16 p = 0.23, 10/16 p = 0.33). **Not supported either:** out of distribution the cut and the
dropout are, if anything, slightly worse than uncut. Also at epoch 200 on the narrowed eval: uncut 39.5, arm cut 29.0
(31/52 vs uncut, p = 0.03), GObalAll 34.0 (p = 0.25). Conclusion: neither token reallocation nor robustness to distribution
shift explains the Square gain in a way that transfers; the robot-point removal / dropout line is closed for the paper.

**hanging_mug dropout run (queued 2026-11-08, user).** Does the Square dropout effect hold on the task where the cut
helped most (4096/64: uncut 42.0, arms cut 59.0, 51/17 discordant, p < 0.001)? GObal itself is not portable (per-link
labels missing from the RoboTwin data; re-collection needs the non-deterministic planner), so the DROP mechanism at
class level: `task.dropout_point_labels=[4,5,6,7]`, keep 0.5 per class per window, training only (checks:
`scripts/sim/check_label_dropout.py`, 13/13; per class dropped 44-53% of the windows where present; arms in view in only
~half the windows). Recipe = the arm cut run's (601 epochs, 64 x 4, ckpts 300 + 600, seed 42). All three 600.ckpts
evaluated on the CURRENT shared scene list (tag `shared`; the old 4096 evals used their own list, 180/200 shared) for a
paired report. Chain `scripts/sim/run_mug_drop.sh`, report `outputs/sim/mug_drop/report.txt`.
Interim (2026-11-08 16:10): dropout 600.ckpt 55.5% on the shared list. Paired on the 180 scenes shared with the OLD
4096 list (each scene is fixed by its seed, eval deterministic per scene): uncut 41.1 / arms cut 59.4 / dropout 54.4;
dropout vs uncut 43 / 19 (p = 0.003), dropout vs arms cut 31 / 40 (p = 0.34), arms cut vs uncut 48 / 15. **The Square
pattern replicates on hanging_mug: dropout ~ arm cut > uncut.** Note: the dropout network is evaluated on the FULL cloud,
so its token allocation at test time equals uncut's -- its gain cannot be token reallocation; it is in how the network
uses the robot points. Full 200-scene re-evals of uncut and arms cut on the shared list follow (chain).
Final (2026-11-08, all 600.ckpts on the shared 200-scene list, paired): uncut 45.0 / arms cut 60.0 / dropout (full cloud)
55.5 / dropout with the arms cut at eval 54.0. Arms cut vs uncut 49 / 19 (p = 0.0004); dropout vs uncut 44 / 23
(p = 0.014); dropout vs arms cut 34 / 43 (p = 0.36); dropout cut-at-eval vs full cloud 17 / 20 (p = 0.74). As on Square:
the dropout network is indifferent to whether the arms are present at test time, and ~ the arm cut. Re-evaluated on the
shared list, uncut moved 42.0 -> 45.0 and arms cut 59.0 -> 60.0 (scene-list change only).

**Sensitivity test on all three tasks (2026-11-09; `scripts/sim/sensitivity_shortcut.py --task square|threading|mug`).**
Pure effect (same seed) on the executed action of a 5 cm horizontal shift of one part, held-out windows; ratio = effect /
the policy's own noise floor; abs = mm (Square, Threading) or rad over 14 joints (hanging_mug).

| task | policy | objects | arm | gripper |
|---|---|---|---|---|
| hanging_mug | uncut | 5.26x (0.090) | 5.33x (0.092) both arms | 4.03x (0.069) |
| hanging_mug | arms cut | 5.21x (0.091) | 0 | 6.79x (0.118) |
| hanging_mug | dropout (full cloud) | 4.79x (0.104) | **1.07x (0.023)** | **1.49x (0.032)** |
| Square | uncut | 4.36x (19.9 mm) | 2.72x (12.4) link5-7 | 2.65x (12.1) |
| Square | arm cut | 6.07x (22.2) | 0 | 4.53x (16.6) |
| Square | DROP (full cloud) | 3.72x (20.1) | **1.33x (7.2)** | **1.57x (8.5)** |
| Threading | uncut | 11.16x (22.7 mm) | 5.72x (11.6) link5-7 | 5.36x (10.9) |
| Threading | arm cut | 14.75x (24.9) | 0 | 8.85x (14.9) |
| Threading | GObalAll (its learned mask) | 15.99x (26.4) | 0.14x * | 2.30x * |

\* GObalAll evaluated on its learned mask (link6, link7, hand_centre removed), so its arm / gripper numbers are partly the
mask. Findings: dropout makes the policy far less sensitive to robot points (-40 to -75% in abs), object sensitivity
barely changes (Square +1%, hanging_mug +15%). Suggestive (3 tasks, different arm perturbations): the uncut policy's
object / arm sensitivity predicts the gain -- hanging_mug 1.0 (gain +15 / +10.5), Square 1.6 (+9), Threading 2.0 (0).

**Threading GObalAll with other masks at inference (2026-11-10, 250.ckpt, 200 paired scenes):** own learned mask 36.5 /
full cloud 37.5 / arm-cut mask (link0-7) 41.5. Arm-cut mask vs uncut (37.5) 47 / 39 (p = 0.45), vs arm cut run (35.0)
46 / 33 (p = 0.18), vs its own full cloud 26 / 18 (p = 0.29); full cloud vs uncut 37 / 37. As on Square / hanging_mug the
dropout network does not depend on which mask it sees (all within noise); unlike them it is not better than uncut.

**Token-dropout control (queued 2026-11-12, user).** Is the Square gain of robot-part dropout (DROP / FTcont / GObal,
17-19.5 vs uncut 8.5) just "dropout", or does it need the structure? Standard ViT token dropout (PatchDropout, Liu et al.
2023; Uni3D pretrains with 0.5): each training frame independently keeps a uniform random 48 of its 64 encoder tokens
(p = 0.25), after the positional embedding, before the transformer; the head sees only the kept tokens and their centres;
eval keeps all 64. Otherwise the uncut s42 recipe (plain DP3, 4096 / 64, 251 epochs, 64 x 2, 200 scenes); ckpt 250 only (user).
Before this, `patch_dropout` was 0 in every config, and the inherited PatchDropout was broken for `pointsam`: it dropped
tokens from x but not from the centres, so pc_pe would not have lined up with the tokens. Fixed: one index set gathers
both (`Uni3DPointcloudEncoder.forward`; at p = 0 the path is unchanged). Note: every run already has UNIFORM POINT
dropout (`random_point_dropout`, 0-80% of points per cloud, training only) and stochastic depth 0.2. Checks
`scripts/sim/check_token_dropout.py` (10/10); chain `scripts/sim/run_token_dropout.sh`, run
`mimicgen_sim-square_d2-p4096-g64-tokdrop25-42`, report `mimicgen/token_dropout/report.txt`.

**Table-in-the-cloud eval (queued 2026-11-12, user: out of interest, not a test of the claim).** DROP s42 and uncut s42,
250.ckpt, 200 scenes, cloud cropped by the workspace box only (`eval_mimicgen_ckpt.py --keep-table 0.02`: floor 2 cm
below the table top, TABLE not filtered; normalizer = training box, table z clamped onto its floor, min -1.03). Measured
first (`scripts/sim/table_cloud_check.py`, 10 eval resets): table 3630 / 4096 points and **53 / 64 tokens**; nut / peg
11.3 / 9.9 -> 1.7 / 1.0 tokens, arm + gripper 41 -> 7. So it measures token starvation by background, not whether a
policy ignores irrelevant content (agreed with the user). A fair version would APPEND table tokens to the unchanged 64.
Chain `scripts/sim/run_table_eval.sh` (after the token-dropout chain), report `mimicgen/table_eval/report.txt`.

**Threading 1024 / 64 (queued 2026-11-12, user).** Completes the point-count comparison: on Square and hanging_mug
1024/64 uncut was the best plain policy; on Threading the arm cut did not help at 4096. If 1024 helps on Threading,
sparser clouds and the cut are separate phenomena. Recipe = Threading 4096 uncut except `task.num_points=1024` (same
zarr, seed 42, 251 epochs, 64 x 2, ckpt 250, 200 scenes). Chain `scripts/sim/run_thr_p1024.sh` (runs after the
token-dropout and table-eval chains), report `mimicgen/threading/p1024_report.txt`.
**Token-dropout RESULT (2026-11-12, 250.ckpt, 200 paired scenes): 8.0% (16/200)** -- vs uncut s42 8.5% 11 / 12 (p = 1.0);
vs DROP s42 13.0% 13 / 23 (p = 0.13); vs DROP s43 17.0% 13 / 31 (p = 0.010); vs FTcont 18.5% 13 / 34 (p = 0.003); vs arm cut
s42 20.0% 12 / 36 (p = 0.001). Classical uniform token dropout (p = 0.25) does not reproduce the robot-part-dropout gain;
it sits exactly at uncut. Single seed; 0.5 not run (user: wait for this result first).

**Claim sweep (queued 2026-11-12, user plan `mimicgen/claim_sweep/claim_sweep_plan.md`).** R1 uncut s43 / s44; R2 armsparse
and R3 scenesparse s42 / s43 (new `task.thin_point_set: arm | non_arm`, `thin_prefix: 1024`, `r3d/common/prefix_thin.py`:
the arm resp. everything else reduced to its points in the stored FPS-1024 prefix, same cut path as the arm cut); R4
DROP-1024 s42 / s43; R5 DROP s44; E1 = uncut networks with the arm cut at inference only (`eval_mimicgen_ckpt.py
--cut-labels 5`; hanging_mug uncut 600 with `--drop-point-labels 4,5`). Checks `scripts/sim/check_prefix_thin.py` (23 dataset
+ 5 eval-reset checks, all pass: option off bit-identical to HEAD; arm cells == label 5 on all 30646 frames; both rules ==
the first 1024 points; live first-1024 == live FPS-1024 on all 200 eval resets; kept points min 1452 / 1455 train, 2767 /
1986 eval). Mechanism (`scripts/sim/claim_sweep_mechanism.py`, 1533 frames) -- gate PASS:

| cloud | points | arm tok / cov / patch cm | nut tok / cov / patch cm | peg tok / cov / patch cm |
|---|---|---|---|---|
| uncut 4096 | 4096 | 35.8 / 0.47 / 2.1 | 6.7 / 0.48 / 2.4 | 6.6 / 0.57 / 2.5 |
| armsparse | 2289 | 34.6 / 0.99 / 4.4 | 6.8 / 0.50 / 2.3 | 6.8 / 0.58 / 2.5 |
| scenesparse | 2831 | 37.0 / 0.49 / 2.1 | 6.6 / 1.00 / 4.5 | 6.3 / 1.00 / 4.4 |
| 1024 | 1024 | 35.8 / 0.99 / 4.4 | 6.8 / 1.00 / 4.5 | 6.5 / 1.00 / 4.4 |

Config diffs vs the reference runs (`claim_sweep/config_diffs.txt`): only seed / name / num_points / thin set /
checkpoint_epochs, plus option keys added after the reference runs (all null). Chain `scripts/sim/run_claim_sweep.sh`
(after the table-eval and Threading-1024 chains), report `scripts/sim/claim_sweep_report.py` -> `claim_sweep/report.md`.

**Token-budget side runs (queued 2026-11-14, user).** Is 64 tokens a fair budget, and does the arm cut's gain survive
more tokens? Square 4096, seed 42 only (second seed only if a result is too close to call), 250.ckpt, 200 scenes:
128 uncut, 128 arm cut [5], 256 uncut. The claim is relative (cut vs uncut at the SAME budget): if 128 cut - 128 uncut
~ 64 cut - 64 uncut, the effect transfers and 64 was a smaller budget to show the concept. 256 cut skipped (the cut
cloud, ~1710 points, is fully covered at 128 x 32 already). Chain `scripts/sim/run_token_budget.sh` (after the claim
sweep), queue + report `mimicgen/token_budget/`.

**Queue restructured (2026-11-14, user).** Rule: one seed for side checks, two for main results. Dropped: uncut s44,
armsparse / scenesparse s43, DROP-1024 s43, DROP s44, E1 s44. Remaining, one chain `scripts/sim/run_final_queue.sh`
(log `mimicgen/final_queue.log`): scenesparse s42 (running) -> DROP-1024 s42 -> E1 uncut s43 -> claim report -> token
block (`token_budget/queue.txt`: 4096/128 uncut, 4096/128 cut, 4096/256 uncut, then 2048/128 cut and 4096/256 cut,
removable) -> `scripts/sim/token_budget_report.py`. Coverage grid measured on stored frames (`scripts/sim/coverage_grid.py`):
token count barely moves Square success at fixed points (1024: 19.0 / 23.0 / 20.5; 2048: 14.0 / 15.5 / 16.5, although
2048 nut coverage 0.84 -> 0.99), while uncut success follows object patch size (4.5 / ~3.3 / 2.5 cm -> ~21 / ~15 / ~8%)
and every cut lands at ~17-20. Working reading (user + me): fine object patches + an arm the policy can rely on hurt;
removing the arm or training with it unreliable restores the coarse-patch level (armsparse: coarsening the arm alone
does little). The 4096/128 and /256 uncut runs test whether coverage matters at 4096 (0.48 -> 0.81 -> 0.99 at
~2.4 cm patches); 4096/128 uncut has the coverage of 4096/64 cut.
Added (user, 2026-11-14): **4096 / 64 tokens, group size k = 128** (`policy.pointcloud_encoder_cfg.group_size=128`) at the
end of the token block: coverage 1.00 / 1.00, object patches 4.7 cm (= 1024's 4.5), arm patches 4.6 cm, arm tokens 35.8,
at full 4096 density. Against 4096/256 (also full coverage, 2.3 cm patches) it separates coarseness from coverage.
Hypothesis behind it (shortcut learning, not measured): with fine object patches the object signal (esp. nut yaw) must be
assembled across tokens and is hard to learn, so the easy arm signal starves it; coarse patches make the object signal
easy, cut / dropout remove the easy alternative.
Queued last (user, 2026-11-14): **nut-yaw sensitivity** (`scripts/sim/sensitivity_yaw.py`, chain `run_yaw_sens.sh`, after
the final queue): round-7 machinery, nut points rotated +-15 / +-45 deg about their centroid (and a 2 cm nut shift as a
position reference); measure = change of the executed actions' goal rotation (deg) and position (mm), per phase.
Prediction of the shortcut hypothesis: uncut 4096 / armsparse react less to the nut's yaw than cut / dropout / coarse-patch
policies. `sensitivity_shortcut.build_obs` now also handles drop_point_links and prefix thinning (needed for cutL07-44,
armsparse, scenesparse). Smoke test (stride 20, ~30 windows only, NOT a result): yaw 45 -> rot change uncut 2.9 deg,
arm cut s44 4.8, FTcont 5.1, scenesparse 2.9.

**DROP-1024 s42 (2026-11-14): 22.0% at 250** (= uncut 1024 23.0, 20 / 22, p = 0.88; the driver also evaluated 200.ckpt,
27.5%, because MID_CKPTS="" falls back to the default list -- harmless). **Combined interpretation as of 2026-11-14:
`docs/current_interpretation.md`** (facts, hypotheses, what each queued run decides, claim wording).

**Density convergence + density specialist (user, launched 2026-11-19 15:35; chain `scripts/sim/run_density_long.sh`,
outputs `R3D/data/outputs/sim/mimicgen/density_long/`).** DOscene (round 2, density only + scene region, NOT the
decoupled DObal) never converged: over its last 75 phi epochs link5, link6, both fingers and the scene were still
pushed sparser (density slope t +2.3 .. +3.2; scene 0.37 -> 0.25, finger1 0.67 -> 0.51), only link7 and the hand had
no push left (|t| < 2). `scripts/sim/density_convergence.py` makes that the criterion: a region has converged when
|density slope t| < 2 over the last 100 phi epochs, or it sits at a bound (scene floor 0.1, snapped 0). A rho-range
test is NOT used: a no-signal region random-walks under Adam (~lr per phi step) and would never pass.
- **DOlong** = DOscene, 4096 / 64, seed 42, with **init rho 0.5** (was 0.9) and a 3.4x longer phi phase: 651 epochs =
  warm-up 50 | phi 601 | **no fine-tune** (user: 25 epochs at the end of the cosine LR do nothing; first launch with a
  25-epoch fine-tune stopped after 2 epochs and restarted), only 650.ckpt saved (disk: 8.9 GB free at launch). Why 0.5: logit 0 is the
  neutral middle of the bounded logit range, so the warm-up's sigma 1 draws span rho ~0.12-0.88 (scene ~500-3600 of
  4096 points) and theta has seen sparse AND dense clouds before phi judges them; 0.9 put the scene's trek (to 0.25)
  outside what theta had seen (the round-2 lesson: with a shared theta the start biases the answer, GO vs GOclosed);
  a start near the known uncut optimum (~0.25) would bake in the answer.
- **SPECrho** = the density specialist: DOlong's final deterministic rho as a FIXED table (`policy.link_remover.fixed_rho`,
  new, checks in `check_link_remover.py` part 14), from scratch, standard budget (251 epochs, 64 x 2, seed 42),
  ckpts 200 + 250, 200 paired scenes. The specialist question for density, as SPEC was for gates (SPEC 11.0 / 11.5).
Disk (2026-11-19, user): deleted the 200.ckpt of 16 runs whose reported result is 250 (rm-DObal-42, rm-DROP-42/43,
rm-GObal-42/43, rm-GObalAll-42, rm-SPEC-42/43, rm-SPEC06-42, cutL06-42/44, cutL07-44, frozen-42, threading p4096,
p4096-cut5, rm-GObalAll-42) -> 41 GB free. Their 250.ckpts and all eval JSONs (200 and 250) are kept.
