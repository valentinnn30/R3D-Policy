# Current interpretation: how the learnings combine (2026-11-14)

Working summary for the presentation. Square D2 unless stated: 200 demos, 251 epochs, batch 64 x 2, final
checkpoint (250), 200 paired eval scenes (seed block 1), k = 32, 64 tokens unless stated. Paired comparisons are
McNemar on the same scenes ("a only / b only"). Raw numbers and history: `docs/sim_token_allocation_learnings.md`;
claim-sweep tables: `R3D/data/outputs/sim/mimicgen/claim_sweep/report.md` (`scripts/sim/claim_sweep_report.py`).
Seed rule (user): one seed for side checks, two for main results.

## 1. The picture in one paragraph

On Square at 4096 points the uncut policy is weak (8.5 / 8.0%). Two different routes bring it to the same level
(~17-23%): **(A) coarse, fully covered object patches** (1024 points; or only the scene thinned, "scenesparse"), or
**(B) taking away the policy's reliance on the arm** (cutting the whole arm, or training with random whole-part
dropout and testing on the full cloud). Neither route stacks with the other (cut / dropout add nothing at 1024).
What does NOT explain it: token reallocation, the token count, classical token dropout, partial cuts, and coarsening
only the arm. Working hypothesis (not measured): **shortcut learning** -- with fine object patches the object's shape
and orientation must be assembled from many small tokens and is hard to learn from 200 demos, so the always-present,
proprio-redundant arm is the easier signal and starves the object features. Cut / dropout remove the easy
alternative; coarse patches make the object signal easy (or help in another, still unknown way).

## 2. Facts, with the runs behind them

### 2.1 Fine patches + an arm the policy can rely on = bad

| cloud (4096 / 64 unless stated) | object patch | arm | success @250 |
|---|---|---|---|
| uncut | 2.5 cm | full | 8.5 (s42) / 8.0 (s43) |
| uncut 2048 / 64 | 3.4 cm | full | 14.0 |
| uncut 1024 / 64 | 4.5 cm | full | 23.0 / 19.0 |
| arm cut (link0-7) | 2.4 cm | removed | 20.0 / 16.5 / 16.5 (s42 / s43 / s44) |
| DROP (whole-part dropout keep 0.5, full cloud at test) | 2.5 cm | full at test | 13.0 and re-run 18.5 (s42), 17.0 (s43) |
| scenesparse (arm full, rest at its 1024 points) | 4.5 cm | full, fine | 17.5 (s42) |
| armsparse (arm at its 1024 points, rest full) | 2.3 cm | full, coarse | 12.0 (s42) |

- Uncut success follows the object patch size: ~21 / ~15 / ~8% at 4.5 / ~3.3 / 2.5 cm.
- Every arm cut lands at ~17-20 regardless of point count (1024 cut 17.5 / 20.5, 2048 cut 19.0, 4096 cut 16.5-20.0).
- Pooled over seeds 42 + 43: arm cut vs uncut +10.0 (62 / 22, p = 1.5e-5); DROP vs uncut +6.8 with DROP-42
  (p = 0.002), +9.5 with the re-run FTcont (p = 2e-5).
- scenesparse vs uncut 30 / 12 (p = 0.008); vs arm cut 23 / 28 (p = 0.58): fine arm + coarse objects is fine.
- armsparse vs uncut 17 / 10 (p = 0.25), vs arm cut 33 / 17 (p = 0.03): coarse arm + fine objects stays bad.

### 2.2 Coarse patches: the arm is harmless, cut / dropout add nothing

| 1024 / 64 | success |
|---|---|
| uncut | 23.0 / 19.0 |
| arm cut | 17.5 / 20.5 (pooled vs uncut 47 / 55, p = 0.49) |
| DROP (s42) | 22.0 (vs uncut 20 / 22, p = 0.88) |

1024 uncut relies on the arm exactly as much as 4096 uncut (round-7 translation test: link5-7 shift 5 cm ->
13.4 / 11.9 mm vs 12.4 mm) and still succeeds. Reliance on the arm is not harmful in itself.

### 2.3 What it is NOT

| candidate | evidence against |
|---|---|
| token reallocation (more tokens on the objects) | SPEC has GObal's allocation (nut 5.8 -> 9.5, peg 6.8 -> 14.6 tokens) but 11.0 / 11.5 vs GObal 19.5 / 18.0; DROP gains with uncut's allocation at test |
| token count | 1024: 32 / 64 / 128 tokens -> 19.0 / 23.0 / 20.5; 2048: 64 / 128 / 256 -> 14.0 / 15.5 / 16.5 (all n.s.; 2048 nut coverage 0.84 -> 0.99 moved nothing) |
| classical token dropout (uniform, p = 0.25) | 8.0% = uncut (11 / 12, p = 1.0); below DROP s43 (p = 0.010), FTcont (p = 0.003), arm cut (p = 0.001) |
| uniform point dropout | present in EVERY run (0-80% of points, training), uncut included |
| partial arm cuts | link5+6 11.0 / 11.5; link0-6 13.0 / 14.5 / 13.0; only the full arm cut reaches ~17-20 (pipeline noise caveat: SPEC07, the full cut through the remover pipeline, 13.0) |
| coarsening only the arm | armsparse 12.0 (see 2.1) |
| a points-per-token rule | same points per token, same full coverage, denser is worse: 1024/32 vs 2048/64 19 vs 14; 1024/64 vs 2048/128 23 vs 15.5 (p = 0.028); 1024/128 vs 2048/256 20.5 vs 16.5 |
| the cut acting at inference | uncut network + arm cut at test only: Square 6.0 vs 8.5 (10 / 15, p = 0.42); hanging_mug 45.0 -> 0.5 (1 / 90) |
| robustness to unseen content | table kept in the cloud: DROP and uncut both 0 / 200 (the table takes 53 of 64 tokens; a token-starvation demo, not a test) |

### 2.4 Coverage and patch size per configuration (stored training frames, 64-token FPS, k = 32 unless stated)

| points / tokens | nut / peg coverage uncut | with arm cut | object patch | success uncut | success cut |
|---|---|---|---|---|---|
| 1024 / 16 | 0.53 / 0.79 | | 4.9 cm | not run | |
| 1024 / 32 | 0.86 / 0.94 | | 4.7 cm | 19.0 | |
| 1024 / 64 | 1.00 / 1.00 | 1.00 / 1.00 | 4.5 cm | 23.0 / 19.0 | 17.5 / 20.5 |
| 1024 / 128 | 1.00 / 1.00 | | 4.4 cm | 20.5 | |
| 2048 / 64 | 0.84 / 0.88 | 0.99 / 1.00 | 3.4 cm | 14.0 | 19.0 |
| 2048 / 128 | 0.99 / 1.00 | 1.00 / 1.00 | 3.3 cm | 15.5 | queued |
| 2048 / 256 | 1.00 / 1.00 | | 3.1 cm | 16.5 | |
| 4096 / 64 | 0.48 / 0.57 | 0.83 / 0.93 | 2.5 cm | 8.5 / 8.0 | 20.0 / 16.5 / 16.5 |
| 4096 / 128 | 0.81 / 0.92 | 0.99 / 1.00 | 2.4 cm | queued | queued |
| 4096 / 256 | 0.99 / 1.00 | 1.00 / 1.00 | 2.3 cm | queued | queued |
| 4096 / 64, k = 128 | 1.00 / 1.00 | | 4.7 cm | queued | |

Note: 4096/128 uncut has the coverage of 4096/64 cut (0.81 / 0.92 vs 0.83 / 0.93): a coverage-matched
"more tokens vs cut" test. Sources: `scripts/sim/coverage_grid.py`, `scripts/sim/claim_sweep_mechanism.py`.

### 2.5 Other tasks

| task | uncut | arm cut | dropout | 1024 / 64 uncut | reading |
|---|---|---|---|---|---|
| hanging_mug (4096 / 64, 600 ep, shared list) | 45.0 | 60.0 | 55.5 (54.0 with the cut at eval) | 61.0 (1024 cut 44.5) | same pattern as Square; at 1024 the cut hurt |
| Threading D1 narrowed (4096 / 64) | 37.5 | 35.0 | GObalAll 36.5 | 35.5 | nothing helps; large, easy tripod -- nothing to starve |

hanging_mug 4096/64 -> 128 tokens: 42.0 -> 50.5 (old list), but the arm cut at 64 beats 128 (37 / 20, p = 0.03).

## 3. Hypotheses and how they are being tested

**H1 (main, measured as a fact):** with fine object patches, an arm the policy can rely on hurts; removing it or
making it unreliable restores the coarse-patch level. Open part: is "coarse" really patch size, or full coverage?

**H2 (mechanism, not measured): shortcut learning.**
- H2a: cut / dropout force the policy to learn the hard object features (nut orientation).
- H2b: coarse patches make those features easy, so coarse-patch policies read the nut's orientation too.
  If H2b fails, route A works through something else (two different mechanisms).

Supporting already: partial cuts barely help (the rest of the arm is still a shortcut); val loss does not track
success (the shortcut keeps the loss low); the translation test shows equal object sensitivity everywhere (position
is the easy part); Threading gains nothing (easy object).

## 4. Runs still queued and what each decides

Chain `scripts/sim/run_final_queue.sh` (log `R3D/data/outputs/sim/mimicgen/final_queue.log`), then
`scripts/sim/run_yaw_sens.sh`. All seed 42. Token runs read `R3D/data/outputs/sim/mimicgen/token_budget/queue.txt`
(remove a line to skip it). Estimated end of training ~Mon 02:00, yaw test a few hours later.

| run | status / ETA | decides |
|---|---|---|
| E1 uncut s43 + arm cut at inference | running (Sat ~18:30) | C3 second seed |
| 4096 / 128 uncut | Sat ~23:30 | coverage at 4096 (0.48 -> 0.81, fine patches). Coverage-matched against 4096/64 cut |
| 4096 / 128 cut | Sun ~04:00 | does the cut's gain survive 2x tokens |
| 4096 / 256 uncut | Sun ~11:00 | **full coverage with fine patches**: coverage predicts ~17-20, coarseness predicts ~8-12 |
| 2048 / 128 cut | Sun ~15:00 | cut at full coverage with medium patches (optional, user) |
| 4096 / 256 cut | Sun ~21:00 | cut at full coverage with fine patches (optional, user) |
| **4096 / 64, k = 128** | Mon ~02:00 | **coarse patches at full density and full coverage, arm present**: coarseness predicts ~17-20 |
| nut-yaw sensitivity (all policies) | Mon morning | H2a (cut / DROP react more to the nut's yaw than uncut) and H2b (do 1024 / scenesparse / k128 react too?) |

Decision table for H1 (coarseness vs coverage):

| 4096/256 uncut | 4096/64 k128 | reading |
|---|---|---|
| low (~8-12) | high (~17-20) | **coarseness** of the object patches is the cause, not coverage |
| high | high | coverage is enough; coarseness not needed |
| low | low | neither alone; only the combination as in 1024 (or something else in density) |
| high | low | coverage, and k = 128 patches are too coarse / something breaks; check |

Decision table for H2 (yaw test; smoke on ~30 windows, NOT a result: rotation change at 45 deg uncut 2.9, arm
cut 4.8, FTcont 5.1, scenesparse 2.9 deg):

| cut / DROP vs uncut | 1024 / scenesparse / k128 vs uncut | reading |
|---|---|---|
| more | more | one mechanism: both routes lead to reading the object's orientation (H2a + H2b) |
| more | same | two mechanisms: cut / dropout force object learning; coarse patches help some other way |
| same | any | H2a not supported; the shortcut story needs another readout than yaw |

Disk: the claim sweep's 200.ckpts were deleted (user, 2026-11-14; uncut s43, armsparse, scenesparse, DROP-1024 --
250.ckpts and evals kept) -> 25 GB free; the token block (6 x 2.15 GB) ends at ~12 GB.

## 5. Claims for the presentation (status today)

Decision rule: effect = mean gap >= 6 AND pooled p < 0.05; not detected = gap < 5 and p > 0.05; else inconclusive.

| claim | status | wording |
|---|---|---|
| C1 at partial coverage / fine patches, training with the arm always present at full resolution is worse than without the arm | holds, 2 seeds | "+10 points from removing the arm (2 seeds, p = 1.5e-5); hanging_mug +15" |
| C2 removing / randomising the arm fixes it | holds, 2 seeds each | as C1; dropout +6.8 to +9.5 |
| C2 coarsening the arm fixes it | not detected (1 seed) | drop it: "coarsening only the arm recovers at most a small part (+3.5, n.s.)" |
| coarse object patches fix it (scenesparse, 1024) | effect (1 seed) | "coarse, fully covered object patches reach the same level, even with the full arm" -- coarse vs coverage after k128 / 256 |
| C3 the fix acts in training, not at inference | holds (Square s42, mug; s43 running) | "cutting the arm from a network trained with it gives nothing (Square) or breaks it (mug); the dropout network works with or without the arm" |
| C4 coverage | being replaced | token count / coverage alone did not help at 1024 / 2048; final word after 4096/128 / 256 and k128 |
| C5 at coarse patches neither cut nor dropout helps | holds (cut 2 seeds, DROP 1 seed) | "at 1024 points the cut (pooled p = 0.49) and dropout (p = 0.88) change nothing" |
| token budget | pending | the claim is relative (cut vs uncut at the same budget); 64 is the tighter budget used to show it |

Do not claim: a points-per-token rule; token reallocation as the mechanism; "the arm distracts" (we measured
dependence, not attention); "more object-centric" (cut / dropout lower the arm dependence, object sensitivity under
translation barely changes; 1024 is not less arm-dependent at all).

## 6. Results of the queued runs (finished 2026-11-16; seed 42 unless noted, 250.ckpt, 200 paired scenes)

**Coarseness vs coverage (H1).**

| run | coverage nut / peg | object patch | success | paired |
|---|---|---|---|---|
| 4096 / 64 uncut | 0.48 / 0.57 | 2.5 cm | 8.5 / 8.0 | |
| 4096 / 128 uncut | 0.81 / 0.92 | 2.4 cm | 11.0 | vs 64: 17 / 12, p = 0.46 |
| 4096 / 256 uncut | 0.99 / 1.00 | 2.3 cm | 14.0 | vs 64: 20 / 9, p = 0.061 |
| **4096 / 64 k = 128** | 1.00 / 1.00 | 4.7 cm | **19.0** | vs 64: 35 / 14, p = 0.004; vs 256: 27 / 17, p = 0.17; vs arm cut 26 / 28, p = 0.89; vs 1024 21 / 29, p = 0.32 |

-> **Coarseness is the main lever, coverage a smaller one**: coarse patches at full density reach the arm-cut / 1024
level; full coverage with fine patches gets about half way (8.5 -> 11 -> 14 with tokens, monotone, each step n.s.).
Matches the plan's original C4 wording ("coverage has a separate, smaller effect").

**Cut vs more tokens.** Coverage-matched: 4096/64 arm cut 20.0 vs 4096/128 uncut 11.0 (31 / 13, p = 0.010). The cut at
64 tokens is still the best 4096 configuration; 4096/256 uncut (14.0) does not reach it (29 / 17, p = 0.10).

**The cut at more tokens.** 4096/128 cut 14.0 (vs 128 uncut +3.0, p = 0.41); 4096/256 cut 17.5 (+3.5, p = 0.37);
2048/128 cut 20.5 (vs 2048/128 uncut +5.0, p = 0.21). The cut's gain shrinks from ~+10 at 64 tokens to +3-5 (n.s.) at
128 / 256: uncut catches up partly via coverage, the cut does not improve with tokens (14.0-20.5). 4096/128 cut (14.0)
is low against 64 cut (20.0 / 16.5 / 16.5; 15 / 27, p = 0.09): within the ~4-5 point run noise, single seed.
-> the gain is largest when the token budget is tight; say so.

**E1, second seed.** uncut s43 + arm cut at inference: 3.0 vs 8.0 (5 / 15, p = 0.041). Pooled s42 + s43: -3.8 (15 / 30,
p = 0.036) -> cutting at inference never helps, if anything it hurts; C3 stands (with hanging_mug 45.0 -> 0.5).

**Nut-yaw sensitivity (H2), all 594 val windows** (`R3D/data/outputs/sim/mimicgen/yaw_sensitivity/sensitivity_yaw.md`):
change of the executed goal rotation for a +-45 deg nut rotation, pure effect, all phases (early < 0.3 in brackets);
floors 0.46-0.85 deg.

| policy | rot change (deg) | success |
|---|---|---|
| uncut 4096 s42 / s43 | 1.38 (2.2) / 1.93 (3.6) | 8.5 / 8.0 |
| armsparse | 1.88 (4.1) | 12.0 |
| 4096/128 uncut | 2.15 (4.4) | 11.0 |
| scenesparse | 2.19 (5.0) | 17.5 |
| 1024 uncut s42 / s43 | 2.23 (5.2) / 2.35 (5.2) | 23.0 / 19.0 |
| DROP s42 / FTcont / DROP s43 | 2.60 (5.3) / 2.58 (5.3) / 3.09 (6.6) | 13.0 / 18.5 / 17.0 |
| arm cut s42 / s43 / s44 | 2.56 (5.8) / 3.21 (7.9) / 3.17 (7.6) | 20.0 / 16.5 / 16.5 |
| 4096/256 uncut | 2.66 (5.7) | 14.0 |
| 4096/64 k128 | 2.69 (6.2) | 19.0 |
| DROP 1024 | 3.39 (8.1) | 22.0 |

- uncut 4096 reacts least to the nut's orientation (1.4-1.9 deg); every route that helps reacts 1.2-2.5x more:
  cut / dropout (H2a) AND the coarse-patch policies 1024 / scenesparse / k128 (H2b). The stride-20 smoke test
  (scenesparse = uncut) was wrong -- 30 windows.
- Across the 16 policies, yaw reaction ranks success: Spearman 0.45 (all phases, permutation p = 0.04), 0.52 before
  the grasp; the 2 cm nut SHIFT does not (0.13). Position is easy for every policy; orientation separates them.
- -> **one mechanism for both routes**, as far as this readout goes: the policies that work read the nut's orientation;
  uncut 4096 reads it least. Consistent with shortcut learning; still a correlation, not a causal test.

**Token geometry on the objects (`scripts/sim/object_token_geometry.py`, 767 frames; figure
`R3D/data/outputs/sim/mimicgen/token_geometry/object_token_geometry.png`).** 4096/256 k32 and 4096/64 k128 have the
same slot budget (8192 = 2x the points), so they separate size from coverage and overlap:

| config (success) | nut coverage | patches per nut point | nut points in >= 2 patches | nut tokens | share of the nut per token | radius |
|---|---|---|---|---|---|---|
| 4096/64 k32 (8.5) | 0.49 | 0.50 | 0.02 | 6.6 | 0.08 | 2.5 cm |
| 4096/256 k32 (14.0) | 0.98 | 1.94 | 0.65 | 26.6 | 0.08 | 2.1 cm |
| 4096/64 k128 (19.0) | 1.00 | 2.16 | 0.73 | 6.6 | 0.30 | 5.0 cm |
| 1024/64 k32 (23.0) | 1.00 | 2.07 | 0.71 | 6.7 | 0.29 | 4.8 cm |

(peg alike.) Gaps and overlap are equal for the three full-coverage configs; only token SIZE differs.

**Statement agreed with the user (2026-11-16):** below full coverage, more coverage helps a little (4096: 8.5 / 11.0 /
14.0 at 64 / 128 / 256 tokens) -- not the main lever and not a claim. The main lever is token coarseness: tokens that
span a larger AREA (~5 cm, ~30% of the nut) perform better; the number of points inside does not matter (k128 with 128
points 19.0 vs 1024 with 32 points 23.0, n.s.) -- FPS samples the surface evenly, so 32 points already trace the shape.
Limits: single seeds (k128 vs 256 p = 0.17); "coarse" is relative to the object (~5 cm on Square), not "bigger is
always better"; a fine-detail task may differ.

## 7. Sparsity limit: 512 and 256 points (finished 2026-11-16; seed 42, 64 tokens, k = 32, uncut, 250.ckpt, 200 paired scenes)

Same box / uncut area, the first N points of the stored FPS-4096 cloud. Queue
`R3D/data/outputs/sim/mimicgen/sparsity/queue.txt`; geometry `scripts/sim/sparsity_geometry.py` (767 frames) ->
`R3D/data/outputs/sim/mimicgen/sparsity/geometry.md`. Object diameter = visible extent (max pairwise distance of its
points in the full 4096 cloud): nut 13.8 cm (CAD: ring 8.75 x 8.75 cm, 12.3 cm with handle), peg 14.1 cm (CAD 3.2 x
3.2 cm, 20 cm tall -- the visible diameter is its height). Patch diameter = 2 x median radius of tokens centred on it.

| points | success | nut pts / tokens / coverage | nut patch diam (/ nut diam) | share of nut per token | peg pts / coverage | peg patch diam (/ peg diam) |
|---|---|---|---|---|---|---|
| 4096 | 8.5 / 8.0 | 476 / 6.6 / 0.48 | 5.0 cm (0.36) | 0.08 | 374 / 0.57 | 5.4 cm (0.38) |
| 2048 | 14.0 | 235 / 6.7 / 0.84 | 6.8 cm (0.50) | 0.16 | 190 / 0.88 | 7.4 cm (0.52) |
| 1024 | 23.0 / 19.0 | 120 / 6.8 / 1.00 | 9.6 cm (0.69) | 0.29 | 97 / 1.00 | 9.6 cm (0.69) |
| **512** | **19.5** | 61 / 6.9 / 1.00 | 14.0 cm (1.02) | 0.50 | 50 / 1.00 | 14.4 cm (1.03) |
| **256** | **9.0** | 32 / 7.1 / 1.00 | 19.8 cm (1.43) | 0.80 | 25 / 1.00 | 35.4 cm (2.51) |

Paired: 512 vs 1024 s42 23 / 30 (p = 0.41; pooled with s43 45 / 51, p = 0.61) -> a plateau, not detectably worse.
256 vs 1024 8 / 36 (p = 2.5e-5; pooled 20 / 68, p = 2.8e-7), vs 512 14 / 35 (p = 0.004), vs 4096 13 / 12 (p = 1.0).
-> **inverted U**: success rises 4096 -> 1024, holds at 512, and collapses at 256 back to the 4096 level. The plateau
spans patch diameters ~0.7-1.0x the object; at 256 a single token holds 80% of the nut and the peg's patches reach
2.5x its extent (25 peg points < k = 32, so every peg patch must pull in table-side scene points). With ~7 tokens
each seeing most of the nut, there is no longer a part-level description of the nut to read its orientation from.
Single seed for 512 / 256; the 1024 -> 512 difference is within run noise.
