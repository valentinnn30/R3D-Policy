# Learned link remover v0: final design (2026-10-26)

Short reference for what was built, why, and which knobs are open for tuning. The brief is
`docs/learned_remover_v0.md`; measurements and reasoning are in
`docs/sim_token_allocation_learnings.md`. Code: `R3D/r3d/model/vision/link_remover.py`
(remover), `R3D/r3d/policy/dp3_link_remover.py` (policy, reward, diagnostics),
`R3D/r3d/common/link_remover_training.py` (schedule, logging).

## What it does

Per robot region, the remover decides once per training window whether to keep the region
(**gate**) and, if kept, how densely (**density**, FPS within the region). This happens
after preprocessing and before the encoder's token FPS. Token budget, k and FPS are
unchanged. φ (the remover) is trained jointly with θ (the policy) by REINFORCE with a
leave-one-out baseline. The reward is the policy's own action-prediction error.

## Design choices

| choice | what | why |
|---|---|---|
| regions | 8 learned: link5, link6, link7, hand f1side / centre / f2side, finger1, finger2. link0–4 are never thinned | link0–4 average 0–5 points/frame (in view ~5%), so a learned value would only drift. The hand is split 3 ways along the opening axis in its mesh frame (user: the gripper sits where cutting helps or hurts) |
| labels | per-point link cell from sim segmentation (`data/point_link`, 22 cells) | re-rendered clouds are bit-identical to the stored ones. Kinematics-based labels agree with segmentation on 98.7% of robot points (5 mm), which is the path for the real robot |
| random variables | **gate** g ~ Bern(sigmoid(a)) plus **density** ρ = sigmoid(z), z ~ N(μ, σ²); keeps g · round(ρn) points | FPS places tokens by surface **extent**. FPS thinning keeps extent, so density only changes patch size; only removal moves tokens (link5–7 at ρ = 0.02 still hold 28 of 35 tokens). With density alone φ would need z ≈ −7 to find the cliff |
| log P | Σ [log Bern(g) + g · log N(z)] | z doesn't touch the reward when the gate is closed, so it is left out (less variance) |
| bounds | μ ∈ [−9, 6], a ∈ [−5, 5] | the logits are bounded, never ρ or π. −9 removes any region below ~4050 points; ±5 keeps π ∈ [0.007, 0.993], so a closed region can recover |
| thinning | FPS within the region on metric xyz | a thinned region looks like the same region in a sparser cloud (C1's lever) |
| N_kept < M | fill back with removed region points (FPS) up to M = 64 | 0.2–1.2% of frames have < 64 non-robot points. The fill is deterministic given the draw, so REINFORCE stays valid |
| reward | x₀ error on executed steps 1–8, t ∈ {20, 50, 80}, no grad, **eval mode** (no point dropout / dropout) | the brief's proxy. Eval mode removes noise the shared seeds cannot |
| CRN | one seed per window, shared by its K masks: token-FPS start (a non-robot point), token shuffle, within-region FPS start (nested masks), fill start, ε | the same mask twice gives an exactly equal reward. Seeds are random (not fixed), so φ is not biased towards one token layout |
| estimator | RLOO, K = 4 masks × 128 windows (both micro-batches) every 4 optimizer steps | toy: lowest drift for neutral regions at a learnable speed |
| φ optimizer | Adam lr 3e-3 on (a, μ), separate from θ | the brief's 1e-5 does not move φ. Adam steps ~lr per step even on pure noise, so only a low lr limits drift |
| schedule | warm-up 20% (fixed φ: keep 0.8, ρ 0.8, σ 1) → φ phase (σ 1 → 0.2) → fine-tune 10% (φ frozen, θ on the fixed snapped mask) | θ sees thinned clouds before φ judges them. In the fine-tune θ trains on exactly the eval clouds |
| inference | gate open iff π ≥ 0.5; ρ = sigmoid(μ), snapped to 0 below 0.05; EMA θ + live φ | deterministic. The snap removes sparse leftovers, which would otherwise attract FPS tokens |

## Tunable knobs (`config/r3d_robotwin2_remover.yaml`)

| key | value | evidence / range to try |
|---|---|---|
| `training.link_remover.lr` | 3e-3 | toy over a full run: 3e-3 learns by step ~1350–2000, no neutral gate closes. 1e-2 is ~3× faster but 0–1 neutral gates close. 1e-3 has the least drift but learns ~3× slower |
| `policy.link_remover.k_masks` | 4 | 2 halves the reward cost (+~20% → +~10% run time at 1024). Toy: K = 4 learns faster at high noise |
| windows per φ step / `phi_every` | 128 / 4 | more windows = less noise. `phi_every` 2 doubles the updates (and the cost) |
| `policy.link_remover.init_keep` | 0.8 | exploration of removal (each region removed in 20% of draws). 0.6–0.9 |
| `policy.link_remover.init_rho` | 0.8 | starting density |
| `policy.link_remover.learn_density` | true | false = **gates only**: ρ = g, so a kept region keeps every point and μ gets no gradient. First tuning step after sweep 1, where the density slopes were weak and ρ drifted to 0.6–0.75 |
| `training.link_remover.warmup_frac` / `finetune_frac` | 0.2 / 0.1 | user: start with 20%, tune afterwards |
| `sigma_start` / `sigma_end` | 1.0 / 0.2 | toy: ending at 0.5 learned no worse. Untested on the real task |
| `policy.link_remover.snap_rho_below` | 0.05 | 0.02–0.1. Can also be applied afterwards to a trained checkpoint (inference only) |
| `policy.link_remover.reward_t` | [20, 50, 80] | the encoder dominates the cost, so fewer t values save almost nothing |
| `policy.link_remover.gate_max` / `mu_min` | 5 / −9 | lower `gate_max` means more exploration of closed regions |
| `region_groups` | 8 regions | a finer hand (6 cells) or fingers prox/dist are possible without re-rendering |

## What to read after a run (`scripts/sim/analyze_remover_run.py`)

- `report.md`:
  - final gates/densities, and the **removal effect per region**: mean advantage with the gate closed minus open, with CI and t. This is the reward's own verdict per region;
  - diagnostics;
  - mechanism (tokens, coverage, patch size per region / nut / peg);
  - success vs epoch against the baselines, and paired McNemar at 250.
- **Held-out check:** held-out agreement close to the floor means the training reward is trustworthy. Clearly lower means φ learns from memorisation.
- **Diagnostic B:** low means the proxy reward ranks masks differently from the sampled action.

## Commands

```bash
python scripts/sim/check_link_remover.py              # all unit / policy / CRN checks (~1 min)
python scripts/sim/toy_reinforce_remover.py [--compare]
setsid nohup bash scripts/sim/run_remover_sweep.sh R3D/data/outputs/sim/mimicgen/remover_queue.txt \
    > R3D/data/outputs/sim/mimicgen/remover_sweep_driver.log 2>&1 &
python scripts/sim/analyze_remover_run.py <run_dir> [...]
python scripts/sim/viz_clouds.py <zarr> --links --rho link6=0,link7=0.05
```

## Open

- The density lever: in the toy it has no effect. In the real run, patch size might matter (C1).
- Seeds: 42 only in the first sweep. A region counts as learned when it agrees across seeds.
- The real-robot path (kinematics-based labels) is measured but not used in training.
