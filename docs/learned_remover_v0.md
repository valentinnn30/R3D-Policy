# Learned link remover (C2): v0 work package

This document is the brief for implementing and testing the learned point remover (C2) in the MimicGen sim setup. MimicGen simulation and the uniform R3D baseline already exist. C1 is in progress separately. The operating point (N, k, M) comes from the C1 setup and stays fixed for every comparison here.

**Scope of v0:** one learnable density per robot link, trained jointly with the policy in one run. The voxel/MLP version (v1) is described at the end. Do not implement it in the first version.

---

## 1. Remover v0: spec

The remover thins or removes robot points **before** token FPS. Token budget M, kNN k and FPS itself are unchanged.

- **Regions:** robot links (Panda: ~7 arm links + hand + 2 fingers, about 10 regions).
- **Random variable:** the **density of each link**, one draw per link per training window. It is not an on/off decision per point.
  - $z_l \sim \mathcal N(\mu_l,\ \sigma^2)$
  - $\rho_l = \mathrm{sigmoid}(z_l)$ (logit-normal keep rate in (0, 1))
  - μ_l sets where the density sits. σ sets how much it varies between draws (exploration).
- **Parameters φ:** one learnable logit μ_l per link (~10 parameters).
- **Bounds:** clamp μ_l to **[−8, 6]** after every φ step (or parameterise with tanh).
  - At μ = −8, ρ ≈ 0.0003, so a link with fewer than ~1500 points keeps 0 points. Complete removal is allowed.
  - Bound μ, never ρ. A ρ floor would prevent complete removal. An unbounded μ could drift to −∞, after which the link is never seen again and φ can never recover it.
- **Applying a draw:** keep `round(ρ_l · n_l)` points of link l, **chosen by FPS within the link**. The kept link then looks like that link in a lower-resolution cloud, which matches the C1 thinning lever.
- Use the **same mask for both observation frames** of a window (n_obs_steps = 2).
- **Non-robot points** (object, scene) are always kept.
- **Constraint:** N_kept ≥ M. Fail loudly if violated.
- **Mask probability:**
  $\log P_\phi(S) = \sum_l \log \mathcal N(z_l;\ \mu_l,\ \sigma^2)$, so $\nabla_{\mu_l}\log P_\phi = (z_l - \mu_l)/\sigma^2$
- **Inference:** deterministic, $\rho_l = \mathrm{sigmoid}(\mu_l)$, no sampling.
  - Pipeline: label points → thin each link via FPS within the link → FPS to M tokens → kNN → encoder → DDIM (10 steps), exactly as in the baseline.
  - v0 is **static**: the same per-link table in every frame, whatever the robot state.

## 2. Point → link labels

- **v0:** exact labels from the simulator's geom segmentation, stored per point next to the cloud.
- Also compute the **FK nearest-link assignment** (point-to-mesh distance plus a threshold) and log how often it agrees with the segmentation.
- The FK path is what the real robot will use later. Keep a distance threshold so that object points close to the fingers are not labelled as robot.

## 3. Joint training of θ and φ (one run)

**θ (policy):** the standard R3D diffusion loss, computed on clouds after the sampled mask.

**φ (remover):** REINFORCE with a leave-one-out baseline. Removal is a discrete selection (and FPS/kNN only output indices), so φ gets no backprop gradient.

- **K = 2 masks per window.** Two independent density draws for the same window, both evaluated with the **same noise** (same t values, same ε). The loss difference then measures only the effect of the mask, as a paired comparison.
- **Reward (A):** x₀-prediction error ‖A⁰ − Â‖² on the **executed action steps 1–8 only** (per sample, `reduction='none'`), averaged over **fixed t ∈ {20, 50, 80}**, shared across the K masks.
  - The encoder runs once per mask. The head runs once per t.
  - Compute the reward pass under `torch.no_grad()`. The reward is detached anyway.
- **Advantage:** $\mathrm{adv}_{ik} = \mathcal L_{ik} - \mathrm{mean}_{j \ne k}\,\mathcal L_{ij}$ (detached). With K = 2, that's $\mathcal L_{i1} - \mathcal L_{i2}$ and its negative.
- **Surrogate loss:** $\mathcal L_\phi = \mathrm{mean}_{i,k}\big[\mathrm{adv}_{ik} \cdot \log P_\phi(S_{ik})\big]$
- `(loss_theta + loss_phi).backward()` gives both gradients cleanly: the diffusion loss only reaches θ, and log P only reaches φ. Use **two optimizers**, with lr_φ ≈ 0.1 × lr_θ.
- **φ update frequency:** every 4 steps (K = 2 on those steps, K = 1 with θ only on the others). If the ρ curves move too slowly after warm-up, go to every 2 steps.
- **Warm-up (same run):** sample masks from a fixed φ (all μ_l = logit 0.8, σ = 1.0) and update only θ until the loss levels off. Then switch the φ updates on. θ must have seen thinned clouds before φ starts judging them.
- **σ schedule:** anneal 1.0 → 0.2 after warm-up.
- **Diagnostic (B), not a reward:** every few epochs, on a validation batch, compute the error of the **fully sampled action** (10 DDIM steps, same starting noise for all masks) on steps 1–8. Log whether A and B rank the masks the same way. If they don't, the reward proxy is suspect.
- **Optional comparison:** compute the reward on ~10% held-out demos instead of the training batch, to check for memorisation.
- **Eval:** EMA θ plus deterministic ρ.

## 4. Implementation guidance

- **Remover off by default.** With the remover disabled, training and eval must be bit-identical to the baseline.
- New code goes in its own module (e.g. `r3d/model/vision/link_remover.py`) and is enabled via a config flag. Do not edit baseline configs.
- `compute_loss` needs per-sample losses (`reduction='none'`) and a variant restricted to executed steps 1–8 for the reward.
- **Variable cloud size:** reuse the robot-cut padding path (`point_cloud_len`, which masks FPS/kNN padding).
- Same mask for both observation frames of a window.
- Fail loudly on contract mismatches (missing link labels, N_kept < M, unknown link IDs), in line with the existing repo style.
- The **toy REINFORCE test (step 0) must pass before any GPU run.**

## 5. Test plan

### Step 0: checks and toy test (no policy training)
1. **Per-link statistics** on the dataset: points per link, robot share of the cloud. Use these to check that μ_min = −8 really means 0 points for every link.
2. **Remover unit test:** a fixed ρ per link gives the expected kept count per link, works with both obs frames, and passes cleanly through the `point_cloud_len` padding path.
3. **Toy REINFORCE test:** replace the diffusion loss with a synthetic reward, e.g. the loss drops when link 3 is thinned and rises when the fingers are thinned. φ must learn exactly that pattern within a few hundred updates. This checks the gradient sign, the RLOO baseline and the variance before any GPU time is spent.

### Step 1: learned remover v0 run
- One run: warm-up with random masks from fixed φ, then φ updates on.
- First 1 seed, then 3 seeds.
- **Success criteria:**
  1. ρ_l converges per link instead of oscillating.
  2. The learned pattern is plausible (e.g. arm links low, hand/fingers high) or explainably different.
  3. Rollout success ≥ the uniform baseline at identical N, k, M (and close to the C1 crop oracle if available).

## 6. Evaluation and logging

- **Evaluation:** same protocol as the baselines.
- **Log per run:**
  - μ_l and ρ_l = sigmoid(μ_l) per link over training, plus σ
  - advantage spread (std of adv)
  - reward A and diagnostic B on the validation batch, and their agreement in ranking masks
  - points kept per link, coverage fraction, token-centre spacing on gripper/object, kNN radius
  - FK-vs-segmentation label agreement
  - success vs. epoch

## 7. Known risks (keep in mind when reading results)

- The diffusion loss is only a **proxy reward**. Final claims rest on rollout success, not loss.
- **Memorisation:** on training windows, a mask can look good because θ memorised the window. The held-out reward is the check.
- **Co-adaptation:** θ adapts to whatever φ removes. A removal can look harmless only because θ compensates.
- **Lock-in:** a link driven to μ_min is rarely revisited. Watch the ρ curves after warm-up.
- **Warm-up matters:** if θ never saw thinned clouds, every removal looks bad and φ learns "keep everything".
- **Contact region:** object points within a few mm of the fingers must not get robot labels. This matters once FK labels replace segmentation.

## 8. Later (do NOT implement in v0)

- **v1: finer regions within a link.**
  - Replace the per-link constant μ_l with a small MLP μ_φ(link ID, voxel centre in that link's own frame), e.g. 2 cm voxels.
  - This allows different densities within a link (fingertip dense, finger base thin).
  - The MLP shares information across voxels, so neighbouring voxels get similar values.
  - Cost: many more random variables, so a noisier REINFORCE signal. Only worth it if v0 shows that per-link resolution is too coarse.
- **State-conditioned μ** (e.g. joint velocity or gripper phase): a dynamic remover, e.g. thinning the arm more while in transit.
- Token-budget sweep: full vs. learned remover.
- Second task, then dexterous hands.
