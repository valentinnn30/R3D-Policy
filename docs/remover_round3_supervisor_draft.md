# Supervisor summary draft (updated 2026-11-02 23:45, round 3 complete)

Round 4 (DROP, DObal) runs overnight; add its result before sending if wanted.

---

**Density.** I trained a density-only remover starting from 4096 points. Besides the robot parts, it also
had a region for the whole scene minus the robot (label-free). The scene density dropped to 25%, which is
roughly the density of the 1024-point clouds there; the nut and peg reached the same coverage as at 1024.
The fingers also became clearly sparser (to 51% and 67%). The arm and hand barely moved (76–88%, from
90%). Success was 13% (16% at an earlier checkpoint), between the 4096 baseline (8.5%) and the 1024
policy (23%). It wasn't significantly better than 4096 and was clearly worse than 1024. Two takeaways:
- The training loss is not well aligned with closed-loop success.
- Many density changes hardly affect the loss. The remover's own reward signal for the arm and hand was
  weak, which is why their density hardly changed.

**Removal: a variant that works, but differently than expected.** The policy network is trained on masks
drawn with a fixed removal probability of 0.5, so it sees every robot part removed half the time. The
learned removal probabilities are trained in parallel from the policy's prediction error. They do not
decide what the policy trains on, which avoids the co-adaptation that blocked earlier versions. Only in the
final fine-tuning phase does the policy train on the learned mask.

| success (200 paired scenes, epoch 250) | seed 42 | seed 43 |
|---|---|---|
| learned removal | 19.5% | 18.0% |
| oracle arm cut | 20.0% | 16.5% |
| policy trained from scratch on the learned mask | 11.0% | 11.5% |
| no removal | 8.5% | – |
| learned-removal network, evaluated on the full cloud | 16.0% | 18.0% |

- **The learned mask reproduces:** both seeds remove the upper arm links (link5, link6) and keep the last
  arm link, the hand and the fingers.
- **Success matches the oracle arm cut** on both seeds (pooled p = 0.92) and is clearly better than no
  removal (p < 0.001).
- **Training from scratch on exactly that mask is clearly worse** (11% on both seeds; pooled p = 0.002 vs
  the learned removal), although the token allocation is identical.
- **The gain is in the network, not in the mask:** the learned-removal network keeps its success when it
  sees the full, uncut cloud at test time (16–18%, no significant difference to its mask), and on that
  identical input it beats the policy trained without removal (16 vs 8.5%, p = 0.02). So random robot-part
  dropout during training seems to be what helps; the learned mask adds no measurable gain at test time.

**Next (running tonight):**
- dropout alone: the same random part dropout for the whole training, no learned removal, full cloud at
  test time. If it matches, the method is effectively a structured dropout augmentation;
- the same decoupled training for density (the density counterpart of the method above);
- then other tasks.
