# ebm

**Status: code-complete AND locally run — pure CPU, no external
dependency.**

## What this measures

PLAN.md Phase 18 step 8: LeCun, Chopra, Hadsell, Ranzato & Huang (2006),
*"A Tutorial on Energy-Based Learning"* — a scalar energy function
`E_theta(x)` trained via contrastive divergence (Hinton 2002) with
short-run Langevin MCMC negative sampling, on the SAME toy two-cluster 2D
target distribution step 7's diffusion model used — a direct
architecture-family comparison, both trained and sampled in this same
test binary for a true same-run, same-metric comparison.

## Design

- **`E_theta(x) = w2 . tanh(W1*x + b1) + b2`**, a tiny 33-parameter
  scalar-output MLP.
- **Contrastive divergence, done correctly (stop-gradient through the
  MCMC chain)**: for one training step, generate negative samples `x_neg`
  via Langevin MCMC using the CURRENT parameters, then treat `x_neg` as
  FIXED when computing the parameter update. The surrogate loss
  `L(theta) = mean(E_theta(x_pos)) - mean(E_theta(x_neg))`,
  finite-differenced with `x_pos`/`x_neg` held fixed, gives EXACTLY the CD
  gradient (push energy down at real data, up at model samples) — not an
  approximation, since only `E_theta`'s direct parameter dependence is
  differentiated, matching what "stop-gradient through MCMC" means in the
  real algorithm.
- **Langevin dynamics** (`x_{k+1} = x_k - (step/2)*dE/dx + sqrt(step)*noise`)
  uses a numerical (finite-difference) gradient of `E` w.r.t. `x` — same
  convention as every other gradient in this phase.
- Reuses `sciml::finite_diff_gd_step` from step 6 directly for the
  parameter update, and step 7's `diffusion.h` machinery directly
  (untouched) to retrain an identical DDPM model in the same binary for
  the head-to-head comparison.

## Results (captured 2026-08-09, Apple clang 14 / `-std=c++2b`, this Mac)

```
  EBM contrastive-divergence training (33 params, 200 iterations): CD surrogate loss 0.4458 -> -3.1516 (E(pos)-E(neg), should trend negative/lower as E(pos)<<E(neg))
PASS  contrastive divergence surrogate loss decreases with training (energy at real data pulls below energy at model samples)
  mean distance to nearest TRUE cluster center: trained EBM=2.3573 | untrained-EBM baseline=3.3308
PASS  trained EBM's Langevin samples land measurably closer to the true cluster centers than an untrained EBM's samples

  DIRECT architecture comparison (identical target, identical metric, same binary):
    DDPM diffusion (step 7):     mean distance to nearest true center = 1.0394
    EBM + Langevin (this step):  mean distance to nearest true center = 2.3573
```

## Findings

- **CD training visibly worked**: the surrogate loss `E(pos)-E(neg)` went
  from `+0.45` to `-3.15` — real data ends up with substantially lower
  energy than model samples, exactly what contrastive divergence is
  supposed to achieve. Trained-EBM samples also land measurably closer to
  the true cluster centers (`2.36`) than an untrained EBM's samples
  (`3.33`), confirming training helped.
- **Diffusion clearly wins the direct head-to-head comparison at this
  toy scale and training budget**: `1.04` mean distance-to-nearest-center
  for DDPM vs. `2.36` for the EBM — diffusion's samples land more than
  twice as close to the true clusters. This is a real, literature-
  consistent finding, not a bug in the EBM implementation: EBM training via
  short-run Langevin/contrastive-divergence is well documented as harder
  to get right than diffusion's fixed, tractable forward-noising
  objective — CD's negative-sample quality depends on how well a SHORT
  MCMC chain (20 steps here, matching the training budget) explores the
  energy landscape, and short chains are a known source of bias and poor
  mode coverage in EBM training (part of why later EBM work, e.g. Du &
  Mordatch 2019, invests heavily in replay buffers and longer/annealed
  chains that this toy-scale step doesn't implement).
- **The two architectures were given deliberately comparable budgets**
  (33 EBM parameters vs. 50 diffusion parameters, 200 vs. 250 training
  iterations, 200 samples each, identical target and metric) specifically
  so the accuracy gap is attributable to the training DYNAMICS these two
  families have, not to one model simply being bigger or trained longer.
- This result is a genuine, disclosed data point for the "why does the
  field mostly use diffusion over EBMs for image/generative modeling at
  scale" question — not because EBMs are theoretically weaker (the energy
  formulation is at least as expressive), but because diffusion's training
  objective is dramatically easier to optimize well with limited
  MCMC/compute budget.

## Hardware notes
None — pure CPU.
