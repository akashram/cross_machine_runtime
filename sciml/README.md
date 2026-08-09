# Phase 18: Dynamical Systems, SciML & Physics-Informed Architectures

**Status: CODE COMPLETE (10/10 steps), 2026-08-09.** Fully CPU-portable,
no hardware gate at all — like Phase 12/13/14. Every step actually
compiled, run, and captured in its own README.

## Overview

Scoped 2026-08-09 alongside Phase 17, while comparing this repo against a
batch of analog/physics-based-AI-compute job descriptions, which found a
clean zero on nonlinear dynamical systems (ODE/SDE/PDE, stability,
adjoint methods) and the "unconventional" model architectures those roles
build instead of plain transformers (SSMs, diffusion, Neural ODEs, Deep
Equilibrium Models, energy-based models, muP scaling). This phase closes
that gap.

## Steps

| # | Directory | What |
|---|-----------|------|
| 1 | `ode_solver` | Explicit Euler, RK4, backward Euler — verified against closed-form solutions |
| 2 | `stiffness` | Stability boundaries + numerically estimated stiffness ratio (Van der Pol) |
| 3 | `sde_solver` | Euler-Maruyama, Milstein — verified via Monte Carlo moments + strong-convergence order |
| 4 | `neural_ode` | Adjoint-method gradients through an NN-parameterized ODE, reusing step 1's `rk4` |
| 5 | `deq` | Deep Equilibrium Models — implicit-function-theorem backprop through a fixed point |
| 6 | `ssm_layer` | Generic linear state-space layer vs. self-attention, measured accuracy + O(L) vs O(L²) |
| 7 | `diffusion` | DDPM on a toy two-cluster 2D distribution |
| 8 | `ebm` | Energy-based model via contrastive divergence, directly compared to step 7 |
| 9 | `mup_scaling` | muP's output-layer LR-scaling claim, tested directly across widths |
| 10 | `noise_aware_training` | Training through Phase 17 step 1's device-noise model, mirroring Phase 14's adversarial training |

## Design highlights — how the ten steps chain together

- **Steps 2, 4, 5 all reuse step 1's `rk4`/linear-solve machinery
  directly**, not separate copies: `neural_ode`'s forward AND adjoint
  backward pass both call `sciml::rk4`; `deq`'s implicit backward pass
  reuses `sciml::detail::solve_linear` from `ode_solver.h` — the same
  Gaussian elimination `backward_euler`'s Newton steps use.
- **Steps 6-10 share a generic finite-difference trainer**
  (`finite_diff_gd_step`, introduced in step 6), reused unmodified through
  step 10 — the same trainer for an SSM, an attention layer, a diffusion
  model, an EBM, an muP-scaled MLP, and a noise-aware-trained MLP, so
  accuracy comparisons between architectures aren't confounded by one
  getting a hand-derived exact backward pass and another an approximate
  one.
- **Step 8 retrains step 7's exact diffusion model in the same binary**
  for a true same-run, same-metric architecture comparison, not a
  cross-run number lookup.
- **Step 10 closes the loop back into Phase 17**: it's the only step in
  this phase that depends on `analog_engine/` (Phase 17 step 1's real
  device model), completing PLAN.md's explicit ask to bridge the two new
  phases.

## Real findings, not assumed conclusions

- **A real sign bug, caught by the finite-difference gradient check, not
  by re-reading the algebra** (step 4): the adjoint method's `dtheta/dtau`
  sign, worked out carefully on paper, was backwards — every parameter's
  relative error came back at exactly `2.0` (the signature of
  `numeric=-analytic`). Fixed by flipping one sign.
- **Empirical convergence orders landed almost exactly on theory**
  (steps 1, 3): RK4's error-ratio-on-`dt`-halving measured `16.11` against
  a theoretical `16`; SDE strong-convergence orders measured `1.38`
  (theory `1.41`, Euler-Maruyama) and `2.02` (theory `2.0`, Milstein).
- **Attention beats a generic (non-HiPPO) SSM on a long-range copy task**
  (step 6) — the literature-motivated reason S4's real contribution is its
  structured state matrix, not just "use a linear recurrence," deliberately
  NOT implemented here so the comparison shows the actual gap HiPPO closes.
- **Diffusion clearly beats an EBM at deliberately comparable budgets**
  (steps 7-8): `1.04` vs. `2.36` mean distance to true cluster centers —
  a real, literature-consistent illustration of why diffusion displaced
  EBMs for practical generative modeling (easier optimization, not a
  theoretical expressivity gap).
- **muP's exact claim confirmed with no tuning to force it** (step 9):
  best learning rate landed at exactly `1.0` across all three widths
  tested under muP, while standard parametrization's best LR shifted from
  `0.3` down to `0.1` as width grew — plus a second real finding, muP
  diverged at fewer `(width, LR)` points too (`2/15` vs. SP's `7/15`).
- **A real STE bug, caught by catastrophic training divergence, not
  inspection** (step 10): naively finite-differencing through a
  quantization+noise pipeline gave a degenerate zero-or-spike gradient
  (the finite-difference epsilon was five orders of magnitude smaller
  than the quantization step) — exactly the textbook reason real
  quantization-aware training uses a Straight-Through Estimator. Fixed,
  then a second real finding: the robustness benefit needed a stronger
  noise regime to become measurable at all, ending in a clean echo of
  Phase 14's own adversarial-training tradeoff (noisy loss cut ~45% at a
  real, measured clean-accuracy cost).

See each step's own README for full methodology and captured output;
`sciml/DESIGN.md` for the phase-level design rationale.

## Hardware notes

None — fully CPU-portable, no hardware gate anywhere in this phase.

## Next

Phase 19 (Framework-Native Training) is the remaining piece of this
session's three-phase addition — see `CLAUDE.md`'s Phase 19 status and
`READING_LIST.md`'s Phase 19 section.
