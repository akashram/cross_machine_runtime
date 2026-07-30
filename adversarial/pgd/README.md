# pgd

**Status: code-complete AND locally run — pure CPU, no external
dependency.**

## What this measures

PLAN.md Phase 14 step 3: PGD (Madry et al. 2017) — the stronger iterative
attack: multiple FGSM-style steps with projection back into an
epsilon-ball around the original input.

## Design

- `pgd_attack()` runs `num_steps` iterations of `x += step_size *
  sign(input_gradient(x))`, each followed by clipping every coordinate
  back into `[x_original - epsilon, x_original + epsilon]` — the standard
  L-infinity projection.
- **A real, checkable structural relationship, not just noted in a
  comment**: PGD with `num_steps=1` and `step_size=epsilon` degenerates
  to EXACTLY FGSM — a single full-budget step lands exactly on the
  epsilon boundary, so the projection clip is a no-op. Verified as
  byte-identical output between `pgd_attack(..., epsilon, epsilon, 1)`
  and `fgsm_attack(..., epsilon)`, not just argued.
- Each PGD iteration calls `input_gradient()` (step 1) fresh at the
  CURRENT perturbed point — the mechanism that lets it escape a single
  linear approximation's error, unlike FGSM's one-shot step.

## Results (captured 2026-07-29, Apple clang 14 / `-std=c++2b`, this Mac)

```
PASS  every perturbed entry stays within the L-infinity epsilon-ball around the original, after 10 iterations
PASS  PGD with num_steps=1, step_size=epsilon is EXACTLY FGSM (byte-identical output)
  FGSM (1 step, epsilon=0.50) loss = 0.0498
  PGD (10 steps, step_size=0.1, epsilon=0.50) loss = 0.0517
PASS  PGD (iterative, same epsilon budget) finds a loss at least as high as a single FGSM step
```

## Findings

- At the same L-infinity budget (`epsilon=0.5`), 10-step PGD finds a
  higher loss (0.0517) than a single FGSM step (0.0498) — a real, if
  modest, measured demonstration of why Madry et al. 2017 treat PGD as
  the stronger attack: iterating lets it correct for FGSM's
  single-linear-approximation error within the same budget. The gap is
  small here because this toy 2-feature task's loss surface is close to
  linear near the decision boundary at this scale — a real, disclosed
  property of this specific task, not evidence the PGD-vs-FGSM gap is
  generally this small (on higher-dimensional, more nonlinear models the
  literature reports much larger gaps).
- The FGSM-equivalence check (byte-identical, not approximately equal)
  is a strong correctness signal for both `pgd_attack()` and
  `fgsm_attack()` simultaneously: any bug that made them diverge at this
  degenerate configuration would very likely be a real implementation
  bug in one or the other, not a legitimate algorithmic difference.

## Hardware notes
None — pure CPU.
