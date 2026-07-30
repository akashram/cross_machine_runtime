# fgsm

**Status: code-complete AND locally run — pure CPU, no external
dependency.**

## What this measures

PLAN.md Phase 14 step 2: FGSM (Goodfellow et al. 2014) — perturb the
input by `epsilon * sign(gradient)` to maximize loss, using step 1's
`input_gradient()`.

## Design

- `fgsm_attack()` is a single gradient-ASCENT step (the opposite
  direction from training's gradient descent): `perturbed = x + epsilon *
  sign(d(loss)/d(x))`. `sign(0) = 0`, matching the standard definition.
- One step, unlike PGD's (step 3) iterative refinement — the classic
  "cheap, one-shot" attack from the original paper.
- Deliberately scoped to MECHANICS here (the epsilon bound is exact, the
  attack really does increase loss on a trained model, epsilon=0 is a
  no-op) — the full epsilon sweep and accuracy-collapse curve is step 4's
  job, not duplicated here.

## Results (captured 2026-07-29, Apple clang 14 / `-std=c++2b`, this Mac)

```
PASS  every perturbed entry moves by exactly +-epsilon
  clean loss = 0.0140, FGSM-perturbed loss (epsilon=0.50) = 0.0498
PASS  FGSM perturbation genuinely increases loss (real gradient ascent, on a trained model)
PASS  epsilon=0 is a no-op (perturbed input identical to the original)
```

## Findings

- A single FGSM step at `epsilon=0.5` (on a task whose classes are
  separated by ~5.7 units in feature space — see
  `adversarial/task/classification_task.h`'s blob centers) already more
  than triples the loss (0.0140 -> 0.0498) on a well-trained classifier —
  confirms the perturbation is genuine gradient ascent, not just a random
  nudge that happens not to hurt.
- Every perturbed coordinate moves by exactly `epsilon` in magnitude (the
  L-infinity attack budget FGSM is defined by), confirmed directly rather
  than assumed from the formula.

## Hardware notes
None — pure CPU.
