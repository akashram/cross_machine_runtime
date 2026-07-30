# adversarial_training

**Status: code-complete AND locally run — pure CPU, no external
dependency.**

## What this measures

PLAN.md Phase 14 step 5: adversarial training defense (Madry et al.
2017's min-max formulation) — replace each training batch with a
PGD-perturbed version instead of clean data.

## Design

- Every epoch: (1) inner maximization — craft a PGD-perturbed batch
  against the model's CURRENT weights (`pgd_attack`, step 3); (2) outer
  minimization — a normal gradient-descent step, but on that adversarial
  batch instead of the clean one. This alternation, every epoch, IS
  Madry et al.'s min-max formulation.
- **Reuses `full_training_loop`'s shape, not its distributed machinery**:
  that step's `grad_clipping.h` clips a norm computed via
  `ring_allreduce` across SHARDED gradients across multiple ranks — real
  machinery this single-process phase doesn't need (there is exactly one
  "rank," so the global norm IS the local norm). `clip_grad_norm_local()`
  is the identical "clip by norm to `max_norm`" RULE with the distributed
  collective removed, not a different rule — same forward -> backward ->
  clip -> step shape, single-process.
- A real API gap, worked around rather than added to the engine:
  `Tensor` exposes `mutable_value()` for optimizer updates but no
  `mutable_grad()` equivalent, since ordinary training code only ever
  READS `grad()`. Clipping needs to WRITE a scaled grad back, so
  `clip_grad_norm_local()` reaches through `Tensor::raw()` (also public,
  by design) rather than adding a new accessor whose only caller would be
  this one function.

## Results (captured 2026-07-29, Apple clang 14 / `-std=c++2b`, this Mac)

```
  grad norm before clip: 9.3263, after clip (max_norm=0.10): 0.1000
PASS  clip_grad_norm_local bounds the gradient norm to max_norm
  adversarially-trained model's clean accuracy: 1.000
PASS  the adversarially-trained model still reaches reasonable clean accuracy (training actually converged)
  robust accuracy under PGD (epsilon=1.50): undefended=0.722, adversarially-trained=0.844
PASS  the adversarially-trained model has measurably higher robust accuracy than the undefended model, at the SAME attack epsilon
```

## Findings

- The defense actually defends: at `epsilon=1.5` (chosen from step 4's
  sweep as a point where the undefended model has already lost a
  meaningful chunk of accuracy, 0.756 clean-eval / 0.722 same-config PGD
  here), the adversarially-trained model's robust accuracy (0.844) is
  clearly higher than the undefended model's (0.722) — the training-time
  min-max alternation measurably transfers to eval-time robustness.
- Training on adversarial examples the entire time still reaches 1.000
  clean accuracy on this task — no visible clean-accuracy cost YET at
  this epsilon. Step 6 looks specifically for whether that cost shows up
  at other points on the tradeoff curve (Tsipras et al. 2018's
  "robustness may be at odds with accuracy" finding), since a single
  epsilon here isn't enough to rule it out.

## Hardware notes
None — pure CPU.
