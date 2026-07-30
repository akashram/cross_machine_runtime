# input_gradients

**Status: code-complete AND locally run — pure CPU, no external
dependency.**

## What this measures

PLAN.md Phase 14 step 1: expose gradients of loss with respect to the
input, not just weights.

## Design

- **No engine modification needed.** `distributed_training/autograd.h`'s
  reverse-mode tape never special-cases parameter Tensors vs. input
  Tensors — every op's `backward_fn` adds its contribution into every
  parent `Node`'s `grad` field uniformly. `distributed_training/
  full_training_loop` already wraps its input batch in a `Tensor` before
  calling `mlp.forward(x)`, so `x.grad()` already holds `d(loss)/d(x)`
  after `loss.backward()`. `input_gradients.h`'s job is making that a
  first-class, TESTED capability for attack code (steps 2/3), not
  discovering or building a new one.
- **A real correctness detail this file exists to handle**: repeated
  calls (PGD, step 3, calls this once per iteration) must not let the
  model's WEIGHT gradients silently accumulate across calls —
  `Tensor::backward()` only seeds the OUTPUT node's grad; it does not
  zero every ancestor's grad first (`add_inplace` onto whatever's already
  there is deliberate, so a `Node` used by two consumers sums both
  contributions — see `autograd.h`'s own design note). `input_gradient()`
  zeros the model's weight grads before every call, making each call
  self-contained regardless of prior calls.
- Uses a shared 3-class Gaussian-blob classification task
  (`adversarial/task/classification_task.h`) built on the SAME
  `distributed_training::MLP` class `full_training_loop` trains — the
  toy model this whole phase attacks, reused rather than reinvented.

## Results (captured 2026-07-29, Apple clang 14 / `-std=c++2b`, this Mac)

```
  median relative error (12 samples) = 0.004112
PASS  input_gradient() matches finite differences: input Tensors get real gradients from the existing tape, no engine change needed
  max |grad1-grad2| = 0.00000000, max |grad1-grad3| = 0.00000000
PASS  repeated input_gradient() calls on the same (x, labels) return identical gradients, not accumulated ones
PASS  weight-parameter gradients stay finite and bounded after repeated input_gradient() calls (no unbounded accumulation)
PASS
```

## Findings

- The finite-difference check confirms directly (not just by reading
  `autograd.h`'s code) that input Tensors receive real, correct
  gradients from the existing engine: median relative error 0.0041
  against numeric differentiation, well within the same tolerance every
  other gradient-checked component in this repo uses.
- Two repeated calls on identical inputs return byte-identical gradients
  (max diff `0.0`), confirming the zero-grad-before-backward design
  genuinely prevents cross-call accumulation — the failure mode that
  WOULD have silently corrupted PGD's iterative gradient computation
  (step 3) had it gone unhandled.

## Hardware notes
None — pure CPU.
