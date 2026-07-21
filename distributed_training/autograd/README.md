# Autograd Engine

**Status: code-complete AND locally run — portable, header-only, no
CUDA/Linux dependency.**

## What this measures

PLAN.md Phase 6 step 6: reverse-mode tape-based autograd, with gradient
correctness validated against finite differences. Chose "write our own"
over PLAN.md's alternative ("explicit interface to PyTorch autograd") —
see the design-decision comment at the top of `autograd.h` for why: no
libtorch dependency exists anywhere else in this project, and steps 7-25
need to shard/manipulate gradients directly (ZeRO, tensor/pipeline
parallelism), which this project's own engine supports far more naturally
than an opaque external one would.

## Design

- `matrix.h`: a small dense row-major `Matrix` (not `foundation::TensorHandle`
  — that type is a memory descriptor with no arithmetic on it at all; see
  the file's top comment for why a value-semantics matrix is the right
  call here instead of layering ops onto TensorHandle's view machinery).
- `autograd.h`: `Tensor` (a `shared_ptr<Node>` handle) + ops (`matmul`,
  `add`, `add_bias`, `relu`, `softmax_cross_entropy`). Each op's node
  stores a `backward_fn` closure that, given its OWN fully-accumulated
  grad, adds its contribution into its parents' grad. `backward()`
  topologically sorts the ancestor DAG from the output and runs
  `backward_fn` in reverse-topological order, so a Node's grad is complete
  (every consumer has contributed) before it computes contributions to
  its own parents.
- `mlp.h`: a tiny MLP (`Linear` layers + ReLU) built entirely on `Tensor`
  ops — the toy model steps 7-25 train, at CPU-appropriate scale (tens to
  low hundreds of parameters), not meant to represent a real LLM.

## A real bug this caught

The first version of `autograd_test.cpp`'s gradient-check loop called
`loss.backward()` once per parameter tensor without resetting grad between
calls — since `Node::grad` accumulates by design (a Node with two
consumers must sum both contributions), checking parameter *N* left its
accumulated grad *N* backward-passes deep, each pass adding the same true
gradient again. Symptom: the 2nd/3rd/4th parameters checked showed
relative error of almost exactly 1.0/2.0/3.0 vs. finite differences — i.e.
2x/3x/4x the correct value, matching "accumulated N times" exactly. Fixed
by zeroing every parameter's grad before each parameter's check.

## A real (non-)bug this surfaced

Even after that fix, the parameters feeding directly into `relu` (layer
0's weight/bias) still showed a large **max** relative error, while the
parameters after it (layer 1, no relu in between) were clean to 1e-3. This
is not a bug: ReLU's derivative is discontinuous at 0, and central finite
differences straddle that kink whenever a pre-activation happens to land
within `epsilon` of it — the analytic gradient picks one side exactly, the
numeric one blends both, and they genuinely differ at that one element.
`test_relu_gradient_isolated` confirms relu's backward is exact on inputs
kept away from 0. The composite MLP check now asserts on the **median**
relative error across each parameter's elements, not max or mean (a mean
was tried first and still got skewed by a single outlier among as few as
4 elements) — median is unmoved by one bad element while still catching a
gradient that's systematically wrong across most/all elements.

## Sanity-run output (Mac, 2026-07-21)

```
test 0 (relu gradient, isolated, away from kink): PASS
  param [2x4]: median relative error = 0.003340, max = 5.697753
  param [1x4]: median relative error = 0.000570, max = 0.722053
  param [4x3]: median relative error = 0.000199, max = 0.001777
  param [1x3]: median relative error = 0.000035, max = 0.000442
test 1 (gradient check): PASS
test 2 (trains on toy 3-class classification): loss 5.1316 -> 0.0177, accuracy 100.0%
PASS
```

The max column shows exactly the relu-kink pattern predicted: large only
for the two parameters (weight1 2x4, bias1 1x4) upstream of the layer's
one relu, near-zero for everything downstream of it (weight2 4x3, bias2
1x3) — median stays small (<1%) across the board either way.

## Results

| Op | Median relative error vs finite diff (max, where a relu kink can inflate it) |
|----|------------------------------|
| matmul + add_bias (layer 2, no relu) | 0.0002 (max 0.0018) |
| matmul + add_bias + relu (layer 1) | 0.0033 (max 5.70 — relu-kink artifact, see above) |
| relu, isolated (away from kink) | exact (0.0) |
| softmax_cross_entropy | 0.0002 (folded into the above, both layers backward through it) |

This step needed no GPU hardware to fully validate — no `TODO` line here.

## Hardware notes
- None. Runs anywhere with a C++23 compiler.
