# neural_ode

**Status: code-complete AND locally run — pure CPU, no external
dependency.**

## What this measures

PLAN.md Phase 18 step 4: Chen, Rubanova, Bettencourt & Duvenaud (2018),
*"Neural Ordinary Differential Equations"* — a small network parameterizes
`dx/dt = f_theta(x)`, integrated forward with step 1's existing `rk4`
solver (reused directly, not copied), with `dL/dtheta` computed via the
ADJOINT method: one augmented ODE solved backward in time, instead of
storing every intermediate activation the way naive backprop-through-
every-solver-step would.

**Scope note**: `f_theta(x)` is autonomous (no explicit `t` dependence) —
the full Neural ODE formulation allows `dx/dt = f(x,t,theta)`, but the
adjoint-method mechanics this step verifies don't need explicit time-
dependence to demonstrate, and dropping it keeps the Jacobian bookkeeping
simpler. A real, disclosed scope reduction.

## Design

- `f_theta(x) = W2 * tanh(W1*x + b1) + b2`, a tiny 1-hidden-layer network
  (state dim 1, hidden width 3 in the test — 10 parameters total).
- The adjoint backward pass integrates the augmented state
  `[x, a, theta_grad]` forward in REVERSED time `tau = t1 - t`, so the
  same `sciml::rk4` from step 1 can be reused unmodified — no separate
  "backward solver" needed. `dx/dtau = -f(x)` reconstructs the forward
  trajectory's `x(t)` on the fly (the celebrated O(1)-memory property of
  the adjoint method: no stored forward activations), while
  `da/dtau` and `dtheta_grad/dtau` accumulate the gradient alongside it.
- `df/dx` and `df/dtheta` are both numerical (central-difference)
  Jacobians — same convention as `backward_euler`'s Newton Jacobian and
  `stiffness.h`'s local Jacobian elsewhere in this phase.
- Two checks: (1) does the reconstructed `x0` from the backward integration
  match the REAL `x0` the forward pass started from (a free correctness
  check on the reconstruction, independent of the gradient); (2) does the
  adjoint-computed `dL/dtheta` match finite differences of the loss,
  RE-RUNNING the entire forward pass with each parameter perturbed (the
  same standard `adversarial/input_gradients` holds its gradient check to).

## Results (captured 2026-08-09, Apple clang 14 / `-std=c++2b`, this Mac)

```
  forward x(t1)=0.914928, loss=0.086083 | adjoint-reconstructed x0=0.200000 (true x0=0.200000), |err|=8.33e-17
PASS  adjoint backward pass reconstructs the ORIGINAL x0 (integrating -f(x) backward, no stored forward trajectory) to within 1e-3
  gradient check over 10 parameters: median relative error=0.000000, max=0.000000
PASS  adjoint-computed dL/dtheta matches finite differences of the actual re-run loss: median relative error < 1%
PASS  every parameter's adjoint gradient matches finite differences within 5% (not just the median)
```

## Findings

- **A real sign bug, caught by the test, not by re-reading the algebra.**
  The hand-derivation of `dtheta/dtau` (working through the `t -> tau`
  reversed-time substitution for the `dL/dtheta` line integral) concluded
  `dtheta/dtau = -a^T*(df/dtheta)`, and looked correct on paper. The first
  run of the finite-difference gradient check came back with EVERY one of
  the 10 parameters' relative error at exactly `2.0` — the unmistakable
  signature of `numeric = -analytic` (if `x = -y`, `|x-y|/|y| = |{-2y}|/|y|
  = 2` exactly), not a subtle per-parameter Jacobian bug (which would show
  varied, not uniform, errors). Flipping that one sign (to
  `dtheta/dtau = +a^T*(df/dtheta)`) brought every parameter's relative
  error to `0.000000` at the printed precision. Kept in the header as a
  case study: sign conventions in a reversed-time substitution are exactly
  the kind of easy-to-flip-once mistake a finite-difference check catches
  immediately and a paper derivation can hide indefinitely.
- The `x0` self-consistency check landed at `8.33e-17` — floating-point
  precision, essentially exact — confirming the reversed-time
  reconstruction of `x(t)` (the part of the adjoint method that gives it
  O(1) memory instead of storing every forward step) is implemented
  correctly, independent of and prior to the gradient bug above.

## Hardware notes
None — pure CPU.
