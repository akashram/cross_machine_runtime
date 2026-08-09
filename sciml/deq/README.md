# deq

**Status: code-complete AND locally run — pure CPU, no external
dependency.**

## What this measures

PLAN.md Phase 18 step 5: Bai, Kolter & Koltun (2019), *"Deep Equilibrium
Models"* — an implicit-depth layer defined by a fixed point
`z* = f_theta(z*, x)`, found via plain fixed-point iteration, with
backprop through the fixed point via the IMPLICIT FUNCTION THEOREM instead
of unrolling the iteration.

## Design

- `f_theta(z, x) = tanh(Wz*z + Wx*x + b)`, weights scaled down at init
  (`0.3/sqrt(n)`) specifically so the map is a contraction near its fixed
  point — plain (Picard) fixed-point iteration only converges reliably
  under that condition, not assumed by default.
- **Implicit-function-theorem backward pass, not unrolling**: differentiate
  `z* = f(z*,x;theta)` w.r.t. `theta` directly to get
  `(I - df/dz) * dz*/dtheta = df/dtheta`, then solve
  `(I - df/dz)^T v = dL/dz*` for `v` — reusing `sciml::detail::solve_linear`
  from step 1's `ode_solver.h` (the same Gaussian-elimination solve
  `backward_euler`'s Newton steps use) rather than writing a second linear
  solver — and compute `dL/dtheta = v^T * (df/dtheta)`. No differentiation
  through the fixed-point iteration's individual steps at all: O(1) memory
  in the number of iterations, the same shape as step 4's adjoint method
  being O(1) memory in the number of solver steps.
- Both `df/dz` and `df/dtheta` are numerical (central-difference)
  Jacobians, same convention as every other Jacobian in this phase.
- Two checks: (1) does the converged `z*` actually satisfy
  `z* = f(z*,x)` (residual, not just "the loop exited"); (2) does the
  implicit-function-theorem gradient match finite differences that
  RE-SOLVE the fixed point from scratch for each perturbed parameter.

## Results (captured 2026-08-09, Apple clang 14 / `-std=c++2b`, this Mac)

```
  fixed point converged=yes in 16 iterations | z*=[-0.0555, -0.3937, -0.0210] | max |f(z*,x)-z*|=4.63e-12
PASS  fixed-point iteration converges within max_iters
PASS  the converged z* actually satisfies z* = f(z*, x) (residual < 1e-8)
  gradient check over 21 parameters (each re-SOLVES the fixed point from scratch): median relative error=0.000000, max=0.000003
PASS  implicit-function-theorem dL/dtheta matches finite differences (re-solving the fixed point each time): median relative error < 1%
PASS  every parameter's gradient matches within 5%, not just the median
```

## Findings

- **No sign bug this time** — unlike step 4's Neural ODE adjoint (which
  had a real `dtheta/dtau` sign flip caught by its finite-difference
  check), this step's implicit-function-theorem derivation matched
  finite differences on the first run (median relative error
  `0.000000`, max `0.000003` across all 21 parameters). Worth stating
  plainly rather than only reporting bugs: the same "verify against
  finite differences, don't trust the hand derivation" discipline applies
  whether or not it catches something, and here it confirmed correctness
  rather than finding a defect.
- Fixed-point iteration converged in 16 steps to a residual of `4.6e-12` —
  the weight-scaling choice at init (`0.3/sqrt(n)`, keeping `f` a
  contraction) is what makes this reliable rather than a lucky seed; an
  unscaled random init would risk non-convergence, a real practical
  constraint DEQ training has to handle that this step's init explicitly
  addresses rather than ignores.
- The gradient check's max error (`3e-6`) landed noticeably above its
  median (`~0`) but still two orders of magnitude inside the 5% bound —
  consistent with ordinary finite-difference/floating-point noise
  compounding slightly more on some parameters (particularly ones with
  smaller true gradient magnitude, where the `max(1e-6, ...)` denominator
  floor in the relative-error calculation has more relative effect) rather
  than a systematic error.

## Hardware notes
None — pure CPU.
