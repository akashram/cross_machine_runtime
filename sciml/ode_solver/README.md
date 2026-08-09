# ode_solver

**Status: code-complete AND locally run — pure CPU, no external
dependency.**

## What this measures

PLAN.md Phase 18 step 1: a small ODE solver library (explicit Euler,
classical RK4, implicit backward Euler) for initial value problems
`dx/dt = f(t, x)`, verified against closed-form solutions rather than just
"it compiles and runs" — the foundation the rest of Phase 18 builds on
(step 2's stiffness analysis reuses these solvers directly; step 4's Neural
ODEs reuse the same `State`/`Rhs` types).

## Design

- `State = std::vector<double>` and `Rhs = std::function<State(double,
  const State&)>` throughout, so the same three solvers work unmodified on
  scalar systems (exponential decay), 2D systems (harmonic oscillator), and
  eventually a network-parameterized right-hand side (step 4).
- **Backward Euler's implicit equation is solved with real Newton
  iteration**, not a linear-system special case: each step solves
  `g(y) = y - x_n - dt*f(t_{n+1}, y) = 0` for `y` via Newton's method with a
  central-finite-difference Jacobian of `g` and a hand-written Gaussian-
  elimination linear solve (`detail::solve_linear`). This matters because
  the logistic-growth test case below has a genuinely nonlinear `f`, so
  Newton's method is actually exercised, not just its degenerate linear
  case.
- Three test ODEs, each chosen for a specific property being checked:
  1. **Exponential decay** (`dx/dt = -kx`) — linear, closed form
     `x0*exp(-kt)`; baseline correctness check for all three solvers.
  2. **Harmonic oscillator** (`x0'=x1, x1'=-omega^2 x0`) — closed form
     `cos(omega t)`; used to verify RK4's empirical 4th-order convergence
     rate directly (halve `dt`, error should drop ~16x) against Euler's
     ~2x (1st order), not just assert it.
  3. **Logistic growth** (`dx/dt = r*x*(1-x/K)`) — genuinely nonlinear `f`,
     closed form `K/(1+((K-x0)/x0)*exp(-rt))`; the test that actually
     exercises Newton's method inside `backward_euler`.

## Results (captured 2026-08-09, Apple clang 14 / `-std=c++2b`, this Mac)

```
  closed form x(2.0) = 0.148547 | euler err=1.69e-03 rk4 err=6.00e-14 backward err=1.69e-03
PASS  explicit_euler matches closed-form exponential decay
PASS  rk4 matches closed-form exponential decay to near machine precision at dt=1e-3
PASS  backward_euler matches closed-form exponential decay
PASS  rk4 is orders of magnitude more accurate than explicit_euler at identical dt
  halving dt: rk4 error ratio=16.11 (expect ~16, 4th order) | euler error ratio=1.96 (expect ~2, 1st order)
PASS  rk4 shows ~4th-order convergence (error ratio > 10 when dt halves)
PASS  explicit_euler shows ~1st-order convergence (error ratio ~2 when dt halves)
  closed form logistic x(3.0) = 9.091066 | backward_euler err=5.85e-05 rk4 err=4.78e-11
PASS  backward_euler's Newton solve converges on a genuinely NONLINEAR implicit equation and matches closed-form logistic growth
PASS  rk4 matches closed-form logistic growth
PASS
```

## Findings

- RK4's empirical convergence order came out at 16.11x per `dt` halving —
  almost exactly the theoretical 2^4 = 16 for a 4th-order method — while
  explicit Euler came out at 1.96x, almost exactly the theoretical 2^1 = 2
  for a 1st-order method. Both are real measurements of the same two
  solvers on the same problem, not separately-tuned numbers.
- **A real bug in the test, not the solver**: the first version of this
  test asserted forward/backward Euler's error at `dt=1e-3` on the
  exponential-decay problem must be `< 1e-3`. The actual error came out to
  `1.69e-3` for both — correct behavior for a 1st-order method (global
  error is O(dt) with a problem-dependent prefactor, here driven by
  `k=1.3`), just not `< 1e-3`. Fixed by loosening the bound to `5e-3`
  rather than the solver code, since RK4's near-machine-precision result on
  the identical problem confirms the solvers themselves are correct.
- Backward Euler and forward Euler land on nearly identical accuracy
  (`1.69e-3` both) on this non-stiff problem, as expected — backward
  Euler's value isn't higher accuracy at the same order, it's stability at
  large `dt` on STIFF problems, which step 2 measures directly.

## Hardware notes
None — pure CPU.
