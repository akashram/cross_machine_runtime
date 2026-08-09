# sde_solver

**Status: code-complete AND locally run — pure CPU, no external
dependency.**

## What this measures

PLAN.md Phase 18 step 3: Euler-Maruyama and Milstein solvers for scalar
SDEs `dx = a(t,x) dt + b(t,x) dW`, verified two different ways —
Monte Carlo weak convergence (matching known closed-form moments) and
pathwise strong convergence (matching a known exact solution on the SAME
underlying Brownian path).

## Design

- Both solvers take a **pre-generated sequence of standard normal
  increments**, not an RNG drawn lazily — the only way to run
  Euler-Maruyama and Milstein (or the same method at two different `dt`)
  against the literal same underlying Brownian path, which is what strong
  (pathwise) error comparisons require.
- Milstein's correction term needs `b'(x)`, estimated via central finite
  differences — same convention as `ode_solver.h::backward_euler`'s
  numerical Jacobian and `stiffness.h`'s numerical Jacobian: no analytic
  derivative required from the caller.
- **Test 1 — Ornstein-Uhlenbeck** (`dx = theta*(mu-x)dt + sigma*dW`,
  additive/state-independent noise): Monte Carlo mean/variance across 3000
  independent paths checked against the closed-form OU moments (weak
  convergence). Also checks a genuine structural fact directly: since
  `b(x)=sigma` is constant, `b'=0`, so Milstein's correction term vanishes
  and it must reduce to Euler-Maruyama EXACTLY on the same path — checked
  bit-for-bit (`0.00e+00` difference), not assumed from the math alone.
- **Test 2 — Geometric Brownian Motion** (`dx = mu*x dt + sigma*x dW`,
  genuinely multiplicative noise, so `b'(x)=sigma != 0`): GBM has an exact
  closed-form solution in terms of the driving Brownian path,
  `x(t) = x0*exp((mu-sigma^2/2)t + sigma*W(t))`, letting strong (pathwise)
  error be measured directly against ground truth rather than just against
  another Monte Carlo estimate.

## Results (captured 2026-08-09, Apple clang 14 / `-std=c++2b`, this Mac)

```
  OU closed form: mean=0.8647 var=0.0614 | Monte Carlo (3000 paths): mean=0.8703 var=0.0626
  max |Euler-Maruyama - Milstein| final state over 50 shared-path checks = 0.00e+00 (b'=0 for constant sigma)
PASS  Euler-Maruyama's Monte Carlo mean matches OU's closed-form mean
PASS  Euler-Maruyama's Monte Carlo variance matches OU's closed-form variance
PASS  Milstein reduces to Euler-Maruyama exactly for additive (state-independent) noise, on the SAME Brownian path -- a structural fact, checked directly
  GBM strong (pathwise) RMS error vs. exact solution, 2000 Monte Carlo paths:
    dt=0.020: euler-maruyama=0.0097  milstein=0.0004
    dt=0.010: euler-maruyama=0.0070  milstein=0.0002
    error ratio halving dt: euler-maruyama=1.38 (theory: strong order 0.5 -> ~1.41) | milstein=2.02 (theory: strong order 1.0 -> ~2.0)
PASS  at the same (coarser) dt, Milstein's strong error is smaller than Euler-Maruyama's on GBM's genuinely multiplicative noise
PASS  same comparison holds at the finer dt too
PASS  Milstein's error shrinks faster than Euler-Maruyama's when dt halves (higher strong order), measured not assumed
```

## Findings

- The empirical strong-convergence order came out almost exactly on
  theory: Euler-Maruyama's error ratio on halving `dt` measured 1.38
  against a theoretical `sqrt(2) ≈ 1.41` (strong order 0.5); Milstein
  measured 2.02 against a theoretical `2.0` (strong order 1.0) — both real
  Monte Carlo measurements (2000 independent paths per `dt`), not
  separately tuned to land near the textbook numbers.
- Milstein's absolute strong error on GBM is roughly **20x smaller** than
  Euler-Maruyama's at the same `dt` (`0.0004` vs `0.0097` at `dt=0.02`) —
  a real, large, measured accuracy gap on a genuinely multiplicative-noise
  problem, exactly where the extra correction term is supposed to matter.
- The `0.00e+00` exact-match result on OU is a structural sanity check, not
  a coincidence: it directly confirms Milstein's implementation correctly
  reduces to Euler-Maruyama when `b'=0`, on the same Brownian path used by
  both — a real, checkable relationship between the two solvers rather
  than an assumed one (the same spirit as Phase 14's PGD-reduces-to-FGSM
  structural check).

## Hardware notes
None — pure CPU.
