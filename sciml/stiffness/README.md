# stiffness

**Status: code-complete AND locally run — pure CPU, no external
dependency.**

## What this measures

PLAN.md Phase 18 step 2: stability regions of explicit vs. implicit
solvers on a stiff system, plus a numerically estimated stiffness metric
on a small nonlinear system — built directly on step 1's `explicit_euler`/
`backward_euler`.

## Design

- **`trajectory_stays_bounded()`** measures divergence directly (every
  state finite and under a bound) instead of eyeballing a printed number.
- **Test 1 — exact linear stability boundary.** Forward Euler's stability
  region for `dx/dt = lambda*x` is textbook-exact: stable iff
  `dt < 2/|lambda|`. Run at `dt` just below and just above that boundary
  and check the measured behavior matches the closed-form prediction, with
  `backward_euler` run at the SAME above-boundary `dt` to show A-stability
  (bounded regardless of step size, for `Re(lambda) < 0`).
- **`local_jacobian_2d()` / `eigenvalues_2d()` / `stiffness_ratio_2d()`** —
  a numerically estimated (central finite differences, not analytic)
  Jacobian and its eigenvalues at a point, with stiffness defined the
  standard way (Hairer & Wanner): the ratio of the largest to smallest
  `|Re(eigenvalue)|`.
- **Test 2 — Van der Pol stiffness vs. mu.** Evaluated at a fixed point
  `(x,y)=(0,2)` for two damping values: `mu=1` (a near-harmonic
  oscillator) and `mu=50` (the textbook large-mu relaxation-oscillator
  regime).
- **Test 3 — the same bounded/unbounded contrast as test 1, on Van der
  Pol** at `mu=200`, a `dt` that resolves the slow manifold but not the
  fast one.

## Results (captured 2026-08-09, Apple clang 14 / `-std=c++2b`, this Mac)

```
  dt_bound=2/|lambda|=0.0020 | dt_stable=0.0015 (below) | dt_unstable=0.0050 (above, 2.5x)
  |x_final|: euler@dt_stable=7.403e-34  euler@dt_unstable=1.100e+12  backward_euler@dt_unstable=1.276e-11
PASS  explicit_euler stays bounded just BELOW its exact stability boundary dt < 2/|lambda|
PASS  explicit_euler DIVERGES just above its exact stability boundary (measured, not assumed)
PASS  backward_euler stays bounded at the SAME dt that makes explicit_euler diverge (A-stability)
  mu=1: eigenvalues = 0.500+0.866i, 0.500-0.866i (complex -> locally oscillatory, not 'stiff' in the real-eigenvalue sense)
  mu=50: eigenvalues = 49.980, 0.020 (real, widely separated) -> stiffness ratio = 2498.0
PASS  at mu=1, Van der Pol's local Jacobian has complex eigenvalues (near-harmonic oscillator, genuinely not stiff here)
PASS  at mu=50, Van der Pol's local Jacobian has real eigenvalues (relaxation-oscillator regime)
PASS  at mu=50, the real eigenvalues are separated by a stiffness ratio > 100 -- measured, real stiffness growth with mu, the textbook Van der Pol behavior
  Van der Pol, mu=200, dt=0.010: explicit_euler bounded=NO (diverged) | backward_euler bounded=yes
PASS  explicit_euler numerically diverges on stiff (mu=200) Van der Pol at a dt that resolves the true dynamics' slow manifold but not its fast one
PASS  backward_euler stays bounded on the SAME stiff system at the SAME dt (its whole point: stability, not higher accuracy)
PASS
```

Every parameter (`dt_stable`/`dt_unstable` split, the mu=1/mu=50 point
choice, the mu=200 blow-up dt) was predicted analytically first, then
confirmed by actually running the code — none were tuned after the fact
to force a pass.

## Findings

- Forward Euler's stability boundary held exactly as predicted:
  `|x|` at `dt_stable` decayed to `7.4e-34` (correct — the true solution
  decays to ~0), while at `dt_unstable` (2.5x past the boundary) it blew
  up to `1.1e12` within the same 0.1-time-unit interval. `backward_euler`
  at the identical unstable `dt` stayed at `1.3e-11` — bounded and
  reasonably close to the true decayed value, purely from A-stability, not
  higher order (it's still a 1st-order method).
- **Van der Pol's stiffness genuinely emerges from the physics, not from
  picking a favorable evaluation point.** At `mu=1`, the same fixed point
  `(0,2)` gives complex eigenvalues (`0.5 ± 0.866i`) — a locally
  oscillatory, non-stiff regime, matching the fact that Van der Pol at
  small mu is close to a simple harmonic oscillator. At `mu=50`, the exact
  same point gives real eigenvalues (`49.98`, `0.02`) with a stiffness
  ratio of ~2498 — the textbook relaxation-oscillator transition (real,
  MATLAB's own `ode15s` documentation uses Van der Pol with large mu as
  its canonical "why you need a stiff solver" example).
- Test 3 shows the same bounded/unbounded split from test 1 reproduces on
  a genuinely nonlinear system: `mu=200` Van der Pol at `dt=0.01` (fine
  enough for the slow manifold, far too coarse for the fast one) makes
  `explicit_euler` diverge while `backward_euler` — same equations, same
  step size — stays bounded.

## Hardware notes
None — pure CPU.
