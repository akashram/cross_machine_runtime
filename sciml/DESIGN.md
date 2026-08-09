# Dynamical Systems, SciML & Physics-Informed Architectures — Design

## 1. Why this phase needed no hardware-gate split

Unlike Phase 3/7/8/15/17, every one of this phase's ten steps is pure
numerical C++ over `std::vector<double>` state — ODE/SDE solvers, small
hand-rolled networks, finite-difference gradients. Nothing here needs a
GPU, an accelerator toolchain, or even a Python environment. That's why
all ten steps are actually compiled and run on this Mac, with real
captured output in every README, the same way Phase 12/13/14 needed no
hardware deferral either.

## 2. The reuse chain: a genuinely shared foundation, not ten separate models

Step 1 (`ode_solver`) is the load-bearing foundation for half the phase:

- Step 2 (`stiffness`) calls `explicit_euler`/`backward_euler` directly on
  a stiff test system.
- Step 4 (`neural_ode`) calls `rk4` TWICE — once for the forward pass,
  once (in reversed time) for the adjoint backward pass — rather than
  writing a second solver.
- Step 5 (`deq`) reuses `sciml::detail::solve_linear`, the same
  Gaussian-elimination routine `backward_euler`'s Newton iteration uses,
  for its implicit-function-theorem linear solve.

Step 6 (`ssm_layer`) introduces a second shared piece —
`finite_diff_gd_step`, a generic finite-difference trainer — that steps
7-10 all reuse unmodified:

- Step 7 (`diffusion`) trains its noise-prediction network with it.
- Step 8 (`ebm`) trains its energy function with it, AND retrains step
  7's exact diffusion model in the same test binary for a same-run
  comparison (not a cross-run number lookup, which would leave open
  "were these actually comparable runs").
- Step 9 (`mup_scaling`) extends it to per-parameter-group learning rates
  (a small, additive change, not a rewrite) to test muP's core mechanism.
- Step 10 (`noise_aware_training`) needed to move AWAY from the shared
  trainer for its actual parameter update (see section 4) — but still
  reuses step 9's exact MLP and regression task, keeping the departure
  minimal and load-bearing rather than a wholesale rewrite.

## 3. Two real gradient-correctness bugs, both caught by verification, not trust

This phase's entire methodology is: verify every non-obvious gradient or
convergence claim against ground truth (closed-form solutions, Monte
Carlo moments, finite differences) rather than trust a hand derivation or
a paper's stated result. That discipline caught two real bugs:

**Step 4's adjoint sign bug.** The `dtheta/dtau` sign, derived carefully
on paper by substituting reversed time into the standard adjoint
parameter-gradient integral, came out backwards. Every one of 10
parameters' finite-difference check reported relative error at EXACTLY
`2.0` — the unmistakable signature of `numeric = -analytic` (if
`x = -y`, then `|x-y|/|y| = |-2y|/|y| = 2` exactly), not a scattered,
noisy set of errors a subtle Jacobian bug would produce. Flipping the one
sign brought every parameter to ~0 relative error. Kept in the source
comment as a case study: a paper derivation can hide a sign error
indefinitely; a finite-difference check catches it on the first run.

**Step 10's Straight-Through-Estimator necessity.** Naively finite-
differencing a loss THROUGH a weight-quantization-plus-noise pipeline (the
natural first thing to try, given every other step in this phase computes
gradients by finite-differencing straight through whatever function is in
front of it) diverged catastrophically. The reason is structural, not a
coding mistake: quantization is a piecewise-constant (staircase) function,
and a finite-difference epsilon much smaller than the quantization step
sees either an EXACT zero (both perturbed evaluations land on the same
step) or a spurious spike (a perturbation that happens to cross a step
boundary) — never the smooth, true gradient a differentiable function
would give. This is precisely why the real quantization-aware-training
literature uses a Straight-Through Estimator instead of differentiating
through `round()`. The fix — compute the gradient w.r.t. the
ALREADY-NOISY weights (a smooth function with no discreteness inside it),
then apply that gradient to the clean parameters, treating
`d(noisy)/d(clean) ≈ 1` — is a real, if approximate, algorithmic choice
the ML systems literature actually makes, not a repo-specific workaround.

## 4. Where a step's finding required real tuning, not just a first try

Step 10's SECOND real finding (after the STE fix) is worth being explicit
about: at the first NON-diverging settings tried (`num_levels=16`, 120
training iterations), noise-aware training showed no measurable
robustness benefit — the baseline model was already accidentally robust
enough to that mild a noise level that there was no real gap for
noise-aware training to close. The benefit only became measurable after
increasing the noise regime (`num_levels=8`) and training budget. This is
reported in the step's own README as a real tuning step, not smoothed
over into "and then it worked" — the same standard this repo holds
elsewhere (e.g. Phase 15 step 2's compression-ratio-bound test fix,
Phase 8's hyperparameter-optimization phase finding random search
competitive on real data rather than assuming sophistication always wins).

## 5. Deliberately reproducing known weaknesses, not working around them

Two steps deliberately do NOT implement the "fixed" version of their
architecture, because the whole point of the comparison is to reproduce
the KNOWN weakness the more sophisticated real version addresses:

- Step 6's SSM uses a generic random state matrix, not S4's HiPPO
  initialization — so it genuinely loses to attention on a long-range
  copy task, the exact failure mode HiPPO exists to fix. Implementing
  HiPPO init would have made the comparison less informative, not more.
- Step 8's EBM uses plain short-run Langevin/contrastive divergence, not
  the replay-buffer/longer-chain machinery later EBM work (Du & Mordatch
  2019) adds specifically to fix short-chain MCMC's known bias — so it
  genuinely loses to diffusion, the real reason the field mostly moved to
  diffusion for practical generative modeling.

## 6. Disclosed simplifications, by step

- **Step 4**: `f_theta(x)` is autonomous (no explicit `t` dependence) —
  the adjoint mechanics being tested don't need it.
- **Step 5**: plain fixed-point (Picard) iteration, not Broyden's method
  (noted as a possible upgrade).
- **Step 6**: a directly-implemented standard self-attention layer, not
  `distributed_training/tensor_parallel_attn`'s Tensor/autograd-based
  module (built for a different phase's training pipeline).
- **Step 7**: DDPM only, not flow matching (PLAN.md phrases the step as
  "and/or").
- **Step 9**: tests muP's output-layer-LR-scaling mechanism specifically,
  not the full multi-parameter init+LR table across every layer type.

## 7. What this phase does and doesn't claim

This phase claims: real, verified implementations of the core mechanisms
behind Neural ODEs, DEQs, SSMs, diffusion, EBMs, and muP, each checked
against a ground truth (closed-form solution, Monte Carlo moment,
finite-difference gradient, or a documented literature-consistent
comparison result) rather than assumed correct from the paper alone — and
two real, disclosed bugs (an adjoint sign error, a quantization-gradient
pathology) caught by that verification discipline, not hidden.

It does NOT claim: state-of-the-art performance on any of these
architectures, or that the toy-scale results here (2D synthetic
distributions, ~10-70 parameter networks) generalize in magnitude to
production scale — only that the underlying MECHANISMS are implemented
correctly and behave the way the literature says they should.
