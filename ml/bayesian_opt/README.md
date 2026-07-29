# bayesian_opt

**Status: code-complete AND locally run — pure CPU, no external
dependency.** Uses real OpenML datasets reused from `ml/openml_bench`
for the real hyperparameter-tuning experiments.

## What this measures

PLAN.md Phase 12c step 15: Gaussian Process surrogate, Expected
Improvement acquisition function, upper confidence bound. Benchmarked
against random search and grid search.

## Design

- `GaussianProcess` (`gp.h`/`gp.cpp`): exact GP regression (Rasmussen &
  Williams, GPML Algorithm 2.1) — RBF kernel, Cholesky-factorized
  `(K + noise*I)`, forward/back substitution instead of an explicit
  matrix inverse. Internal linear algebra runs in `double` (same
  rationale as `pca.cpp`'s own note: Cholesky factorization is
  numerically sensitive the same way randomized SVD is), public API is
  `float`.
- `expected_improvement`/`upper_confidence_bound`: standard acquisition
  functions, **maximization form** (every metric this repo's benchmarks
  produce — accuracy, purity, explained variance — is higher-is-better;
  a minimization-oriented version would need negation at every call
  site, so this deliberately diverges from the original stub's
  minimization-oriented, named-`SearchSpace` API — see `bayesian_opt.h`'s
  own Design note).
- `BayesianOptimizer::optimize`: `n_initial_random` random points, then
  `n_iterations` GP-guided steps — each step fits a GP to all points
  observed so far, evaluates the acquisition function at
  `n_candidates` (default 500) uniformly random candidate points, and
  evaluates the true objective at whichever candidate scored highest.
- `random_search`/`grid_search`: the two baselines this step is
  benchmarked against, with `grid_search` generalized to N dimensions
  (full Cartesian product of `n_points_per_dim` evenly-spaced values per
  dimension).

## Results (captured 2026-07-28, Apple clang 14 / `-std=c++2b`, this Mac)

```
  at observed x=2.0: predicted mean=0.9090 (true=0.909), std=0.001000
PASS  GP posterior mean matches an observed point's value almost exactly
PASS  GP posterior std is near-zero at an observed point (noise_variance only)
  far from any observed point (x=10.0): std=1.0000
PASS  GP posterior std is larger far from any observed data than at an observed point
  EI(mean=0.5,std=0,best=0.9)=0.0000, EI(mean=0.5,std=0.1)=0.0000, EI(mean=0.85,std=0.1)=0.0169
  EI(mean=0.5,std=0.05)=0.0000, EI(mean=0.5,std=0.5)=0.0580
PASS  EI is exactly zero when there is no uncertainty and the mean can't beat best_so_far
PASS  EI increases with a higher predicted mean (holding std fixed)
PASS  EI increases with higher predictive uncertainty (holding mean fixed)
  UCB(mean=0.5,std=0.1)=0.7000, UCB(mean=0.7,std=0.1)=0.9000, UCB(mean=0.5,std=0.5)=1.5000
PASS  UCB increases with a higher predicted mean
PASS  UCB increases with higher predictive uncertainty (kappa rewards exploration)
  found x=1.9976 (true optimum x=2.0), value=-0.0000 (true optimum value=0.0)
PASS  Bayesian optimization finds a point close to the true optimum within 20 total evaluations
```

`bayesian_opt_bench` (real hyperparameter tuning + the needle-in-haystack
experiment, see Findings):

```
=== Synthetic needle-in-haystack: narrow spike at x=3, range [-5,5], 15-evaluation budget, 10 seeds ===
  grid search:  x=2.857 value=0.1969 (found=0)
  seed=1  BO value=0.9884 (found spike)  random-search value=0.0959
  [9 other seeds: BO value ~0.099-0.100, random-search value ~0.091-0.100, neither finds the spike]
  spike found: BO 1/10, random search 0/10, grid search 0/1

=== blood-transfusion: tuning SVM(RBF) gamma only ===
  Bayesian optimization: gamma=0.66587, val_acc=0.8188 (15 evaluations)
  Random search:         gamma=0.39080, val_acc=0.8188 (15 evaluations)
  Grid search:           gamma=0.26827, val_acc=0.8188 (15 evaluations)

=== breast-w: tuning SVM(RBF) gamma only ===
  Bayesian optimization: gamma=0.70289, val_acc=0.9712 (15 evaluations)
  Random search:         gamma=0.39080, val_acc=0.9784 (15 evaluations)
  Grid search:           gamma=0.71969, val_acc=0.9712 (15 evaluations)

=== wdbc: tuning SVM(RBF) gamma only ===
  Bayesian optimization: gamma=0.01767, val_acc=0.9823 (15 evaluations)
  Random search:         gamma=0.01767, val_acc=0.9823 (15 evaluations)
  Grid search:           gamma=0.00518, val_acc=0.9823 (15 evaluations)

=== blood-transfusion: tuning SVM(RBF) C and gamma jointly ===
  Bayesian optimization: val_acc=0.8188 (16 evaluations)
  Random search:         val_acc=0.8255 (16 evaluations)
  Grid search:            val_acc=0.8188 (16 evaluations)

=== breast-w: tuning SVM(RBF) C and gamma jointly ===
  Bayesian optimization: val_acc=0.9640 (16 evaluations)
  Random search:         val_acc=0.9784 (16 evaluations)
  Grid search:            val_acc=0.9640 (16 evaluations)

=== wdbc: tuning SVM(RBF) C and gamma jointly ===
  Bayesian optimization: val_acc=0.9823 (16 evaluations)
  Random search:         val_acc=0.9823 (16 evaluations)
  Grid search:            val_acc=0.9823 (16 evaluations)
```

## Findings — an honest complication of PLAN.md's "BO beats random search" premise

- **The fundamentals are correct, verified directly**: the GP posterior
  mean matches an observed point almost exactly with near-zero
  uncertainty there, and is meaningfully more uncertain far from any
  observed data; EI and UCB both increase monotonically with predicted
  mean and with predictive uncertainty, exactly as their formulas
  require; on a simple, single-optimum synthetic function, the full
  `BayesianOptimizer` finds `x=1.9976` against a true optimum of `x=2.0`
  within 20 total evaluations — essentially exact.
- **On real hyperparameter tuning (SVM's `gamma`, and jointly `C` and
  `gamma`, across 3 real datasets), Bayesian optimization does *not*
  reliably beat random search** — the opposite of PLAN.md's stated
  Definition of Done. In the 1D case, all three methods land on
  essentially the same best validation accuracy (within 0.007) on every
  dataset. In the 2D case, random search actually *wins* on 2 of 3
  datasets, and by a real margin on `breast-w` (0.9784 vs. BO's 0.9640).
  This isn't a bug: it matches a well-known, published empirical result
  (Bergstra & Bengio, "Random Search for Hyper-Parameter Optimization,"
  2012) — random search is a surprisingly strong baseline specifically
  when the objective landscape has a wide, forgiving optimum and the
  search space is low-dimensional, exactly the regime `hyperparam_sensitivity`'s
  own `gamma` sweep already showed for these datasets (accuracy stays
  high across a wide swath of `gamma` before collapsing only at the
  extremes).
- **The needle-in-haystack experiment reveals the real, disclosed reason
  why, specific to this implementation**: on a synthetic function with a
  narrow spike (the kind of landscape where a smart search *should*
  matter), Bayesian optimization only found the spike in 1 of 10 seeds
  — and diagnosing that one success shows it only happened because one
  of the 5 *initial random* points happened to land close enough to the
  spike for the GP to notice it was there at all. On the other 9 seeds,
  BO performed no better than random search. **Root cause**: this
  implementation approximates acquisition-function maximization by
  evaluating it at `n_candidates` (500) *uniformly random* points per
  iteration, rather than a proper local optimization of the acquisition
  surface (e.g. multi-start L-BFGS, as production BO libraries use).
  Away from any existing observation, the GP's posterior mean and
  variance are nearly identical everywhere (both revert to the prior),
  so every unobserved candidate looks statistically the same to the
  acquisition function — nothing points the search toward a feature the
  GP hasn't already stumbled onto. Bayesian optimization can only
  *refine* a promising region it already knows about; this
  implementation's random-candidate acquisition maximization can't
  *discover* one on its own the way a properly-optimized acquisition
  step could.
- **The honest overall picture**: this implementation's GP/EI/UCB
  machinery is correct and does what the textbook says (verified
  directly), and it reliably solves easy, single-optimum problems. But
  its practical advantage over random search on both real
  hyperparameter surfaces and a synthetic hard case is much smaller than
  PLAN.md's Definition of Done assumes — a genuine, measured limitation
  worth knowing about a naive-acquisition-maximization BO
  implementation, not a result to paper over.

## Hardware notes
None — every result above is a real, measured number from this Mac.
Real dataset hyperparameter tuning reuses `ml/openml_bench`'s already-
committed data.
