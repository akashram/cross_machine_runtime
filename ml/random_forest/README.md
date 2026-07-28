# random forest

**Status: code-complete AND locally run — pure CPU, no external
dependency, TSan clean on the parallel fit path. The `vs sklearn/
LightGBM on OpenML CC-18` comparison stays TODO — deferred pending the
install decision covering that benchmark suite (see project memory);
real correctness/parameter-effect results below don't depend on it.**

## What this measures

PLAN.md Phase 12a step 2: bagging (bootstrap aggregation), per-split
random feature subsampling (`mtry`), out-of-bag (OOB) error estimation,
permutation feature importance, and parallel tree fitting.

## Design

- Built directly on `decision_tree`'s `DecisionTree` (step 1), reusing
  its `TreeParams::max_features`/`random_state` addition unchanged — that
  addition exists specifically so this step gets Breiman's actual
  per-split-node feature subsampling, not a once-per-tree subset (a
  different, weaker technique).
- `RFParams::max_features` convention: `>= 1.0` is an absolute feature
  count, `(0, 1)` is a fraction of `n_features`, `-1` (default) is
  `sqrt(n_features)` — the classification default PLAN.md names.
- Bootstrap sampling: each tree draws `n` samples with replacement from
  the training set using a per-tree `std::mt19937` seeded from
  `random_state + tree_index`. `in_bag_[t][i]` records whether sample `i`
  was drawn for tree `t`'s bootstrap sample — the ~36.8% of samples not
  drawn (the classic `(1 - 1/n)^n -> 1/e` bootstrap-exclusion fraction)
  are that tree's OOB set.
- Parallel fitting: `foundation::WorkStealingPool::parallel_for` runs one
  bootstrap-draw + `DecisionTree::fit` per estimator across threads. Each
  thread only ever writes `trees_[t]` and `in_bag_[t]` for its own index
  `t` into vectors pre-sized before the parallel region starts — no
  shared mutable state, no locks needed. Confirmed TSan-clean (see
  Results).
- OOB error / permutation importance share one helper,
  `oob_accuracy_with_permutation(permute_feature, permutation)`: for each
  training sample, majority-vote only among trees where that sample was
  OOB (using a given feature column optionally shuffled per
  `permutation`), compare to the true label. `oob_error()` calls it
  unpermuted; `feature_importances()` calls it once per feature with that
  column shuffled and reports the OOB-accuracy drop versus the unpermuted
  baseline — Breiman's original permutation-importance definition,
  distinct from `decision_tree`'s impurity-based importance (measured on
  held-out predictions, so a feature that only helps fit training noise
  doesn't get credit).

## Results (captured 2026-07-27, Apple clang 14 / `-std=c++2b`, this Mac)

```
  held-out accuracy: single tree=0.775, forest (50 trees)=0.848
PASS  bagging + feature subsampling generalizes better than a single unconstrained tree on noisy data
  oob_error=0.136, true held-out error=0.128
PASS  OOB error (computed from the training set alone) is close to true held-out error on a fresh sample
  permutation importance: informative=0.4775 noise=0.0000
PASS  the informative feature shows a real OOB-accuracy drop when permuted
PASS  the pure-noise feature's permutation importance is much smaller
PASS  two forests fit with the same random_state produce identical predictions despite parallel fitting
PASS
```

TSan (`--preset tsan`) run of the same binary: identical output, zero
race reports — confirms the parallel-fit path (each worker thread writing
only its own `trees_[t]`/`in_bag_[t]` slot) is race-free.

## Findings

- On a noisy (15% label-flip) XOR-shaped dataset, a single unconstrained
  tree hits 77.5% held-out accuracy (it overfits the noise directly) vs.
  84.8% for a 50-tree forest — a real, measured demonstration of bagging's
  actual purpose: decorrelating trees' errors via bootstrap sampling +
  per-split feature subsampling so majority voting cancels out variance a
  single tree can't avoid.
- OOB error (0.136, computed entirely from the training set) came within
  0.008 of true held-out error measured on a genuinely fresh sample
  (0.128) — this is the real value of OOB estimation: a free, unbiased
  generalization estimate with no validation-split cost.
- Permutation importance cleanly separates the informative feature
  (0.4775 OOB-accuracy drop when shuffled) from pure noise (0.0000 drop)
  — consistent with `decision_tree`'s impurity-based importance result on
  an analogous dataset, via a completely different measurement path
  (held-out OOB accuracy rather than training-set impurity decrease).
- Fixing `random_state` reproduces bit-identical predictions across two
  independent `fit()` calls despite fitting trees across multiple threads
  — per-tree RNG streams are seeded deterministically from
  `(random_state, tree_index)`, so thread scheduling order never affects
  the result.

## Hardware notes
None for the algorithm itself, correctness, parameter-effect, or TSan
results above. Real dataset comparison against sklearn/LightGBM on OpenML
CC-18 needs the deferred Python + `scikit-learn` + `lightgbm` +
dataset-fetch install (tracked in project memory alongside the
JAX/Java/OpenML deferrals).
