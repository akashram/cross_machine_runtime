# decision tree

**Status: code-complete AND locally run — pure CPU, no external
dependency. The `vs sklearn/LightGBM on OpenML CC-18` comparison stays
TODO — deferred pending the install decision covering that benchmark
suite (see project memory); real correctness/parameter-effect results
below don't depend on it.**

## What this measures

PLAN.md Phase 12a step 1: CART (Gini/entropy splitting), pre-pruning
(`max_depth`, `min_samples_leaf`, `min_samples_split`,
`min_impurity_decrease`), vectorized split search.

## Design

- Classification only (labels are small non-negative integers stored as
  `float`) — PLAN.md frames this step against Gini/entropy impurity,
  which is inherently a classification concept; an MSE-reduction
  regression variant would be a real but separate extension, not
  attempted here.
- Flat, index-based node storage (`std::vector<TreeNode>`, `left`/`right`
  as indices not pointers) rather than a pointer tree — cache-friendly,
  and `random_forest` (step 2) holds many of these, so avoiding per-node
  heap allocation matters more here than in a single-tree-only design.
- `find_best_split`: per feature, sort `(value, class)` pairs once
  (`O(n log n)`), then a single left-to-right sweep accumulates
  class-count histograms and evaluates impurity decrease at every valid
  split point in one linear pass over contiguous arrays — the
  "vectorized split search" shape PLAN.md asks for. **Honest caveat**: no
  hand-written AVX intrinsics — this relies on `-O3` auto-vectorizing the
  linear accumulation loop, not on hand-tuned SIMD; a genuinely
  hand-vectorized version is a real, scoped-out follow-up, not
  implicitly claimed here.
- `feature_importance`: unnormalized impurity-decrease sum per feature,
  weighted by node sample count (the standard Breiman-et-al. definition),
  normalized to sum to 1 across features.

## Results (captured 2026-07-27, Apple clang 14 / `-std=c++2b`, this Mac)

```
PASS  a perfectly separable 1D dataset is fit exactly
  stump (depth<=1) train accuracy=0.553, deep (depth<=6) train accuracy=1.000
PASS  a depth-1 stump cannot solve XOR (accuracy stays well below the deep tree's)
PASS  a depth-6 tree solves XOR almost exactly
  feature_importance: informative=1.0000 noise=0.0000
PASS  the informative feature gets nearly all importance
PASS  the pure-noise feature gets nearly none
  n_leaves: min_samples_leaf=1 -> 36, min_samples_leaf=20 -> 15
PASS  a stricter min_samples_leaf produces a smaller (more pruned) tree once there's label noise for the unconstrained tree to overfit
```

## Findings

- A depth-1 stump gets 55.3% accuracy on XOR (barely above the 50%
  chance floor for a balanced 2-class problem) while a depth-6 tree gets
  100% — a real, measured demonstration of *why* tree depth matters:
  XOR has no single-feature split that separates the classes at all, so
  a stump is structurally incapable of doing better than guessing on the
  majority class.
- Feature importance correctly attributes ~100% of impurity decrease to
  the genuinely informative feature and ~0% to pure noise — the split
  search never even considers noise splits worth taking (a Gini/entropy
  split on random noise has no consistent impurity decrease across the
  greedy search, so it's never selected).
- `min_samples_leaf`'s pruning effect only shows up once there's label
  noise to overfit to (36 leaves unconstrained → 15 leaves at
  `min_samples_leaf=20`) — on a cleanly-separated dataset with large
  natural clusters, a modest `min_samples_leaf` threshold never actually
  binds, which is itself a useful, real thing to know about this
  parameter's behavior (caught during test development: an earlier
  version of this test used clean XOR data and found the constraint
  never bound at all).

## Hardware notes
None for the algorithm itself, correctness, or parameter-effect results
above. Real dataset comparison against sklearn/LightGBM on OpenML CC-18
needs the deferred Python + `scikit-learn` + `lightgbm` + dataset-fetch
install (tracked in project memory alongside the JAX/Java deferrals).
