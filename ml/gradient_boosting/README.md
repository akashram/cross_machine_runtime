# gradient boosting

**Status: code-complete AND locally run — pure CPU, no external
dependency. The `vs sklearn/LightGBM on OpenML CC-18` comparison stays
TODO — deferred pending the install decision covering that benchmark
suite (see project memory); real correctness/parameter-effect results
below don't depend on it.**

## What this measures

PLAN.md Phase 12a step 3: Friedman (2001) gradient boosting with Newton
(second-order) split-gain/leaf-value steps. Binary classification via
logistic loss, row subsampling (stochastic GB, Friedman 2002), column
subsampling, L1/L2-regularized leaf weights.

## Design

- `regression_tree.{h,cpp}`: a dedicated base-learner tree, separate from
  `decision_tree`'s classification `DecisionTree` — boosting needs a
  regression tree fit to continuous (gradient, Hessian) pairs, not class
  labels, so this isn't a mode of `DecisionTree` but its own small class
  (same flat-node/index-based design as `decision_tree` for consistency).
  Both the split-gain criterion and the leaf value use Friedman's
  second-order (Newton) approximation, the same formulation Chen &
  Guestrin (2016) generalized as XGBoost:
  - leaf weight `w* = -softthresh(G, alpha) / (H + lambda)`
  - split gain `= 0.5*[GL_term + GR_term - G_term] - gamma`, where
    `*_term = softthresh(sum_g, alpha)^2 / (sum_h + lambda)`
  - `softthresh` is L1 (`alpha`) soft-thresholding on a gradient sum —
    the closed-form Newton leaf weight under an L1 penalty (a dead zone
    of width `2*alpha` around zero, linear shrinkage outside it).
- `gbt.cpp`'s `fit()`: for each of `n_estimators` rounds, computes
  `g_i = p_i - y_i`, `h_i = p_i*(1-p_i)` (binary logistic loss gradient/
  Hessian w.r.t. the running raw score `F`), draws a fresh row subsample
  (`subsample`, without replacement, redrawn every round — Friedman
  2002's stochastic GB) and a fixed column subsample for the whole tree
  (`colsample`, XGBoost-style `colsample_bytree` — **deliberately
  different** from `random_forest`'s per-split-node resampling), fits a
  `GBRegressionTree` to `(g, h)` on that subset, then updates
  `F += learning_rate * tree(x)` for every sample (not just the
  subsampled ones — standard).
- `base_score_` initializes `F` to the prior class log-odds
  (`log(p/(1-p))` of the positive-class rate), not zero — so the first
  tree only has to correct the deviation from the class prior, not learn
  it from scratch.
- **Honest scope caveat**: histogram-based binned split finding
  (LightGBM-style, `max_bins` in PLAN.md's framing) is NOT implemented —
  this uses exact per-feature sort-based split search, the same
  `O(n log n)`-per-feature shape (and the same honest gap) as
  `decision_tree`'s "vectorized split search" caveat. A real, scoped-out
  follow-up.

## Results (captured 2026-07-27, Apple clang 14 / `-std=c++2b`, this Mac)

```
  train accuracy on separable 1D data=1.000
PASS  a perfectly separable 1D dataset is fit exactly
  training logloss: after 3 rounds=0.6319, after 60 rounds=0.0024
PASS  training loss drops substantially as more boosting rounds are added
  train accuracy: depth-1 stumps=0.627, depth-3 trees=0.998
PASS  an ensemble of depth-3 trees solves XOR almost exactly
PASS  deeper base learners solve XOR better than depth-1 stumps, same as a single tree would
  predicted-probability variance: l2_reg=0.1 -> 0.20325, l2_reg=50.0 -> 0.05333
PASS  stronger L2 regularization shrinks Newton leaf weights, reducing prediction spread
PASS
```

## Findings

- Training log-loss on XOR drops from 0.632 (near `ln(2)`, i.e. close to
  the loss of a naive constant predictor) after 3 rounds to 0.0024 after
  60 rounds — a real, measured demonstration of boosting's core
  mechanism: each round's tree fits the *current* residual gradient, so
  loss keeps dropping as long as there's gradient signal left to fit.
- An ensemble of depth-1 stumps stalls at 62.7% train accuracy on XOR
  (each stump can only split on one feature, so no single tree — however
  many rounds — can represent the XOR boundary through additive boosting
  of one-feature splits alone), while depth-3 trees reach 99.8% — the
  same tree-depth argument `decision_tree_test.cpp` makes for a single
  tree, now shown to hold for the boosted ensemble too.
- Raising `l2_reg` from 0.1 to 50.0 shrinks the variance of predicted
  probabilities across samples from 0.203 to 0.053 — a direct, measured
  consequence of the Newton leaf-weight formula `w* = -G/(H + lambda)`:
  larger `lambda` pulls every leaf weight toward zero regardless of its
  gradient signal, which is exactly the regularization mechanism L2 is
  supposed to provide.

## Hardware notes
None for the algorithm itself, correctness, or parameter-effect results
above. Real dataset comparison against sklearn/LightGBM on OpenML CC-18
needs the deferred Python + `scikit-learn` + `lightgbm` + dataset-fetch
install (tracked in project memory alongside the JAX/Java/OpenML
deferrals).
