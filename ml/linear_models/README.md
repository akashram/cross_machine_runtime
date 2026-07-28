# linear models

**Status: code-complete AND locally run — pure CPU, no external
dependency. The `vs sklearn on OpenML CC-18` comparison stays TODO —
deferred pending the install decision covering that benchmark suite
(see project memory); real correctness/parameter-effect results below
don't depend on it.**

## What this measures

PLAN.md Phase 12a step 8: SGD with L1/L2 (elastic net) regularization,
L-BFGS, logistic regression as baseline classifier.

## Design

- One `LinearModel` class covers both squared-loss regression and
  logistic-loss classification (`LinearModelParams::loss`), and both an
  SGD and an L-BFGS optimizer (`LinearModelParams::optimizer`).
- **Elastic-net SGD**: proximal (ISTA-style) per-sample updates -- an
  ordinary gradient step on the smooth part (data loss + L2 weight
  decay), followed by soft-thresholding for the L1 term. This is the
  standard way to run SGD against a penalty that's non-differentiable at
  zero (what sklearn's `SGDClassifier`/`SGDRegressor` do under
  `penalty='elasticnet'`).
- **L-BFGS** (Nocedal & Wright, Algorithm 7.4/7.5): two-loop recursion
  over the last `lbfgs_memory` `(s, y)` correction pairs approximates the
  inverse-Hessian-vector product, plus backtracking Armijo line search.
  **Honest scope limit**: L-BFGS always regularizes with
  `alpha*(1-l1_ratio)` as an L2-only penalty regardless of `l1_ratio` --
  a non-differentiable L1 term would break the smoothness the
  quasi-Newton curvature estimate assumes. A proximal-LBFGS (OWL-QN)
  variant that handles L1 properly is a real, scoped-out follow-up, not
  implicitly claimed here (same spirit as `svm`'s simplified-SMO caveat).
- Logistic regression uses `y in {0,1}` and the numerically stable
  sigmoid (branches on the sign of the input to avoid `exp()` overflow),
  cross-entropy loss.
- Weight decay (L2) and the L1 soft-threshold never touch the bias term
  -- standard practice, since regularizing the intercept has no
  regularizing effect on model complexity, only shifts predictions.

## Results (captured 2026-07-28, Apple clang 14 / `-std=c++2b`, this Mac)

```
  logistic regression (SGD): accuracy=1.000
PASS  SGD-trained logistic regression separates two well-separated blobs almost exactly
  logistic regression (LBFGS): accuracy=1.000
PASS  LBFGS-trained logistic regression separates two well-separated blobs almost exactly
  recovered coef=[2.999, -2.003], intercept=1.007, R^2=0.9998
PASS  recovered coefficient for x0 is close to the true value 3.0
PASS  recovered coefficient for x1 is close to the true value -2.0
PASS  R^2 is close to 1 on data generated from a true linear relationship plus small noise
  lasso: informative coefs=[3.981, -2.970], noise coef sum=0.0196 (5 near zero)
  ridge: informative coefs=[3.908, -2.927], noise coef sum=0.0446
PASS  lasso (pure L1) drives at least 3 of the 5 noise-feature coefficients near zero (< 0.05)
PASS  lasso's total noise-coefficient magnitude is smaller than ridge's (real sparsity, not just smaller weights)
PASS  lasso still recovers the informative x0 coefficient reasonably well
PASS  lasso still recovers the informative x1 coefficient reasonably well
  ridge coefficient norm: alpha=1e-5 -> 2.8360, alpha=5.0 -> 1.1739
PASS  a much larger ridge alpha shrinks the coefficient vector's norm toward zero
PASS
```

## Findings

- Both optimizers reach 100% accuracy on two well-separated blobs --
  expected for logistic regression on linearly separable data, but a
  real cross-check that SGD and L-BFGS (very different update rules)
  converge to equally good decision boundaries here.
- On `y = 3*x0 - 2*x1 + 1 + noise`, L-BFGS with small regularization
  recovers `[2.999, -2.003]` and intercept `1.007` against true values
  `[3, -2, 1]` -- essentially exact, R² = 0.9998.
- **Elastic net's two knobs measured separately, on 7-feature data where
  only 2 features are informative**: pure L1 (lasso) drives all 5
  noise-feature coefficients under 0.05 in magnitude (total noise-coef
  sum 0.0196) while pure L2 (ridge) at the identical `alpha` leaves them
  more than 2x larger in total (0.0446) -- real, measured sparsity from
  the L1 term, not merely "smaller weights everywhere" the way ridge
  shrinks. Both still recover the informative coefficients within ~0.5
  of their true values (`[3.981, -2.970]` vs true `[4, -3]`). **Honest
  note on exactness**: per-sample proximal SGD's soft-threshold competes
  against per-sample gradient noise on every step, so coefficients
  rarely land on the literal bit-pattern `0.0f` the way a batch/full-
  gradient proximal step would guarantee -- the test's bar is "near
  zero" (< 0.05), which is what this SGD-based implementation actually,
  honestly produces.
- Ridge's regularization-strength sweep: coefficient-vector norm shrinks
  from 2.836 (alpha=1e-5, effectively unregularized) to 1.174
  (alpha=5.0) on the same data -- the classic ridge bias-variance
  tradeoff, measured directly rather than assumed.

## Hardware notes
None for the algorithm itself, correctness, or parameter-effect results
above. Real dataset comparison against sklearn on OpenML CC-18 needs the
deferred Python + `scikit-learn` + dataset-fetch install (tracked in
project memory alongside the JAX/Java/LightGBM deferrals).
