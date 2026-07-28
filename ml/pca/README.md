# pca

**Status: code-complete AND locally run — pure CPU, no external
dependency. The `vs sklearn on OpenML CC-18` comparison stays TODO —
deferred pending the install decision covering that benchmark suite
(see project memory); real correctness/parameter-effect results below
don't depend on it.**

## What this measures

PLAN.md Phase 12a step 7: PCA via randomized SVD (Halko, Martinsson &
Tropp 2011), explained variance ratio, whitening.

## Design

- **Randomized range finder**: `Y = A * Omega` for a random Gaussian
  `d x l` matrix `Omega` (`l = n_components + n_oversamples`), refined by
  `n_power_iterations` steps of `Y = A(A^T Y)` (re-orthonormalized
  between steps for numerical stability) — this is what lets the method
  find the dominant singular subspace of `A` without ever forming or
  decomposing the full `n x d` matrix directly.
- **QR via modified Gram-Schmidt** (`orthonormalize_columns`): only `Q`
  is needed (the projection basis), not `R`, so MGS is enough. **Honest
  caveat**, same convention as `decision_tree`/`svm`/`knn`: this is not
  Householder QR — fine at the well-conditioned, modest-dimension scale
  exercised here, not the most numerically robust QR available. A
  Householder version is a real, scoped-out follow-up.
- **Small-matrix SVD via Jacobi eigendecomposition**: `B = Q^T A` (an
  `l x d` matrix, `l` small) is decomposed by eigendecomposing the
  symmetric `l x l` matrix `B B^T = U_hat S^2 U_hat^T` with the classic
  cyclic Jacobi algorithm (Golub & Van Loan) — appropriate specifically
  because `l` is small (`n_components + n_oversamples`), not the full
  data dimension. Right singular vectors (principal components) are then
  recovered as `V[:,c] = (1/s_c) B^T U_hat[:,c]`, re-normalized to guard
  floating-point drift.
- `explained_variance_ratio_` uses **total variance summed across every
  original feature** (computed directly from the centered data, not
  derived from the truncated SVD) as the denominator — the standard
  sklearn-style definition. Using only the retained components' own
  variance as the denominator would make the ratios trivially sum to 1
  regardless of how much of the data's actual variance they cover.
- `whiten`: divides each transformed component by
  `sqrt(explained_variance)`, so components end up with unit variance
  (verified directly, see Results).

## Results (captured 2026-07-28, Apple clang 14 / `-std=c++2b`, this Mac)

```
  top-2 explained_variance_ratio sum=0.9990, reconstruction relative error=0.0016
PASS  top 2 components capture >98% of variance on data that is genuinely 2D plus small noise
PASS  reconstructing from just the top 2 components recovers the original 5D data almost exactly
  components: all_unit_norm=1 all_orthogonal=1
PASS  every principal component has unit norm
PASS  every pair of principal components is orthogonal
  full-rank explained_variance_ratio sum=1.0000, descending=1
PASS  at full rank, explained variance ratios sum to ~1 (all variance accounted for)
PASS  explained variance ratios are sorted descending by component
  first component=[-1.000, 0.005], explained_variance_ratio[0]=0.9882
PASS  the first principal component aligns with the axis carrying 10x more variance
PASS  the dominant axis accounts for >95% of variance
  whitened component 0 variance=1.0000
PASS  whitened component variance is close to 1
  whitened component 1 variance=1.0000
PASS  whitened component variance is close to 1
PASS
```

## Findings

- On 200 points in 5D where only 2 dimensions carry real signal (std 5
  and 3) and the remaining 3 are near-zero noise (std 0.1), the top 2
  randomized-SVD components capture 99.90% of variance, and
  reconstructing from just those 2 components recovers the original 5D
  data with only 0.16% relative squared error — a real, measured
  demonstration that the randomized method finds the true low-rank
  structure rather than an approximation that merely looks plausible.
- Components are unit-norm and pairwise orthogonal to floating-point
  tolerance on every test dataset — the fundamental SVD invariant the
  Jacobi-eigendecomposition-based recovery step has to preserve,
  checked directly rather than assumed from the math.
- At full rank (`n_components == n_features`), explained variance ratios
  sum to 1.0000 — confirms `total_variance`'s "sum across every
  original feature" definition is self-consistent with the SVD's own
  variance decomposition, not just numerically close by coincidence.
- On 2D data with 10x more variance along one axis, the first component
  aligns with that axis to `|cos| = 1.000` (up to the sign flip SVD
  leaves undetermined) and captures 98.8% of variance — matches the
  10^2/(10^2+1^2) ≈ 99% variance ratio the setup implies.
- Whitening measured directly (not just implemented and trusted): both
  components' transformed variance lands at 1.0000 despite a 10x
  variance difference in the original data — the entire point of
  whitening, verified rather than assumed.

## Hardware notes
None for the algorithm itself, correctness, or parameter-effect results
above. Real dataset comparison against sklearn on OpenML CC-18 needs the
deferred Python + `scikit-learn` + dataset-fetch install (tracked in
project memory alongside the JAX/Java/LightGBM deferrals).
