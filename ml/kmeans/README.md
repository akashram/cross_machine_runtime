# kmeans

**Status: code-complete AND locally run — pure CPU, no external
dependency. The `vs sklearn on OpenML CC-18` comparison stays TODO —
deferred pending the install decision covering that benchmark suite
(see project memory); real correctness/parameter-effect results below
don't depend on it.**

## What this measures

PLAN.md Phase 12a step 6: k-means++ (Arthur & Vassilvitskii 2007)
initialization, Lloyd's iteration, elbow method for k selection.

## Design

- **k-means++ init**: pick the first centroid uniformly at random, then
  repeatedly pick the next centroid from the remaining points with
  probability proportional to `D(x)^2` (squared distance to the nearest
  centroid chosen so far) — implemented as a cumulative-weight roulette
  wheel over `min_dist_sq`, updated incrementally per new centroid rather
  than recomputed from scratch. Structurally can't place two initial
  centroids right next to each other the way independent uniform
  sampling can, which is the entire mechanism behind its known
  local-optima advantage (measured below, not just cited).
- **Lloyd's iteration**: assign step (nearest centroid, contiguous
  squared-distance loop — the auto-vectorizable "SIMD centroid update"
  shape, same convention as `decision_tree`/`svm`/`knn`'s honest
  no-hand-AVX caveat) + update step (mean of assigned points). A cluster
  that loses all its points keeps its previous centroid rather than
  producing NaN — a real, if rare, degenerate case. Converges once the
  largest single centroid movement (squared L2) drops below `tol^2`.
- `KMeansParams::use_kmeanspp_init` (default true): a real, working
  plain-uniform-random-init alternative, kept specifically so
  `kmeans_test.cpp` can measure k-means++'s benefit against a genuine
  baseline instead of asserting it (same spirit as `decision_tree`'s
  `max_features` design note).
- `elbow_curve()`: fits `KMeans` independently for `k = 1..max_k` on the
  same data and returns the inertia at each; callers look for the point
  where the marginal inertia drop flattens.

## Results (captured 2026-07-28, Apple clang 14 / `-std=c++2b`, this Mac)

```
  3-blob clustering purity=1.000, converged in 2 iterations, inertia=93.20
PASS  k-means with k=3 recovers three well-separated blobs almost exactly
  elbow inertia by k: k=1:11806.4 k=2:6320.1 k=3:96.5 k=4:81.5 k=5:71.7 k=6:61.4
  drop(2->3)=6223.57 drop(5->6)=10.22
PASS  inertia drops far more sharply going from k=2 to the true k=3 than from k=5 to k=6 past it (the elbow)
  over 25 seeds: kmeans++ avg inertia=63.63 (worst=244.12), random-init avg inertia=1511.70 (worst=5926.67)
PASS  k-means++ reaches lower (or equal) average inertia than plain random init across seeds
PASS  k-means++'s worst-case seed is still better than random init's worst-case seed (fewer bad local optima)
  held-out predictions: 1 2 0
PASS  three held-out points near three different blob centers get three different predicted labels
PASS
```

## Findings

- On 3 well-separated 2D blobs, k-means (k=3, k-means++ init) recovers
  the true grouping with 100% purity in just 2 Lloyd iterations — the
  clusters are separated enough that even a single reassignment pass
  after init settles it.
- The elbow curve shows the expected shape sharply: inertia falls by
  ~6224 going from k=2 to the true k=3 (still splitting real clusters
  apart), but only ~10 going from k=5 to k=6 (splitting an
  already-correct cluster in half, which barely reduces within-cluster
  scatter) — a >600x difference in marginal benefit, a real, measured
  elbow rather than an assumed one.
- **The core k-means++ claim, measured against a real random-init
  baseline, not cited from the paper**: on 8 well-separated but
  similarly-sized small clusters (a scenario where independent uniform
  sampling is plausibly likely to draw two initial centroids from the
  same cluster and permanently starve another), across 25 random seeds
  k-means++ reaches average final inertia 63.6 vs. plain random init's
  1511.7 — roughly 24x lower — and k-means++'s *worst* seed (244.1)
  beats random init's *average* seed (1511.7), let alone random init's
  worst seed (5926.7, ~24x worse again). This is the local-optima
  problem k-means++ was designed to fix, actually reproduced.
- `predict()` on three held-out points placed near three different
  training-blob centers assigns three different labels, confirming it
  uses the fitted centroids from `fit()` rather than silently refitting
  or ignoring cluster identity.

## Hardware notes
None for the algorithm itself, correctness, or parameter-effect results
above. Real dataset comparison against sklearn on OpenML CC-18 needs the
deferred Python + `scikit-learn` + dataset-fetch install (tracked in
project memory alongside the JAX/Java/LightGBM deferrals).
