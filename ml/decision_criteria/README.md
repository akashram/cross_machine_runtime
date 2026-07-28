# decision_criteria

**Status: written analysis, complete.** No code in this step — PLAN.md
Phase 12b step 11 is a decision-framework document, not a new algorithm.
Grounded in `ml/openml_bench`'s real measured results and
`ml/cross_method`'s per-dataset analysis wherever direct evidence
exists; explicitly marked as inference-from-design (not measured) where
it isn't.

## What this measures

PLAN.md Phase 12b step 11: given a data type/size/constraint, which
model family to reach for first, and why — covering tabular vs.
unstructured data, small vs. large data, interpretability constraints,
and latency constraints.

## Scope

This covers the 8 algorithms in `ml/`'s Phase 12a: `DecisionTree`,
`RandomForest`, `GradientBoostedTrees` (GBT), `SVM`, `KNNClassifier`,
`KMeans`, `PCA`, `LinearModel`. All are classical, tabular-data methods.
Deep learning (images, text, audio, long sequences) is out of scope for
this document and this phase entirely — see `transformer/` and
`distributed_training/` for this repo's deep-learning work, which uses a
completely different toolset.

## Decision axis 1: data type

**Tabular / structured (rows × heterogeneous columns)**: default to a
tree ensemble (`RandomForest` or `GBT`) as the first thing to try. This
matches both this repo's own measured results (an ensemble is best or
tied-best on 5 of 8 `openml_bench` datasets — see `cross_method`) and the
broader, well-established empirical finding that tree ensembles
outperform deep learning on tabular data at the sizes typical of
real-world tabular problems.

**Unstructured (images, text, audio, raw sequences)**: none of this
phase's 8 algorithms are the right first reach — they need engineered or
learned feature representations first. `LinearModel`/`SVM` on top of a
pretrained embedding is a reasonable classical-ML baseline once such a
representation exists, but building that representation is squarely
Phase 6/9's territory (`transformer/`, `distributed_training/`), not
this one's.

## Decision axis 2: data size

- **Small (n ≲ 1000)**: favor simple, regularized, low-variance models.
  **Measured**: on `blood-transfusion` (n=748) and `diabetes` (n=768),
  `LinearModel` (logistic) and `SVM` tie for best, while a single
  unconstrained `DecisionTree` is worst on both (0.705 and 0.732) —
  overfitting idiosyncrasies of a small training split with no ensemble
  averaging to smooth it out. Avoid a lone `DecisionTree` at this size;
  it was the worst algorithm on 6 of the 8 datasets `openml_bench`
  tested, most starkly at small n.
- **Medium (n ≈ 500-2000, this benchmark's range)**: `RandomForest`/`GBT`
  are consistently competitive-to-best and are the safest "just works"
  default — best or tied-best on `breast-w`, `vehicle`, and close second
  almost everywhere else.
- **Large (n ≳ 100,000)**: **not measured** by this benchmark (the
  largest dataset tested was 2000 rows) — inferred from each
  algorithm's own documented complexity instead. `SVM` here is a
  concrete, measured warning: its `O(n^2)` kernel matrix made the
  10-class one-vs-rest fit on just 2000 rows take 116 seconds
  (`openml_bench`'s own captured number) — this scales unusably past
  tens of thousands of rows without a linear-kernel or approximate-kernel
  variant (out of scope here; see `svm`'s README for the
  simplified-SMO scope note). `RandomForest` parallelizes trivially over
  independent trees (this repo's version uses `foundation`'s
  work-stealing pool), so it scales better with more hardware thrown at
  it; `GBT`'s exact per-feature sort-based split search (not the
  histogram-binned version PLAN.md names as an option — see `gbt`'s own
  README scope note) means its per-round cost grows with `n log n`,
  worth watching at large n even though it wasn't directly measured
  here.

## Decision axis 3: dimensionality

- **Low-d (< 10 features)**: most algorithms land close together
  (**measured**: `balance-scale`, `banknote-authentication`,
  `blood-transfusion`, `diabetes` all have ≤ 8 features and every
  algorithm's accuracy falls within a fairly narrow band per dataset).
- **High-d + correlated features**: margin/linear methods pull ahead of
  trees. **Measured**: on `wdbc` (30 correlated cell-nucleus
  measurements), `SVM`/`LinearModel` tie for best (0.982) while trees
  trail (`DecisionTree` 0.920) — trees split one feature at a time and,
  without enough rows to compensate, can't exploit redundant/correlated
  information as efficiently as a kernel or an L2 penalty can.
- **Very high-d (thousands of features, e.g. one-hot-encoded
  categoricals or genomic data)**: **not measured** directly, but this
  repo has two concrete, tested tools for it: `PCA` (randomized SVD) as
  a dimensionality-reduction preprocessing step, or `LinearModel` with
  `l1_ratio` near 1 (lasso) for automatic feature selection — measured
  in `linear_models`' own tests to drive noise-feature coefficients
  near zero while preserving informative ones.

## Decision axis 4: categorical features

None of these 8 algorithms accept raw categorical (string) features
directly — even `ml/openml_bench`'s own ARFF loader documents this as an
explicit scope limit (numeric feature columns plus one final nominal
*class* column only). Practical guidance: one-hot encode low-cardinality
categoricals before `SVM`/`KNNClassifier`/`LinearModel`/`PCA` (distance-
and margin-based methods need numeric, meaningfully-scaled inputs
regardless). Tree-based methods (`DecisionTree`/`RandomForest`/`GBT`)
could in principle split natively on categorical values, but this
implementation requires numeric input too — a real, disclosed
limitation, not silently worked around. A native categorical-split CART
variant is a real, scoped-out follow-up.

## Decision axis 5: noise level

- **Clean, well-separated data**: **measured** on
  `banknote-authentication` — every algorithm lands at or near 100%
  accuracy. When data is this clean, algorithm choice stops mattering;
  effort is better spent elsewhere (feature engineering, more data)
  than on model selection.
- **Noisy, weak-signal data**: **measured** on `blood-transfusion` — a
  single `DecisionTree` overfits worst (0.705), and notably `GBT`
  (0.758) underperforms `RandomForest` (0.792) here despite being the
  more sophisticated method: boosting fits each round's residual,
  including noisy residuals, a real and measured sensitivity (previewed
  further in `failure_modes`, step 14). Prefer regularized linear models
  or bagged ensembles over boosting when label noise is suspected and
  unconfirmed.

## Decision axis 6: interpretability constraints

Only two of these 8 algorithms offer real per-prediction
interpretability: a single `DecisionTree` (the root-to-leaf path *is* a
human-readable rule) and `LinearModel` (coefficients are directly
interpretable, and `linear_models`' sparsity results show lasso can
additionally surface *which* features matter). `RandomForest` and `GBT`
offer only *global* interpretability tools (`RandomForest`'s
permutation importance, built and tested in this repo) — useful for
understanding a model in aggregate, not for explaining one specific
prediction to a person. `SVM` (especially with an RBF kernel) and
`KNNClassifier` are effectively black-box. **If a regulatory or
compliance context requires explaining individual decisions** (e.g. a
credit or medical decision, the real-world context `blood-transfusion`
and `diabetes` come from), a single `DecisionTree` or `LinearModel` are
the only defensible choices among these 8, even when an ensemble would
score higher.

## Decision axis 7: latency constraints

- **Training-time budget**: `KNNClassifier`'s `fit()` is near-instant
  (**measured**: 0.6-2.2 ms across every dataset in `openml_bench` — it
  only builds a tree structure, no optimization loop) — the right choice
  when retraining frequently on fresh data. `SVM` is the opposite
  extreme (**measured**: up to 116 s for one 10-class multiclass fit) —
  a poor fit for frequent retraining.
- **Inference-latency budget**: `DecisionTree`/`LinearModel` give
  `O(depth)`/`O(d)` fixed-cost predictions with no data-dependent lookup
  — the safest choice under a hard real-time budget. `RandomForest`/`GBT`
  inference cost scales with `n_estimators`, a direct, tunable
  accuracy/latency dial (quantified further in `hyperparam_sensitivity`,
  step 12). `KNNClassifier`'s inference cost depends on the tree
  structure and dataset size even with the `knn` module's
  branch-and-bound pruning (see that module's own measured node-visited
  counts) — not a fixed cost the way the others are.

## Unsupervised needs (not classification)

- **Need clusters, no labels**: `KMeans` (k-means++ init, measured to
  reach ~24x lower inertia than plain random init on adversarial
  cluster configurations — see `kmeans`'s README) with the elbow method
  for choosing k.
- **Need dimensionality reduction, whitening, or visualization**: `PCA`
  (randomized SVD) — measured to recover a true low-rank embedded
  structure with 99.9% variance captured and to produce real unit-variance
  whitened components.

## Quick-reference table

| Situation | Reach for first | Why (measured or design-inferred) |
|---|---|---|
| Tabular, no other constraint | `RandomForest` or `GBT` | Best/tied-best on 5/8 benchmark datasets |
| Small (n < 1000), noisy | `LinearModel` (logistic) | Best/tied-best on both small-noisy datasets tested; avoids single-tree overfitting |
| High-d, correlated features | `SVM` or `LinearModel` | Measured best on the one high-d dataset tested (wdbc, 30 features) |
| Need per-prediction explainability | `DecisionTree` or `LinearModel` | Only two of the 8 with real per-prediction interpretability |
| Hard real-time inference latency | `DecisionTree` or `LinearModel` | Fixed `O(depth)`/`O(d)` cost, no data-dependent lookup |
| Frequent retraining | `KNNClassifier` | `fit()` measured at ~1-2 ms |
| Large n (not directly tested) | `RandomForest` | Parallelizes trivially over independent trees; avoid `SVM` (measured `O(n^2)` cost) |
| Suspected label noise, unconfirmed | `RandomForest` over `GBT` | Bagging measured more robust to noise than boosting on this benchmark's noisiest dataset |
| Need clustering (no labels) | `KMeans` | k-means++ init measured ~24x better than random init |
| Need dimensionality reduction | `PCA` | Measured to recover true low-rank structure and correct whitened variance |

## Hardware notes
None — this is a written decision framework built on `ml/openml_bench`'s
real, already-captured measurements plus each algorithm's own tested
properties. See those steps' READMEs for the underlying numbers.
