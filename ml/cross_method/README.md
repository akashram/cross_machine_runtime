# cross_method

**Status: written analysis, complete.** No code in this step — PLAN.md
Phase 12b step 10 is a written comparison, not a new algorithm. All
numbers below are the real, measured results from `ml/openml_bench`
(step 9), not estimated or assumed.

## What this measures

PLAN.md Phase 12b step 10: for each of the 8 datasets `openml_bench`
ran, which model wins and why, explained in terms of the data's actual
characteristics (size, dimensionality, categorical features, noise
level) — not just restating the accuracy numbers.

## Summary

| Dataset | n | d | classes | Best | Worst | Best−Worst |
|---|---|---|---|---|---|---|
| balance-scale | 625 | 4 | 3 | SVM (0.944) | DecisionTree (0.832) | 0.112 |
| banknote-authentication | 1372 | 4 | 2 | GBT/SVM/KNN (1.000) | LogisticRegression (0.978) | 0.022 |
| blood-transfusion | 748 | 4 | 2 | SVM/LogReg (0.805) | DecisionTree (0.705) | 0.101 |
| diabetes | 768 | 8 | 2 | LogReg/GBT (0.804) | DecisionTree (0.732) | 0.072 |
| breast-w | 699 | 9 | 2 | RandomForest/SVM (0.957) | DecisionTree (0.914) | 0.043 |
| wdbc | 569 | 30 | 2 | SVM/LogReg (0.982) | DecisionTree (0.920) | 0.062 |
| vehicle | 846 | 18 | 4 | LogReg (0.823) | KNN (0.710) | 0.113 |
| mfeat-morphological | 2000 | 6 | 10 | LogReg (0.733) | DecisionTree (0.705) | 0.028 |

A single, unconstrained `DecisionTree` is the worst algorithm on **6 of
8 datasets** — the clearest pattern in this table (see below).

## Per-dataset analysis

### balance-scale — nonlinear interaction favors kernels/ensembles over a single tree
The true label rule (`left_weight*left_distance` vs.
`right_weight*right_distance`) is a *product* of two raw features — a
curved decision boundary with no axis-aligned approximation that a
small number of splits can capture well. `SVM`'s RBF kernel implicitly
maps into a space where such curved boundaries become close to linear
(0.944, the best score on this dataset), and `GBT`'s many shallow trees
piecewise-approximate the same curve far better than one tree can
(0.912). A single `DecisionTree` is worst (0.832) for exactly the
reason its own README predicts: axis-aligned splits fundamentally
struggle with feature *interactions*, not just nonlinearity in one
feature. Interestingly `LogisticRegression` still reaches 0.904 despite
being purely linear — the raw features are correlated enough with the
true multiplicative rule that a linear separator captures most of the
signal even without modeling the interaction directly.

### banknote-authentication — clean, well-separated data: algorithm choice barely matters
Four wavelet-transform features cleanly separate genuine from forged
notes; `GBT`, `SVM`, and `KNN` all reach 100% accuracy, `RandomForest`
0.996, `DecisionTree` 0.993. Only `LogisticRegression` trails at 0.978 —
plausible given L-BFGS was capped at 100 iterations and L2
regularization (`alpha=1e-3`) trades a small amount of boundary
sharpness for margin, on a problem where the true boundary needs to be
sharp to hit the last percentage point. **The lesson**: when data is
large enough, low-noise, and well-separated, nearly every reasonable
classifier saturates near ceiling — the interesting differences in this
table come from datasets *without* this property.

### blood-transfusion — small + noisy data favors simple/regularized models
Recency-Frequency-Monetary-Time blood donation features are known to
have a genuinely noisy relationship to the repeat-donation label (human
behavior isn't perfectly predictable from these 4 summary statistics).
`LogisticRegression` and `SVM` tie for best (0.805) — both apply strong,
simple inductive bias that resists fitting noise. `GBT` (0.758) actually
underperforms `RandomForest` (0.792) here despite being the more
sophisticated method: boosting's known sensitivity to label noise (each
round fits the *residual*, including noisy residuals) shows up as a
real, measured effect on a small, noisy real dataset — not just a
theoretical caveat (this previews step 14's failure-mode catalog). A
single `DecisionTree` is worst (0.705): with only 748 rows and no
ensemble averaging, it overfits idiosyncrasies of the training split.

### diabetes — same shape as blood-transfusion, with GBT now competitive
The Pima Indians diabetes dataset shares blood-transfusion's profile
(modest size, real-world medical noise, borderline-diagnosable cases)
but has twice as many features (8 vs. 4). Here `GBT` ties for best
(0.804) rather than underperforming — with more, more-informative
features to split gradients on, boosting's extra modeling capacity pays
off instead of just fitting noise. `LogisticRegression` again ties for
best, and a single `DecisionTree` is again worst (0.732) for the same
small-noisy-data overfitting reason.

### breast-w — well-separated classes, ensembles edge out a single tree
Cytological measurements separate benign from malignant tumors fairly
cleanly (every algorithm scores above 91%). `RandomForest` and `SVM` tie
for best (0.957); the margin over `DecisionTree` (0.914) is real but the
smallest of the "DecisionTree is worst" datasets (0.043) — this dataset
also has 16 real mean-imputed missing values (`Bare_Nuclei`), and
ensemble-averaging across many trees or a max-margin boundary is more
robust to those imputed values landing slightly off than a single
tree's specific split thresholds are.

### wdbc — high-dimensional, correlated features favor margin/linear methods
30 features here are mostly redundant transformations of a few
underlying cell-nucleus properties (radius, perimeter, area, and their
"worst"/"mean"/"standard error" variants are all highly correlated with
each other). `SVM` and `LogisticRegression` tie for best (0.982) —
margin-based and regularized-linear methods both handle correlated,
high-dimensional features gracefully (a kernel or an L2 penalty doesn't
care that features overlap in what they measure). Tree-based methods
split on one feature at a time and, with only 455 training rows against
30 features, don't get enough splits to exploit the redundant
information as efficiently — `RandomForest`/`GBT` tie at 0.956,
`DecisionTree` again worst at 0.920.

### vehicle — overlapping classes penalize KNN's local voting
The 4-class vehicle silhouette task is a documented hard case: two of
the four classes (Opel and Saab, both cars) have highly overlapping
shape-feature distributions even for a human observer. `LogisticRegression`
wins here (0.823, its best relative showing across all 8 datasets) with
`GBT` close behind (0.817). `KNN` is worst on this dataset specifically
(0.710) — not `DecisionTree` this time — because with 18 features and
only ~170 training rows per class, local neighborhoods become less
reliable exactly the way `knn`'s own README describes for higher
dimensionality; a global decision boundary (linear or boosted) handles
the overlapping-class structure better than KNN's local majority vote in
this feature-count/sample-size regime.

### mfeat-morphological — a feature-set ceiling, not an algorithm gap
All 6 algorithms land within a narrow 0.705-0.733 band despite this
being the largest dataset (2000 rows) and hardest task (10 classes).
This is a real, known property of the underlying data, not a modeling
failure: `mfeat-morphological` is one view from a multi-view digit
dataset, and the morphological feature set alone (6 shape descriptors)
is one of that dataset's weaker individual views for digit
discrimination — most of the classification-relevant information lives
in other views (pixel averages, Fourier descriptors, Zernike moments)
not included in this feature set. **The lesson this dataset teaches
that none of the other 7 do**: when the features themselves cap out at
a ceiling, no algorithm choice recovers the missing information —
feature engineering / feature selection matters more than model choice
once every model converges to roughly the same accuracy.

## Cross-cutting lessons (bridge to step 11's decision criteria)

1. **A single, unconstrained `DecisionTree` is the worst performer on
   6 of 8 datasets** — never the right default when any ensemble
   (`RandomForest`, `GBT`) is available at similar implementation cost;
   its only advantage (not measured directly here) is interpretability.
2. **Small + noisy data favors simple/regularized models**
   (`blood-transfusion`, `diabetes`): `LogisticRegression` ties for best
   on both, and `GBT` only wins once it has enough features to model
   real signal rather than residual noise.
3. **High-dimensional + correlated features favor margin/linear
   methods over trees** (`wdbc`): 30 correlated features is exactly
   where `SVM`/`LogisticRegression` pull ahead of tree ensembles.
4. **KNN degrades specifically when classes genuinely overlap in
   higher dimensions** (`vehicle`) — consistent with, not contradicting,
   `knn`'s own measured curse-of-dimensionality findings.
5. **When every algorithm converges to the same accuracy
   (`mfeat-morphological`, `banknote-authentication`), the ceiling is
   set by the data, not the model** — worth recognizing before spending
   effort on algorithm tuning that can't move the number.

## Hardware notes
None — this is a written analysis of `ml/openml_bench`'s real, already-
captured measurements. See that step's README for the full results
table and its own Hardware notes.
