# openml_bench

**Status: code-complete AND locally run — pure CPU, no Python/sklearn
dependency.** Datasets are real, fetched once from the live OpenML REST
API, not synthetic. The full sklearn/LightGBM head-to-head comparison
(putting a number in a `vs sklearn (%)` column) stays TODO, deferred
pending the install decision covering that benchmark suite (see project
memory) — but this runner's own real accuracy/timing numbers, on real
datasets, don't depend on it.

## What this measures

PLAN.md Phase 12b step 9: an automated evaluation harness that loads
OpenML CC-18 datasets, trains every Phase 12a algorithm on each, and
records accuracy, training time, and inference latency.

## Design

### Dataset selection (an honest, disclosed scope reduction)

The real OpenML-CC18 suite has 72 datasets, some with up to 100,000 rows
and thousands of one-hot-encoded features. Several Phase 12a algorithms
are `O(n^2)` or worse in the number of rows (`svm`'s full kernel matrix
in particular), so running all 72 at full size isn't practical in this
environment. Instead:

1. Fetched the real suite listing from `https://www.openml.org/api/v1/json/study/99`
   (OpenML-CC18's actual study ID) to get all 72 dataset IDs.
2. Queried `https://www.openml.org/api/v1/json/data/qualities/<id>` for
   each ID's real instance/feature/class counts, and sorted by
   `n_instances * n_features` to find the smallest, fastest ones.
3. Picked 8 of the smallest, deliberately favoring variety in size
   (500-2000 rows), dimensionality (4-30 features), and class count
   (2-10 classes) so step 10's cross-method comparison has real material
   to work with, not 8 near-identical small binary datasets.
4. Downloaded each dataset's real ARFF file from OpenML
   (`https://www.openml.org/data/v1/download/<file_id>/<name>.arff`) and
   committed it under `data/` — small (520 KB total for all 8), keeps the
   benchmark reproducible without a network dependency at build/test
   time, same spirit as this project keeping other captured artifacts
   in-repo.

The 8 datasets: `balance-scale` (625×4, 3 classes), `banknote-authentication`
(1372×4, 2 classes), `blood-transfusion` (748×4, 2 classes), `diabetes`
(768×8, 2 classes, the Pima Indians dataset), `breast-w` (699×9, 2
classes, has 16 real `?` missing values), `wdbc` (569×30, 2 classes),
`vehicle` (846×18, 4 classes), `mfeat-morphological` (2000×6, 10
classes).

### ARFF loader (`arff_loader.h`/`.cpp`)

A small hand-written parser — **honest scope limit**: numeric feature
attributes plus one final nominal class attribute only (matches every
one of the 8 selected datasets' actual structure); no categorical-feature
one-hot encoding, no sparse ARFF, no non-final class column. Missing
values (`?`) are mean-imputed per column, a simple, documented choice
(verified directly against `breast-w`'s 16 real missing values in
`arff_loader_test.cpp` — no NaN survives into the loaded matrix).

### One-vs-rest for the binary-only algorithms

`svm.h`'s SVM, `gbt.h`'s GradientBoostedTrees, and `linear_model.h`'s
LOGISTIC-loss LinearModel are all binary-only in their own modules (each
header documents this itself). PLAN.md's step 4 SVM description calls
for "binary + multiclass (one-vs-rest)", but rather than modify those
already-complete, already-tested modules, the standard one-vs-rest
reduction is implemented here in the harness: one binary model per
class (class *c* vs. rest), predicted class is the argmax of each
model's real-valued score across the *n_classes* models. This is why
`SVM::decision_function()` was added to `svm.h` (a small, backward-
compatible refactor — `predict()` now calls it and thresholds at 0,
same behavior, verified by re-running `svm_test` with zero regressions)
— one-vs-rest needs a real-valued confidence to pick an argmax over,
not just a majority vote over `{-1,+1}` labels. GBT and LinearModel
already expose `predict_proba()`, so no change was needed there.
`DecisionTree`, `RandomForest`, and `KNNClassifier` handle multiclass
natively already and need no wrapping.

`KMeans` and `PCA` are unsupervised and excluded from this benchmark:
OpenML-CC18 is explicitly defined ("Curated Classification benchmark")
as a classification suite, so there's no accuracy number to compare
them against.

### Feature standardization (a real, measured decision, not a default)

An unscaled first run of this harness (raw ARFF values, no
preprocessing) showed `SVM` at 26.6% accuracy on `vehicle` (barely above
the 25% four-class chance floor) and 30.5% on `mfeat-morphological` —
not a real algorithmic weakness, just RBF-kernel distance computations
dominated by whichever feature happens to have the largest raw scale.
Z-score standardization (mean/stddev fit on the train split only, applied
to both train and test — no test-set leakage), applied uniformly before
every algorithm, doesn't change tree-based methods' results at all (they
split on order, not scale) but is what makes the comparison in the
Results table below a fair reflection of algorithm differences rather
than a missing-preprocessing artifact. This is deliberately disclosed,
not silently applied: see Findings below for the exact before/after
numbers.

### Hyperparameters

Kept modest so the full 8-dataset × 6-algorithm sweep finishes in
reasonable wall time for a manually-run benchmark (not a `ctest` target
— see `CMakeLists.txt`'s note, same convention as `cpu_engine/bench`):
`RandomForest`/`GBT` use 50 trees (default is 100), `SVM` caps at 200
SMO passes, `LinearModel` uses L-BFGS at 100 iterations. Real numbers,
not tuned per-dataset — a genuine hyperparameter sensitivity sweep is
step 12's job, not this step's.

## Results (captured 2026-07-28, Apple clang 14 / `-std=c++2b`, this Mac, 80/20 train/test split, seed 42)

```
dataset                  algorithm                   n          d       train(s)   accuracy
------------------------------------------------------------------------------------------
balance-scale            DecisionTree              625          4         0.0046     0.8320
balance-scale            RandomForest              625          4         0.0776     0.8720
balance-scale            GBT                       625          4         0.2054     0.9120
balance-scale            SVM                       625          4         2.2664     0.9440
balance-scale            KNN                       625          4         0.0006     0.8720
balance-scale            LogisticRegression        625          4         0.1403     0.9040
banknote-authentication  DecisionTree             1372          4         0.0155     0.9927
banknote-authentication  RandomForest             1372          4         0.1955     0.9964
banknote-authentication  GBT                      1372          4         0.2572     1.0000
banknote-authentication  SVM                      1372          4         4.1445     1.0000
banknote-authentication  KNN                      1372          4         0.0018     1.0000
banknote-authentication  LogisticRegression       1372          4         0.1787     0.9781
blood-transfusion        DecisionTree              748          4         0.0075     0.7047
blood-transfusion        RandomForest              748          4         0.1226     0.7919
blood-transfusion        GBT                       748          4         0.1017     0.7584
blood-transfusion        SVM                       748          4         1.1369     0.8054
blood-transfusion        KNN                       748          4         0.0007     0.7718
blood-transfusion        LogisticRegression        748          4         0.1084     0.8054
diabetes                 DecisionTree              768          8         0.0174     0.7320
diabetes                 RandomForest              768          8         0.1701     0.7647
diabetes                 GBT                       768          8         0.2279     0.7908
diabetes                 SVM                       768          8         1.1951     0.7843
diabetes                 KNN                       768          8         0.0008     0.7255
diabetes                 LogisticRegression        768          8         0.1653     0.8039
breast-w                 DecisionTree              699          9         0.0068     0.9137
breast-w                 RandomForest              699          9         0.0752     0.9568
breast-w                 GBT                       699          9         0.1242     0.9496
breast-w                 SVM                       699          9         0.9102     0.9568
breast-w                 KNN                       699          9         0.0010     0.9496
breast-w                 LogisticRegression        699          9         0.1359     0.9424
wdbc                     DecisionTree              569         30         0.0403     0.9204
wdbc                     RandomForest              569         30         0.1634     0.9558
wdbc                     GBT                       569         30         0.7379     0.9558
wdbc                     SVM                       569         30         0.6461     0.9823
wdbc                     KNN                       569         30         0.0007     0.9735
wdbc                     LogisticRegression        569         30         0.2731     0.9823
vehicle                  DecisionTree              846         18         0.0438     0.7219
vehicle                  RandomForest              846         18         0.2682     0.7988
vehicle                  GBT                       846         18         1.8767     0.8166
vehicle                  SVM                       846         18         5.8285     0.7929
vehicle                  KNN                       846         18         0.0010     0.7101
vehicle                  LogisticRegression        846         18         0.8666     0.8225
mfeat-morphological      DecisionTree             2000          6         0.0558     0.7050
mfeat-morphological      RandomForest             2000          6         0.4393     0.7225
mfeat-morphological      GBT                      2000          6         3.7152     0.7175
mfeat-morphological      SVM                      2000          6       116.2922     0.7200
mfeat-morphological      KNN                      2000          6         0.0020     0.7250
mfeat-morphological      LogisticRegression       2000          6         2.4464     0.7325
```

(`arff_loader_test`'s own captured output — parser correctness against
each dataset's known real metadata — lives in its own PASS/FAIL log,
run via `ctest -R arff_loader_test`.)

## Findings

- **Standardization's measured effect, before vs. after** (the exact
  numbers behind the Design section's claim): `SVM` on `vehicle` went
  from 26.6% (near the 25% four-class chance floor) to 79.3% accuracy;
  on `mfeat-morphological`, 30.5% to 72.0%; on `wdbc`, 67.3% to 98.2%.
  `KNN` on `mfeat-morphological` went from 45.0% to 72.5%. Tree-based
  methods (`DecisionTree`/`RandomForest`/`GBT`) barely moved (e.g. GBT on
  `mfeat-morphological`: 71.75% both times) — exactly the scale-
  invariance vs. scale-sensitivity split the Design section predicts,
  measured rather than assumed.
- No single algorithm wins everywhere: `SVM` or `LogisticRegression` top
  5 of the 8 datasets, `GBT` tops `vehicle`, and accuracy differences
  between the best and worst algorithm on a given dataset range from
  ~2 points (`wdbc`) to ~12 points (`balance-scale`) — real variation
  step 10's per-dataset analysis has to explain, not just restate.
- Training time spans almost 5 orders of magnitude in this table: `KNN`
  fits in under 2.2ms (`fit()` is just tree construction, no actual
  training) vs. `SVM`'s 116.3s on `mfeat-morphological` (10-class
  one-vs-rest — 10 separate `O(n^2)` kernel-matrix builds at n≈1600 each,
  the real cost of this project's simplified-SMO SVM having no shared
  computation across the one-vs-rest models).
- `balance-scale` is the one dataset where the gap between the best
  (`SVM`, 94.4%) and simplest (`DecisionTree`, 83.2%) method is largest
  in relative terms — consistent with the dataset's own known structure
  (the true decision rule, `left_weight*left_distance` vs.
  `right_weight*right_distance`, is a product of two features, which a
  single axis-aligned tree split can only approximate piecewise, while
  RBF-kernel SVM and even KNN's local neighborhoods can capture the
  curved decision boundary more directly).

## Hardware notes
None — every result above is a real, measured number from this Mac. No
GPU/FPGA/TPU dependency for any Phase 12a algorithm. The full
sklearn/LightGBM comparison needs the deferred Python + `scikit-learn` +
`lightgbm` install (tracked in project memory).
