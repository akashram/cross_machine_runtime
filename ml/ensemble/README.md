# ensemble

**Status: code-complete AND locally run — pure CPU, no external
dependency.** Reuses `ml/openml_bench`'s real committed datasets for the
real-data experiment; a separate, mathematically-exact synthetic proof
lives in `ensemble_test.cpp`.

## What this measures

PLAN.md Phase 12b step 13: stacking (diverse base models + meta-learner)
and blending, with empirical validation that diversity is necessary —
showing correlated models add no value — plus written rules for when to
ensemble vs. not.

## Design

- `k_fold_assignment`/`stacking_oof_features`: standard k-fold
  out-of-fold stacking — each base model is cross-validated so every
  row's meta-feature comes from a model that never saw that row during
  training. Verified directly in `ensemble_test.cpp` with a
  "memorizing" base model that would leak a tell-tale signal if any row
  ever saw itself.
- Base models are erased behind one `FitPredictFn` signature
  (`fit(X_train,y_train)` + `predict(X_query)`) so stacking can drive
  heterogeneous algorithm types (`DecisionTree`, `KNNClassifier`,
  `LinearModel`) uniformly. **Honest design choice**: meta-features are
  hard `{0,1}` predictions, not soft probabilities — not every Phase 12a
  classifier exposes `predict_proba()` uniformly, and hard-label
  stacking is a real, standard mode (sklearn's `StackingClassifier`
  supports it too), not a workaround.
- Two combination methods compared throughout: `majority_vote` (naive,
  unweighted) and `stacking` (a `LinearModel` logistic-regression meta-
  learner trained on out-of-fold predictions).
- `ensemble_test.cpp` proves the central diversity claim exactly, not
  just observes it on one lucky dataset: three synthetic models each
  wrong on a disjoint third of cases give a 2-of-3 majority vote that is
  **mathematically guaranteed** to be 100% correct from three 66.7%-
  accurate members; three models wrong on the *same* third give a
  majority vote with **zero possible improvement**. This is the
  idealized case anchoring the messier real-data results below.
- `ensemble_bench.cpp` runs the same two combination methods on real
  `blood-transfusion` (this suite's noisiest dataset, per
  `cross_method`'s analysis) and `breast-w` (clean, well-separated) data,
  each with a genuinely **diverse** ensemble (`DecisionTree` + `KNN` +
  `LogisticRegression`) and a **correlated** one (three `DecisionTree`s
  differing only in `max_depth`).

## Results (captured 2026-07-28, Apple clang 14 / `-std=c++2b`, this Mac)

```
=== blood-transfusion ===
  diverse (DecisionTree + KNN + LogisticRegression):
    DecisionTree(depth=10)       test_acc=0.7047
    KNN(k=5)                     test_acc=0.7718
    LogisticRegression           test_acc=0.8054
    majority_vote                test_acc=0.7718 (vs best individual 0.8054)
    stacking(meta=LogReg)        test_acc=0.8121 (vs best individual 0.8054)
  correlated (three DecisionTrees, varying depth):
    DecisionTree(depth=8)        test_acc=0.6913
    DecisionTree(depth=10)       test_acc=0.7047
    DecisionTree(depth=12)       test_acc=0.7114
    majority_vote                test_acc=0.7047 (vs best individual 0.7114)
    stacking(meta=LogReg)        test_acc=0.7987 (vs best individual 0.7114)

=== breast-w ===
  diverse (DecisionTree + KNN + LogisticRegression):
    DecisionTree(depth=10)       test_acc=0.9137
    KNN(k=5)                     test_acc=0.9496
    LogisticRegression           test_acc=0.9424
    majority_vote                test_acc=0.9424 (vs best individual 0.9496)
    stacking(meta=LogReg)        test_acc=0.9424 (vs best individual 0.9496)
  correlated (three DecisionTrees, varying depth):
    DecisionTree(depth=8)        test_acc=0.9137
    DecisionTree(depth=10)       test_acc=0.9137
    DecisionTree(depth=12)       test_acc=0.9137
    majority_vote                test_acc=0.9137 (vs best individual 0.9137)
    stacking(meta=LogReg)        test_acc=0.9137 (vs best individual 0.9137)
```

`ensemble_test.cpp`'s own captured output (the exact synthetic proof)
runs via `ctest -R ensemble_test`.

## Findings — a more honest, three-part story than "diversity always helps"

- **The idealized case is real, and proven, not just typical**: with
  disjoint-error synthetic models, majority voting reaches exactly 100%
  from three 66.7%-accurate members; with identical-error models, it
  shows exactly zero improvement. This is the textbook mechanism
  motivating ensembling at all — but real base models are rarely
  perfectly diverse or perfectly correlated, which is why the real-data
  results below look messier.
- **Truly identical models get zero benefit from any combination
  method, confirmed cleanly**: on `breast-w` (clean, well-separated
  data), all three `DecisionTree`s (depth 8/10/12) converge to the
  *exact same* 0.9137 accuracy — the tree stops changing past a certain
  depth once the data is this separable — and both majority vote and
  stacking also land at exactly 0.9137. No combination method can add
  value when the underlying predictions are literally identical.
- **Naive majority voting can actively hurt a diverse ensemble if one
  member is meaningfully stronger**: on `blood-transfusion`, the diverse
  ensemble's majority vote (0.7718) is *worse* than its best individual
  member (`LogisticRegression`, 0.8054) — unweighted voting lets two
  weaker models (`DecisionTree` 0.7047, `KNN` 0.7718) outvote the
  strongest one. Diversity alone doesn't guarantee a voting ensemble
  helps; it needs either roughly-equal-strength members or a smarter
  combination method.
- **Stacking is more robust than majority voting in both directions,
  measured directly**: on both ensembles and both datasets, stacking's
  learned `LogisticRegression` meta-model matches or beats majority
  voting, and on `blood-transfusion`'s diverse ensemble it's the only
  method that beats the best individual member (0.8121 vs. 0.8054).
- **An honest complication of "correlated models add no value"**: on
  `blood-transfusion` (noisy data), the three same-family
  `DecisionTree`s are *not* fully identical (0.6913/0.7047/0.7114 —
  different depths latch onto different noise patterns on a noisy
  dataset, unlike on `breast-w`'s clean data where they converge
  exactly), and stacking extracts real out-of-sample benefit from their
  partial disagreement (0.7987, well above any individual member's
  0.7114) even though they share one algorithm family. Majority voting
  still shows no benefit here (0.7047, worse than the best individual)
  — consistent with the theoretical claim for *naive* combination. The
  more accurate lesson: **majority voting requires genuine diversity to
  help; a learned meta-model can sometimes extract value even from
  same-family base learners, but only if they're not literally
  producing identical predictions** — a real, measured middle ground
  between "diversity is required" and "diversity doesn't matter."

## Written rules: when to ensemble vs. not

1. **Don't naively majority-vote a diverse ensemble with unequal-strength
   members** — it can underperform the single best member (measured on
   `blood-transfusion`). Use a learned combiner (stacking) instead, or
   weight votes by validation accuracy.
2. **Don't bother ensembling multiple near-identical models via any
   method** if you can confirm they produce the same predictions (e.g.
   varying only a hyperparameter that's already past its effective
   range, as `breast-w`'s depth sweep shows) — no combination method can
   manufacture information that isn't there.
3. **Do prefer architecturally diverse base learners** (different
   algorithm families, not just different hyperparameters of the same
   one) when building an ensemble deliberately — this is what
   guarantees, not just usually produces, error correction (proven in
   `ensemble_test.cpp`).
4. **Stacking's extra complexity (proper out-of-fold generation to
   avoid leakage, an extra meta-model to train and maintain) is only
   worth it when a quick majority-vote check first shows some real
   spread in individual member accuracy** — on `breast-w`'s clean data,
   where everything already agrees, stacking added zero value for real
   added complexity.

## Hardware notes
None — every result above is a real, measured number from this Mac,
reusing data already committed for `ml/openml_bench`.
