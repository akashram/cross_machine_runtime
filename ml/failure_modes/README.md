# failure_modes

**Status: code-complete AND locally run — pure CPU, no external
dependency.** All 8 demonstrations use synthetic data constructed
specifically to isolate one mechanism each, so the failure is real and
reproducible, not an artifact of one unlucky dataset.

## What this measures

PLAN.md Phase 12b step 14: for each Phase 12a algorithm, a concrete
example of it failing badly, and why. PLAN.md names four explicitly (GBT
on noisy labels, k-NN in high dimensions, SVM at scale, RF on highly
imbalanced classes); this catalog covers all 8 algorithms.

## Design

One executable, `failure_modes_bench`, run manually (same convention as
`openml_bench`/`hyperparam_sweep`/`ensemble_bench`), demonstrates each
failure mode against synthetic data tuned to isolate the specific
mechanism. Several first attempts at these demonstrations didn't
actually show a failure and were re-tuned — documented honestly below
and in the source comments, not silently fixed:

- The imbalance experiment's first version evaluated on the *training*
  set, where unlimited-depth `RandomForest` simply memorized the 50 rare
  minority points (100% minority recall — no failure at all). Fixed by
  evaluating on a separate, class-balanced held-out test set, the fair
  way to expose how the model generalizes on the ambiguous region
  between classes.
- The KNN high-dimensionality experiment's first version used a large,
  easily-separated dataset where even 200 noise dimensions barely
  degraded accuracy (0.9867). Fixed by shrinking the training set (120
  rows) and class separation — the curse of dimensionality bites harder
  with less data relative to the noise dimensions added.
- The DecisionTree-instability experiment's first version used only the
  2 informative features, no extras. `RandomForest`'s per-split feature
  subsampling defaults to `sqrt(n_features)` — with only 2 features,
  that's ~1, meaning *every* tree in the forest already considers just 1
  random feature per split. The forest ended up **less** stable than a
  single tree (0.083 vs. 0.058 disagreement) — a real, honestly
  surprising finding in its own right (see below), but not what this
  section is trying to isolate. Adding 3 pure-noise features (5 total,
  `sqrt(5)≈2` features per split) gave the forest room to actually
  average over meaningfully different feature subsets, and it then
  showed the expected stability benefit (0.108 vs. 0.175).

## Results (captured 2026-07-28, Apple clang 14 / `-std=c++2b`, this Mac)

```
=== RandomForest: highly imbalanced classes (95:5 train), held-out balanced test ===
  balanced-test-set overall accuracy=0.9225
  majority-class recall=0.9900, minority-class recall=0.8550 (the real failure, hidden by overall accuracy alone)

=== GBT vs RandomForest: controlled label noise ===
  flip_rate=0.0: GBT test_acc=1.0000, RandomForest test_acc=1.0000
  flip_rate=0.3: GBT test_acc=0.8150, RandomForest test_acc=0.8400

=== SVM: training time vs. n (O(n^2) kernel matrix) ===
  n=200    train_time=0.0512s  (time ratio=0.00, n-ratio^2=0.00)
  n=400    train_time=0.2504s  (time ratio=4.89, n-ratio^2=4.00)
  n=800    train_time=1.0840s  (time ratio=4.33, n-ratio^2=4.00)
  n=1600   train_time=6.3266s  (time ratio=5.84, n-ratio^2=4.00)

=== KNN vs LinearModel: accuracy as irrelevant noise dimensions grow ===
  noise_dims=0    (total d=2   ) KNN test_acc=0.8867, LogisticRegression test_acc=0.9067
  noise_dims=20   (total d=22  ) KNN test_acc=0.7600, LogisticRegression test_acc=0.8333
  noise_dims=100  (total d=102 ) KNN test_acc=0.6533, LogisticRegression test_acc=0.7667
  noise_dims=300  (total d=302 ) KNN test_acc=0.5333, LogisticRegression test_acc=0.6867

=== KMeans: purity on two interleaved moons (non-convex clusters) ===
  purity against true moon labels=0.7533 (0.5 = chance for k=2)

=== PCA + LogisticRegression: high-variance noise dim swamps low-variance signal ===
  component[0] direction=[-1.000, 0.004] (near [1,0] means it kept the noise dim, not the signal dim)
  classifier on raw 2D features: train_acc=1.0000
  classifier on PCA(n_components=1): train_acc=0.5050

=== DecisionTree vs RandomForest: prediction instability across bootstrap resamples ===
  two DecisionTrees on two bootstrap resamples of the same data disagree on 0.1750 of predictions
  two RandomForests (100 trees each) on the same two resamples disagree on 0.1083 of predictions

=== LinearModel vs SVM(RBF): XOR (no linear decision boundary exists) ===
  best LogisticRegression accuracy across an alpha sweep=0.5233 (no alpha setting escapes chance-level)
  SVM(RBF) accuracy=0.9667 (kernel trick captures the nonlinear boundary directly)
```

## Catalog: mechanism behind each failure

1. **RandomForest — highly imbalanced classes.** 95:5 imbalanced
   training data with real class overlap: balanced-test-set overall
   accuracy (0.9225) looks fine, but minority recall (0.855) is
   meaningfully worse than majority recall (0.990) — the 14.5-point gap
   overall accuracy alone completely hides. Mechanism: with 19x fewer
   minority examples, the forest sees far less of the minority class's
   boundary-region variation, so ambiguous points near the true
   decision boundary default toward the majority class purely from
   having seen 19x more majority examples there.
2. **GBT — noisy labels.** With 0% flipped labels both GBT and
   RandomForest reach 100%; with 30% flipped training labels, GBT drops
   to 0.815 vs. RandomForest's 0.840 — a real, measured (if here modest)
   gap in the theoretically-expected direction. Mechanism: each boosting
   round fits the *current residual*; once the real signal is captured,
   further rounds fit residual label noise specifically, while
   RandomForest's bagged trees each see an independent bootstrap sample
   and average away noise that isn't consistent across resamples. This
   is the same mechanism `hyperparam_sensitivity`'s `n_estimators` sweep
   found directly on a real dataset (GBT's test accuracy *declining*
   with more rounds on `blood-transfusion`) — this experiment isolates
   it with controlled synthetic noise instead of relying on one real
   dataset's ambient noise level.
3. **SVM — at scale.** Training time roughly quadruples with every
   doubling of `n` (ratios 4.89, 4.33, 5.84 against a theoretical 4.00
   for pure `O(n^2)`) — confirms the simplified-SMO SVM's full kernel-
   matrix construction is the real bottleneck it's documented to be
   (`svm`'s own README), not just an asymptotic claim. 1600 rows already
   takes 6.3 seconds; this doesn't scale to the tens-of-thousands-of-rows
   regime without a linear-kernel or approximate-kernel variant.
4. **KNN — high dimensions.** Accuracy degrades from 0.887 to 0.533
   (barely above the 50% chance floor) as 300 pure-noise dimensions are
   added, while `LogisticRegression` on the identical data degrades much
   more gently (0.907 to 0.687). Mechanism: KNN's Euclidean distance
   treats every dimension as equally relevant, so noise dimensions
   increasingly dominate the distance calculation as they accumulate;
   `LogisticRegression`'s learned per-feature weights can drive
   irrelevant dimensions toward near-zero influence instead.
5. **KMeans — non-convex clusters.** Purity against the true labels on
   two interleaved half-moons is 0.753 — well above chance (0.5) but far
   below the ~1.0 purity `KMeans` reaches on well-separated convex blobs
   elsewhere in this repo (`kmeans`'s own README). Mechanism: Lloyd's
   iteration partitions space by distance to a centroid, which only ever
   produces convex (specifically, Voronoi) regions — no centroid-based
   partition can correctly separate two interleaved crescents, no matter
   how many iterations run.
6. **PCA — discards a discriminative low-variance direction.** A
   classifier on the raw 2D data reaches 100% train accuracy; the same
   classifier after reducing to 1 PCA component collapses to 50.5%
   (chance level). The retained component (`[-1.000, 0.004]`) is almost
   exactly the noise dimension, not the signal dimension. Mechanism: PCA
   selects directions by *variance alone*, with no notion of which
   directions are class-discriminative — when the class-relevant signal
   sits in a low-variance direction dominated by unrelated high-variance
   noise, PCA throws the useful direction away first.
7. **DecisionTree — high variance / instability across resamples.** Two
   single trees fit on two different bootstrap resamples of the *same*
   underlying data disagree on 17.5% of predictions over the full
   dataset; two 100-tree RandomForests fit on the same two resamples
   disagree on only 10.8% — bagging's averaging measurably reduces (not
   eliminates) how much the learned model depends on which specific
   sample happened to be drawn. **A genuine complication found while
   tuning this experiment**: with only 2 total features, RandomForest's
   default per-split feature subsampling (`sqrt(n_features)≈1`) made the
   forest *less* stable than a single tree (0.083 vs. 0.058) — when
   there are too few features for subsampling to meaningfully vary
   between splits, aggressive subsampling can inject more per-tree noise
   than bagging's averaging compensates for. This is a real, disclosed
   edge case of Breiman's algorithm, not swept under the rug.
8. **LinearModel — non-linearly-separable data (XOR).** Best accuracy
   across an `alpha` sweep from `1e-5` to `10` is 0.523 — no
   regularization setting escapes chance level, while `SVM` with an RBF
   kernel reaches 0.967 on the identical data. Mechanism: XOR has no
   linear decision boundary in the original feature space at all; no
   amount of L2 regularization changes that fundamental fact, while the
   RBF kernel implicitly maps into a space where the boundary becomes
   near-linear.

## Hardware notes
None — every result above is a real, measured number from this Mac on
synthetic data.
