// random_forest_test.cpp — real correctness checks for bagging, OOB error
// estimation, and permutation importance, not just "it runs": OOB error
// (computed for free from the training set alone) must track true
// held-out error on a fresh sample from the same distribution; permutation
// importance must separate an informative feature from pure noise; and a
// forest of unconstrained trees must beat a single unconstrained tree on
// noisy data (the actual reason bagging exists -- variance reduction).
#include "random_forest.h"

#include <cmath>
#include <cstdio>
#include <random>

namespace {

int g_fails = 0;
void require(bool ok, const char *name) {
  std::printf("%s  %s\n", ok ? "PASS" : "FAIL", name);
  if (!ok) ++g_fails;
}

// XOR-shaped 2D dataset (see decision_tree_test.cpp) with `flip_prob`
// random label noise -- noise gives an unconstrained single tree real
// overfitting to chase, which is what lets bagging's variance reduction
// show up as a measurable accuracy gain.
void make_noisy_xor(int n, float flip_prob, Features &X, Labels &y, std::mt19937 &rng) {
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  std::uniform_real_distribution<float> flip_roll(0.0f, 1.0f);
  X.clear();
  y.clear();
  for (int i = 0; i < n; ++i) {
    float x0 = dist(rng), x1 = dist(rng);
    X.push_back({x0, x1});
    bool label = (x0 > 0) != (x1 > 0);
    float y_val = label ? 1.0f : 0.0f;
    if (flip_roll(rng) < flip_prob) y_val = 1.0f - y_val;
    y.push_back(y_val);
  }
}

// A forest of deep, unconstrained trees on noisy data should generalize
// better than a single deep, unconstrained tree -- the single tree
// overfits the noise directly, while bagging + per-split feature
// subsampling decorrelates the trees' errors so majority voting cancels
// much of it out. Measured on a FRESH held-out sample (not the training
// set), since train-set accuracy alone can't show overfitting.
void test_forest_beats_single_tree_on_noisy_data() {
  std::mt19937 rng(1);
  Features X_train, X_test;
  Labels y_train, y_test;
  make_noisy_xor(400, 0.15f, X_train, y_train, rng);
  make_noisy_xor(400, 0.15f, X_test, y_test, rng);

  TreeParams tp;
  tp.max_depth = -1;
  tp.min_samples_leaf = 1;
  DecisionTree single(tp);
  single.fit(X_train, y_train);
  float single_acc = single.score(X_test, y_test);

  RFParams rp;
  rp.n_estimators = 50;
  rp.max_depth = -1;
  rp.min_samples_leaf = 1;
  rp.random_state = 7;
  RandomForest forest(rp);
  forest.fit(X_train, y_train);
  float forest_acc = forest.score(X_test, y_test);

  std::printf("  held-out accuracy: single tree=%.3f, forest (50 trees)=%.3f\n",
              static_cast<double>(single_acc), static_cast<double>(forest_acc));
  require(forest_acc > single_acc,
          "bagging + feature subsampling generalizes better than a single unconstrained tree on noisy data");
}

// OOB error is computed entirely from the training set (majority vote
// among only the trees that didn't see a given sample in their bootstrap
// draw) -- it's supposed to approximate true held-out error without
// needing a separate validation split. Verify that by comparing it to
// error measured on an actual fresh sample from the same distribution.
void test_oob_error_tracks_held_out_error() {
  std::mt19937 rng(2);
  Features X_train, X_test;
  Labels y_train, y_test;
  make_noisy_xor(500, 0.1f, X_train, y_train, rng);
  make_noisy_xor(500, 0.1f, X_test, y_test, rng);

  RFParams rp;
  rp.n_estimators = 80;
  rp.random_state = 11;
  RandomForest forest(rp);
  forest.fit(X_train, y_train);

  float oob_err = forest.oob_error();
  float true_test_err = 1.0f - forest.score(X_test, y_test);
  std::printf("  oob_error=%.3f, true held-out error=%.3f\n",
              static_cast<double>(oob_err), static_cast<double>(true_test_err));
  require(std::fabs(oob_err - true_test_err) < 0.08f,
          "OOB error (computed from the training set alone) is close to true held-out error on a fresh sample");
}

// Same feature-importance separation test as decision_tree_test.cpp, but
// through RandomForest::feature_importances() (Breiman's OOB-permutation
// definition) rather than DecisionTree's impurity-based one -- a
// different measurement path that should reach the same qualitative
// conclusion.
void test_permutation_importance_ignores_noise() {
  std::mt19937 rng(3);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  Features X;
  Labels y;
  for (int i = 0; i < 400; ++i) {
    float x0 = dist(rng), noise = dist(rng);
    X.push_back({x0, noise});
    y.push_back(x0 > 0 ? 1.0f : 0.0f);
  }

  RFParams rp;
  rp.n_estimators = 50;
  rp.random_state = 5;
  RandomForest forest(rp);
  forest.fit(X, y);

  std::vector<float> importances = forest.feature_importances();
  std::printf("  permutation importance: informative=%.4f noise=%.4f\n",
              static_cast<double>(importances[0]), static_cast<double>(importances[1]));
  require(importances[0] > 0.1f, "the informative feature shows a real OOB-accuracy drop when permuted");
  require(importances[1] < importances[0] / 2.0f, "the pure-noise feature's permutation importance is much smaller");
}

// Same random_state must reproduce identical trees/predictions -- the
// bootstrap sampler and per-tree DecisionTree RNGs are both seeded
// deterministically from RFParams::random_state, so parallel fitting
// (via foundation::WorkStealingPool) must not introduce nondeterminism.
void test_same_seed_is_reproducible() {
  std::mt19937 rng(4);
  Features X;
  Labels y;
  make_noisy_xor(200, 0.1f, X, y, rng);

  RFParams rp;
  rp.n_estimators = 20;
  rp.random_state = 99;
  RandomForest forest_a(rp);
  forest_a.fit(X, y);
  RandomForest forest_b(rp);
  forest_b.fit(X, y);

  Labels pred_a = forest_a.predict(X);
  Labels pred_b = forest_b.predict(X);
  bool identical = pred_a == pred_b;
  require(identical, "two forests fit with the same random_state produce identical predictions despite parallel fitting");
}

} // namespace

int main() {
  test_forest_beats_single_tree_on_noisy_data();
  test_oob_error_tracks_held_out_error();
  test_permutation_importance_ignores_noise();
  test_same_seed_is_reproducible();
  std::printf("%s\n", g_fails == 0 ? "PASS" : "FAIL");
  return g_fails == 0 ? 0 : 1;
}
