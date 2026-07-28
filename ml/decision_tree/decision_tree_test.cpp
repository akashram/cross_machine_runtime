// decision_tree_test.cpp — real correctness + parameter-effect checks,
// not just "it runs": a linearly-inseparable (XOR-shaped) dataset a
// depth-1 stump cannot solve but a deeper tree can, a feature-importance
// check against a dataset with one informative and one pure-noise
// feature, and a min_samples_leaf pruning-strength check.
#include "decision_tree.h"

#include <cstdio>
#include <random>

namespace {

int g_fails = 0;
void require(bool ok, const char *name) {
  std::printf("%s  %s\n", ok ? "PASS" : "FAIL", name);
  if (!ok) ++g_fails;
}

// XOR-shaped 2D dataset: label = (x0 > 0) XOR (x1 > 0). No single split
// on one feature separates the classes -- a depth-1 stump is stuck near
// 50% accuracy; a depth-2+ tree (split on x0, then x1 within each side)
// solves it exactly. This is the textbook reason tree depth matters.
void make_xor_dataset(int n, Features &X, Labels &y, std::mt19937 &rng) {
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  X.clear();
  y.clear();
  for (int i = 0; i < n; ++i) {
    float x0 = dist(rng), x1 = dist(rng);
    X.push_back({x0, x1});
    bool label = (x0 > 0) != (x1 > 0);
    y.push_back(label ? 1.0f : 0.0f);
  }
}

void test_xor_needs_depth() {
  std::mt19937 rng(1);
  Features X;
  Labels y;
  make_xor_dataset(400, X, y, rng);

  TreeParams stump_params;
  stump_params.max_depth = 1;
  DecisionTree stump(stump_params);
  stump.fit(X, y);
  float stump_acc = stump.score(X, y);

  TreeParams deep_params;
  deep_params.max_depth = 6;
  DecisionTree deep(deep_params);
  deep.fit(X, y);
  float deep_acc = deep.score(X, y);

  std::printf("  stump (depth<=1) train accuracy=%.3f, deep (depth<=6) train accuracy=%.3f\n",
              static_cast<double>(stump_acc), static_cast<double>(deep_acc));
  require(stump_acc < 0.7f, "a depth-1 stump cannot solve XOR (accuracy stays well below the deep tree's)");
  require(deep_acc > 0.98f, "a depth-6 tree solves XOR almost exactly");
}

// Feature 0 fully determines the label (label = x0 > 0); feature 1 is
// pure noise independent of the label. A correct impurity-decrease
// importance measure must attribute nearly all importance to feature 0.
void test_feature_importance_ignores_noise() {
  std::mt19937 rng(2);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  Features X;
  Labels y;
  for (int i = 0; i < 300; ++i) {
    float x0 = dist(rng), noise = dist(rng);
    X.push_back({x0, noise});
    y.push_back(x0 > 0 ? 1.0f : 0.0f);
  }

  DecisionTree tree;
  tree.fit(X, y);
  float imp0 = tree.feature_importance(0);
  float imp1 = tree.feature_importance(1);
  std::printf("  feature_importance: informative=%.4f noise=%.4f\n",
              static_cast<double>(imp0), static_cast<double>(imp1));
  require(imp0 > 0.95f, "the informative feature gets nearly all importance");
  require(imp1 < 0.05f, "the pure-noise feature gets nearly none");
}

// min_samples_leaf only binds when there's something for it to prevent:
// on a cleanly-separated XOR dataset (~100 samples per natural quadrant),
// a threshold of 20 never constrains anything -- both settings converge
// to the same natural tree. Add label noise (5% random flips) so an
// unconstrained tree has real overfitting to chase (isolating single
// mislabeled points into their own tiny leaves) and min_samples_leaf has
// something real to prevent.
void test_min_samples_leaf_reduces_leaves() {
  std::mt19937 rng(3);
  Features X;
  Labels y;
  make_xor_dataset(400, X, y, rng);

  std::uniform_real_distribution<float> flip_roll(0.0f, 1.0f);
  for (auto &label : y)
    if (flip_roll(rng) < 0.05f) label = 1.0f - label;

  TreeParams loose;
  loose.min_samples_leaf = 1;
  loose.max_depth = 10;
  DecisionTree loose_tree(loose);
  loose_tree.fit(X, y);

  TreeParams strict;
  strict.min_samples_leaf = 20;
  strict.max_depth = 10;
  DecisionTree strict_tree(strict);
  strict_tree.fit(X, y);

  std::printf("  n_leaves: min_samples_leaf=1 -> %d, min_samples_leaf=20 -> %d\n",
              loose_tree.n_leaves(), strict_tree.n_leaves());
  require(strict_tree.n_leaves() < loose_tree.n_leaves(),
          "a stricter min_samples_leaf produces a smaller (more pruned) tree "
          "once there's label noise for the unconstrained tree to overfit");
}

void test_perfectly_separable_reaches_100_percent() {
  Features X = {{0.1f}, {0.2f}, {0.3f}, {0.9f}, {1.0f}, {1.1f}};
  Labels y = {0, 0, 0, 1, 1, 1};
  DecisionTree tree;
  tree.fit(X, y);
  float acc = tree.score(X, y);
  require(acc == 1.0f, "a perfectly separable 1D dataset is fit exactly");
}

} // namespace

int main() {
  test_perfectly_separable_reaches_100_percent();
  test_xor_needs_depth();
  test_feature_importance_ignores_noise();
  test_min_samples_leaf_reduces_leaves();
  std::printf("%s\n", g_fails == 0 ? "PASS" : "FAIL");
  return g_fails == 0 ? 0 : 1;
}
