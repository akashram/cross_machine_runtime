// gbt_test.cpp — real correctness checks for Newton-step gradient
// boosting, not just "it runs": training loss must actually decrease
// round over round (the entire point of boosting), a linearly separable
// dataset must be fit exactly, an XOR-shaped dataset needs multiple
// rounds of depth>=2 trees the way it needed tree depth in
// decision_tree_test.cpp, and stronger L2 regularization must shrink
// predictions toward the base rate (Newton leaf weight w* = -G/(H+lambda)
// shrinks as lambda grows).
#include "gbt.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <random>

namespace {

int g_fails = 0;
void require(bool ok, const char *name) {
  std::printf("%s  %s\n", ok ? "PASS" : "FAIL", name);
  if (!ok) ++g_fails;
}

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

double logloss(const std::vector<float> &proba, const Labels &y) {
  double total = 0.0;
  for (std::size_t i = 0; i < y.size(); ++i) {
    double p = std::clamp(static_cast<double>(proba[i]), 1e-7, 1.0 - 1e-7);
    total -= static_cast<double>(y[i]) * std::log(p) + (1.0 - static_cast<double>(y[i])) * std::log(1.0 - p);
  }
  return total / static_cast<double>(y.size());
}

void test_perfectly_separable_reaches_high_accuracy() {
  Features X = {{0.1f}, {0.2f}, {0.3f}, {0.9f}, {1.0f}, {1.1f}};
  Labels y = {0, 0, 0, 1, 1, 1};
  GBTParams p;
  p.n_estimators = 30;
  p.max_depth = 2;
  p.subsample = 1.0f;
  p.colsample = 1.0f;
  GradientBoostedTrees gbt(p);
  gbt.fit(X, y);
  float acc = gbt.score(X, y);
  std::printf("  train accuracy on separable 1D data=%.3f\n", static_cast<double>(acc));
  require(acc == 1.0f, "a perfectly separable 1D dataset is fit exactly");
}

// Boosting's entire premise: each round's tree is fit to the CURRENT
// residual gradient, so adding it should reduce training loss further.
// Verify via staged_predict that loss after all rounds is well below
// loss after the first few rounds.
void test_training_loss_decreases_with_rounds() {
  std::mt19937 rng(1);
  Features X;
  Labels y;
  make_xor_dataset(400, X, y, rng);

  GBTParams p;
  p.n_estimators = 60;
  p.max_depth = 3;
  p.learning_rate = 0.2f;
  p.subsample = 1.0f;
  p.colsample = 1.0f;
  GradientBoostedTrees gbt(p);
  gbt.fit(X, y);

  double loss_early = logloss(gbt.staged_predict(X, 3), y);
  double loss_late = logloss(gbt.staged_predict(X, 60), y);
  std::printf("  training logloss: after 3 rounds=%.4f, after 60 rounds=%.4f\n", loss_early, loss_late);
  require(loss_late < loss_early * 0.5, "training loss drops substantially as more boosting rounds are added");
}

// XOR has no single-feature split, so it needs BOTH tree depth (to
// combine both features in one tree) AND multiple boosting rounds (to
// refine the decision boundary) -- analogous to decision_tree_test.cpp's
// stump-vs-deep-tree XOR check, but for the boosted ensemble.
void test_boosted_ensemble_solves_xor() {
  std::mt19937 rng(2);
  Features X;
  Labels y;
  make_xor_dataset(400, X, y, rng);

  GBTParams shallow;
  shallow.n_estimators = 40;
  shallow.max_depth = 1;  // stumps: each tree sees only one feature's split
  shallow.subsample = 1.0f;
  shallow.colsample = 1.0f;
  GradientBoostedTrees shallow_gbt(shallow);
  shallow_gbt.fit(X, y);
  float shallow_acc = shallow_gbt.score(X, y);

  GBTParams deep;
  deep.n_estimators = 40;
  deep.max_depth = 3;  // deep enough for one tree to combine both features
  deep.subsample = 1.0f;
  deep.colsample = 1.0f;
  GradientBoostedTrees deep_gbt(deep);
  deep_gbt.fit(X, y);
  float deep_acc = deep_gbt.score(X, y);

  std::printf("  train accuracy: depth-1 stumps=%.3f, depth-3 trees=%.3f\n", static_cast<double>(shallow_acc),
              static_cast<double>(deep_acc));
  require(deep_acc > 0.95f, "an ensemble of depth-3 trees solves XOR almost exactly");
  require(deep_acc > shallow_acc, "deeper base learners solve XOR better than depth-1 stumps, same as a single tree would");
}

// Newton leaf weight w* = -G/(H + lambda): larger lambda shrinks every
// leaf weight toward zero, which should pull predicted probabilities
// toward the constant base rate (less spread across samples).
void test_l2_regularization_shrinks_predictions() {
  std::mt19937 rng(3);
  Features X;
  Labels y;
  make_xor_dataset(300, X, y, rng);

  GBTParams weak_reg;
  weak_reg.n_estimators = 30;
  weak_reg.max_depth = 3;
  weak_reg.l2_reg = 0.1f;
  weak_reg.subsample = 1.0f;
  weak_reg.colsample = 1.0f;
  GradientBoostedTrees weak_gbt(weak_reg);
  weak_gbt.fit(X, y);

  GBTParams strong_reg = weak_reg;
  strong_reg.l2_reg = 50.0f;
  GradientBoostedTrees strong_gbt(strong_reg);
  strong_gbt.fit(X, y);

  auto variance = [](const std::vector<float> &v) {
    double mean = 0.0;
    for (float x : v) mean += static_cast<double>(x);
    mean /= static_cast<double>(v.size());
    double var = 0.0;
    for (float x : v) var += (static_cast<double>(x) - mean) * (static_cast<double>(x) - mean);
    return var / static_cast<double>(v.size());
  };

  double weak_var = variance(weak_gbt.predict_proba(X));
  double strong_var = variance(strong_gbt.predict_proba(X));
  std::printf("  predicted-probability variance: l2_reg=0.1 -> %.5f, l2_reg=50.0 -> %.5f\n", weak_var, strong_var);
  require(strong_var < weak_var, "stronger L2 regularization shrinks Newton leaf weights, reducing prediction spread");
}

}  // namespace

int main() {
  test_perfectly_separable_reaches_high_accuracy();
  test_training_loss_decreases_with_rounds();
  test_boosted_ensemble_solves_xor();
  test_l2_regularization_shrinks_predictions();
  std::printf("%s\n", g_fails == 0 ? "PASS" : "FAIL");
  return g_fails == 0 ? 0 : 1;
}
