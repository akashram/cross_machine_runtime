// ensemble_test.cpp — real correctness checks: k-fold assignment
// actually partitions and balances indices, stacking's out-of-fold
// features never leak a row's own label through memorization, and --
// PLAN.md step 13's central empirical claim -- a majority vote across
// models with genuinely diverse (disjoint) error patterns fully
// corrects those errors, while the identical vote across models with
// correlated (identical) error patterns doesn't improve on any single
// member at all. Constructed so the diverse/correlated cases are
// mathematically guaranteed, not just usually true on this seed.
#include "ensemble.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <set>

namespace {

int g_fails = 0;
void require(bool ok, const char *name) {
  std::printf("%s  %s\n", ok ? "PASS" : "FAIL", name);
  if (!ok) ++g_fails;
}

void test_k_fold_assignment_partitions_and_balances() {
  std::size_t n = 103;
  int k = 5;
  std::vector<int> fold = k_fold_assignment(n, k, 11);

  bool all_in_range = true;
  std::vector<int> counts(static_cast<std::size_t>(k), 0);
  for (int f : fold) {
    if (f < 0 || f >= k) all_in_range = false;
    else ++counts[static_cast<std::size_t>(f)];
  }
  int min_count = counts[0], max_count = counts[0];
  for (int c : counts) { min_count = std::min(min_count, c); max_count = std::max(max_count, c); }

  std::printf("  fold sizes: min=%d max=%d (n=%zu, k=%d)\n", min_count, max_count, n, k);
  require(all_in_range, "every index gets a fold id in [0, k)");
  require(max_count - min_count <= 1, "fold sizes are balanced to within 1 (n not evenly divisible by k)");
}

// A "memorizing" base model: predicts 1.0 for any query row whose first
// feature value was present anywhere in its training set, 0.0
// otherwise. Since every row here has a unique feature value, this
// model can only predict 1.0 for a row if that exact row leaked into
// its training fold. Real k-fold stacking must never let a row see
// itself, so every out-of-fold prediction below must be 0.0.
Labels memorizing_model(const Features &X_train, const Labels &, const Features &X_query) {
  std::set<float> seen;
  for (const auto &row : X_train) seen.insert(row[0]);
  Labels out(X_query.size());
  for (std::size_t i = 0; i < X_query.size(); ++i) out[i] = seen.count(X_query[i][0]) ? 1.0f : 0.0f;
  return out;
}

void test_stacking_oof_features_never_leak() {
  Features X;
  Labels y;
  for (int i = 0; i < 50; ++i) {
    X.push_back({static_cast<float>(i)});
    y.push_back(0.0f);
  }
  std::vector<FitPredictFn> models = {memorizing_model};
  Features meta = stacking_oof_features(X, y, models, 5, 3);

  bool any_leak = false;
  for (const auto &row : meta)
    if (row[0] != 0.0f) any_leak = true;

  std::printf("  memorizing-model OOF leak detected=%d\n", any_leak);
  require(!any_leak, "no out-of-fold prediction ever reflects a row that leaked into its own training fold");
}

FitPredictFn make_parity_model(int wrong_mod) {
  return [wrong_mod](const Features &, const Labels &, const Features &X_query) {
    Labels out(X_query.size());
    for (std::size_t i = 0; i < X_query.size(); ++i) {
      int idx = static_cast<int>(X_query[i][0]);
      float true_label = (idx % 2 == 0) ? 1.0f : 0.0f;
      bool make_wrong = (idx % 3 == wrong_mod);
      out[i] = make_wrong ? (1.0f - true_label) : true_label;
    }
    return out;
  };
}

float accuracy(const Labels &pred, const Labels &truth) {
  int correct = 0;
  for (std::size_t i = 0; i < truth.size(); ++i)
    if (pred[i] == truth[i]) ++correct;
  return static_cast<float>(correct) / static_cast<float>(truth.size());
}

// The central empirical claim of PLAN.md step 13, made mathematically
// exact rather than merely likely: three "diverse" models each wrong on
// a disjoint third of cases (idx%3 == 0, 1, 2 respectively) -- for any
// given case, exactly one of the three is wrong, so a 2-of-3 majority
// vote is ALWAYS correct, 100% ensemble accuracy from three 66.7%-
// accurate individual models. Three "correlated" models that are all
// wrong on the SAME third of cases produce a majority vote that is
// exactly as wrong as any individual member -- no improvement at all.
void test_diverse_ensemble_beats_correlated_ensemble() {
  Features X;
  Labels y;
  for (int i = 0; i < 99; ++i) {
    X.push_back({static_cast<float>(i)});
    y.push_back((i % 2 == 0) ? 1.0f : 0.0f);
  }

  std::vector<FitPredictFn> diverse_models = {make_parity_model(0), make_parity_model(1), make_parity_model(2)};
  std::vector<FitPredictFn> correlated_models = {make_parity_model(0), make_parity_model(0), make_parity_model(0)};

  float individual_acc = accuracy(diverse_models[0]({}, {}, X), y);  // same for every parity model by construction
  Labels diverse_vote = majority_vote(X, y, X, diverse_models);
  Labels correlated_vote = majority_vote(X, y, X, correlated_models);
  float diverse_acc = accuracy(diverse_vote, y);
  float correlated_acc = accuracy(correlated_vote, y);

  std::printf("  individual model accuracy=%.4f, diverse ensemble=%.4f, correlated ensemble=%.4f\n",
              static_cast<double>(individual_acc), static_cast<double>(diverse_acc), static_cast<double>(correlated_acc));
  require(std::fabs(individual_acc - (2.0f / 3.0f)) < 1e-6f, "each individual parity model is exactly 66.7% accurate by construction");
  require(diverse_acc == 1.0f, "three models with disjoint error patterns reach 100% ensemble accuracy via majority vote");
  require(correlated_acc == individual_acc, "three models with identical error patterns show zero improvement from ensembling");
}

}  // namespace

int main() {
  test_k_fold_assignment_partitions_and_balances();
  test_stacking_oof_features_never_leak();
  test_diverse_ensemble_beats_correlated_ensemble();
  std::printf("%s\n", g_fails == 0 ? "PASS" : "FAIL");
  return g_fails == 0 ? 0 : 1;
}
