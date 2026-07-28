#pragma once
#include <functional>
#include <vector>

// PLAN.md Phase 12b step 13: stacking (diverse base models + meta-
// learner), blending, empirical validation that diversity is necessary.

using Features = std::vector<std::vector<float>>;
using Labels   = std::vector<float>;

// A base learner, erased behind one signature so stacking can drive
// heterogeneous algorithm types (DecisionTree, KNNClassifier,
// LinearModel, ...) uniformly: fit on (X_train, y_train), predict on
// X_query. Hard {0,1} predictions are used as meta-features rather than
// soft probabilities -- an honest, simpler design choice (not every
// Phase 12a classifier exposes predict_proba(); see README's Design
// note), the same "hard-label stacking" mode sklearn's
// StackingClassifier also supports, not a workaround.
using FitPredictFn = std::function<Labels(const Features& X_train, const Labels& y_train, const Features& X_query)>;

// Assigns each of the n samples a fold id in [0, k_folds), shuffled.
std::vector<int> k_fold_assignment(std::size_t n, int k_folds, unsigned random_state);

// Out-of-fold stacking meta-features: for each base model, k-fold cross-
// validate over (X, y) so every row's meta-feature comes from a model
// that never saw that row during training (no leakage). Returns an
// [n, base_models.size()] matrix.
Features stacking_oof_features(const Features& X, const Labels& y, const std::vector<FitPredictFn>& base_models, int k_folds,
                                unsigned random_state);

// Test-time meta-features: each base model is trained once on the full
// (X_train, y_train) and predicts on X_test. Returns an
// [X_test.size(), base_models.size()] matrix.
Features stacking_test_features(const Features& X_train, const Labels& y_train, const Features& X_test,
                                 const std::vector<FitPredictFn>& base_models);

// Simple unweighted majority vote across base models' hard predictions
// on X_test -- the "blending" baseline stacking is compared against.
Labels majority_vote(const Features& X_train, const Labels& y_train, const Features& X_test, const std::vector<FitPredictFn>& base_models);
