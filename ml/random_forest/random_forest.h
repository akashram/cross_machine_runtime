#pragma once
#include "decision_tree.h"
#include <cstddef>

// PLAN.md Phase 12a step 2: bagging with bootstrap samples,
// max_features = sqrt(p) (classification) by default, out-of-bag error
// estimation, feature importance via permutation, parallelized over
// trees using foundation/'s work-stealing pool.

struct RFParams {
    int    n_estimators    = 100;
    int    max_depth       = -1;  // unlimited
    int    min_samples_leaf = 1;
    // >= 1.0: absolute feature count per split. In (0, 1): fraction of
    // n_features. -1 (default): sqrt(n_features), the classification
    // default PLAN.md names.
    float  max_features    = -1.0f;
    int    random_state    = 42;
};

class RandomForest {
public:
    explicit RandomForest(RFParams params = {});

    void fit(const Features& X, const Labels& y);
    Labels predict(const Features& X) const;
    float score(const Features& X, const Labels& y) const;

    // Out-of-bag error: for each training sample, majority-vote only
    // among trees whose bootstrap sample excluded it, compare to the
    // true label. A held-out-quality estimate that costs nothing extra
    // (no separate validation split needed) -- valid only right after
    // fit() on the same X/y it was trained on.
    float oob_error() const { return oob_error_; }

    // Permutation importance (Breiman's original RF definition, not
    // decision_tree's impurity-based importance): for each feature,
    // shuffle that column across all training samples, re-run OOB
    // prediction with the shuffled column, and report the OOB accuracy
    // drop. A feature the forest doesn't actually rely on shows ~0 drop;
    // a load-bearing feature shows a real drop, measured on held-out
    // (OOB) predictions rather than training-set impurity, so it isn't
    // fooled by a feature that just happens to fit the training data
    // (unlike a naive impurity-based measure would be).
    std::vector<float> feature_importances() const;

private:
    RFParams params_;
    std::vector<DecisionTree> trees_;
    std::vector<std::vector<char>> in_bag_;  // in_bag_[t][i] = true if sample i was in tree t's bootstrap sample
    Features X_;  // retained post-fit() for permutation importance
    Labels y_;
    int n_features_ = 0;
    int n_classes_ = 0;
    float oob_error_ = -1.0f;

    // OOB accuracy of the forest against `X_eval`'s column `permute_feature`
    // shuffled per `permutation` (or unshuffled, if permute_feature < 0).
    // Shared by oob_error() computation (permute_feature=-1) and
    // feature_importances() (one call per feature).
    float oob_accuracy_with_permutation(int permute_feature, const std::vector<int>& permutation) const;
};
