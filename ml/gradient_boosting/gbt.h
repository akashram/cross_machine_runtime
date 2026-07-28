#pragma once
#include "decision_tree.h"  // Features/Labels aliases
#include "regression_tree.h"

#include <random>
#include <vector>

// PLAN.md Phase 12a step 3: Friedman gradient boosting (2001, section 4)
// with Newton (second-order) split-gain/leaf-value steps -- see
// regression_tree.h for the base learner. Binary classification via
// logistic loss: each round computes the gradient/Hessian of the
// log-loss w.r.t. the running score F(x), fits a GBRegressionTree to
// them, and adds learning_rate * tree(x) to F. Row subsampling
// (`subsample`, Friedman 2002's stochastic gradient boosting) and column
// subsampling (`colsample`, one fixed subset per tree -- XGBoost-style
// colsample_bytree, distinct from random_forest's per-split resampling)
// both regularize against overfitting to any single round's gradient.
//
// Histogram-based binned split finding (LightGBM-style, `max_bins` in
// PLAN.md's framing) is NOT implemented -- this uses exact per-feature
// sort-based split search, same honest scope caveat as decision_tree
// (see its README's "vectorized split search" note). A real, scoped-out
// follow-up, not implicitly claimed here.
struct GBTParams {
    int      n_estimators    = 100;
    float    learning_rate   = 0.1f;
    int      max_depth       = 5;
    float    subsample       = 0.8f;   // fraction of samples per round (without replacement)
    float    colsample       = 0.8f;   // fraction of features per tree (fixed subset, whole tree)
    float    l1_reg          = 0.0f;   // alpha -- leaf-weight soft-thresholding
    float    l2_reg          = 1.0f;   // lambda -- Newton-step leaf regularization
    unsigned random_state    = 0;
};

class GradientBoostedTrees {
public:
    explicit GradientBoostedTrees(GBTParams params = {});

    void fit(const Features& X, const Labels& y);  // y in {0.0, 1.0} -- binary logistic loss
    Labels predict(const Features& X) const;        // thresholded at p=0.5
    float score(const Features& X, const Labels& y) const;

    // Predicted positive-class probability per row (sigmoid of the
    // full-ensemble raw score).
    std::vector<float> predict_proba(const Features& X) const;

    // Predicted positive-class probability using only the first k trees
    // -- a training/validation curve over boosting rounds.
    std::vector<float> staged_predict(const Features& X, int k) const;

private:
    GBTParams params_;
    std::vector<GBRegressionTree> trees_;
    float base_score_ = 0.0f;  // initial log-odds (prior class balance)
    int n_features_ = 0;
    mutable std::mt19937 rng_;

    float raw_score(const std::vector<float>& row, int n_trees) const;
};
