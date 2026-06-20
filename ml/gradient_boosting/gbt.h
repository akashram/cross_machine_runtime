#pragma once
#include "decision_tree/decision_tree.h"
// TODO: implement Friedman GBT with Newton (second-order) steps
// Reference: Friedman (2001) "Greedy Function Approximation: A Gradient Boosting Machine"

struct GBTParams {
    int   n_estimators     = 100;
    float learning_rate    = 0.1f;
    int   max_depth        = 5;
    float subsample        = 0.8f;       // fraction of samples per tree
    float colsample        = 0.8f;       // fraction of features per tree
    float l1_reg           = 0.0f;       // alpha
    float l2_reg           = 1.0f;       // lambda
    int   max_bins         = 256;        // histogram-based split finding (LightGBM-style)
};

class GradientBoostedTrees {
public:
    explicit GradientBoostedTrees(GBTParams params = {});

    void fit(const Features& X, const Labels& y);
    Labels predict(const Features& X) const;
    float score(const Features& X, const Labels& y) const;

    // Staged prediction for validation curve (predict after k trees)
    std::vector<float> staged_predict(const Features& X, int k) const;
};
