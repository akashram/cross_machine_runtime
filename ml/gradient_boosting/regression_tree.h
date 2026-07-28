#pragma once
#include "decision_tree.h"  // Features/Labels aliases only -- no coupling to DecisionTree itself

#include <random>
#include <vector>

// The base learner GradientBoostedTrees boosts: a regression tree fit not
// to labels but to per-sample (gradient, Hessian) pairs of the current
// loss w.r.t. the running score, using Friedman (2001) section 4 / Chen &
// Guestrin (2016)'s second-order (Newton) approximation for both the
// split-gain criterion and the leaf value -- this is what makes it a
// "Newton step" tree rather than a plain MSE-reduction regression tree:
//   leaf weight w* = -softthresh(G, l1_reg) / (H + l2_reg)
//   split gain    = 0.5*[GL_term + GR_term - G_term] - min_split_gain
// where G/H are gradient/Hessian sums over a region and *_term is
// softthresh(sum, l1_reg)^2 / (sum_h + l2_reg).
struct GBTreeParams {
    int   max_depth          = 5;
    int   min_samples_leaf   = 1;
    int   min_samples_split  = 2;
    float l1_reg             = 0.0f;  // alpha
    float l2_reg             = 1.0f;  // lambda
    float min_split_gain     = 0.0f;  // gamma
};

struct GBTreeNode {
    bool  is_leaf = true;
    float leaf_value = 0.0f;
    int   feature_idx = -1;
    float threshold = 0.0f;
    int   left = -1;
    int   right = -1;
};

class GBRegressionTree {
public:
    explicit GBRegressionTree(GBTreeParams params = {});

    // Fits to (X[indices], gradients[indices], hessians[indices]).
    // `indices` is the (possibly row-subsampled) training subset for this
    // round; `feature_subset` is the (possibly column-subsampled) fixed
    // candidate feature set for the WHOLE tree -- GBTParams::colsample is
    // XGBoost-style colsample_bytree (one fixed subset per tree), a
    // deliberately different scheme from random_forest's per-split-node
    // resampling. Empty `feature_subset` means "use every feature".
    void fit(const Features& X, const std::vector<float>& gradients, const std::vector<float>& hessians,
             const std::vector<int>& indices, const std::vector<int>& feature_subset);

    float predict_one(const std::vector<float>& row) const;

private:
    GBTreeParams params_;
    std::vector<GBTreeNode> nodes_;
    int n_features_ = 0;

    int build_node(const Features& X, const std::vector<float>& g, const std::vector<float>& h,
                    const std::vector<int>& feature_subset, std::vector<int> indices, int depth);

    bool find_best_split(const Features& X, const std::vector<float>& g, const std::vector<float>& h,
                          const std::vector<int>& feature_subset, const std::vector<int>& indices, double G,
                          double H, int& best_feature, float& best_threshold, double& best_gain,
                          std::vector<int>& left_out, std::vector<int>& right_out) const;
};
