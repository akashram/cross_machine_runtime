#pragma once
#include <random>
#include <vector>
#include <span>

// PLAN.md Phase 12a step 1: CART (Breiman et al.) — Gini/entropy split
// search, pre-pruning (max_depth, min_samples_leaf, min_samples_split,
// min_impurity_decrease). Classification only (labels are small
// non-negative integers stored as float, e.g. 0.0/1.0/2.0 for a 3-class
// problem) — PLAN.md's Phase 12a explicitly frames this against Gini/
// entropy impurity, which is a classification-only concept; a regression
// variant (MSE-reduction splits) would be a real but separate extension,
// not attempted here.

using Features = std::vector<std::vector<float>>;  // [n_samples, n_features]
using Labels   = std::vector<float>;

struct TreeParams {
    int   max_depth        = 10;
    int   min_samples_leaf = 1;
    int   min_samples_split = 2;
    float min_impurity_decrease = 0.0f;
    enum class Criterion { GINI, ENTROPY } criterion = Criterion::GINI;

    // -1 = consider every feature at every split (plain CART). >0 =
    // consider a random subset of this size at EACH split node
    // independently -- the actual per-node feature-subsampling
    // RandomForest (step 2) needs, per Breiman's algorithm (not a
    // once-per-tree subset, which is a different, weaker technique
    // sometimes confused with it).
    int max_features = -1;
    unsigned random_state = 0;
};

// One node of the tree, stored flat in DecisionTree::nodes_ (indices, not
// pointers) so the tree is a single contiguous, cache-friendly, trivially
// copyable vector -- RandomForest (step 2) holds many of these.
struct TreeNode {
    bool  is_leaf = true;
    int   predicted_class = 0;
    int   feature_idx = -1;
    float threshold = 0.0f;
    int   left = -1;
    int   right = -1;
    int   n_samples = 0;
    float impurity = 0.0f;
};

class DecisionTree {
public:
    explicit DecisionTree(TreeParams params = {});

    void fit(const Features& X, const Labels& y);
    Labels predict(const Features& X) const;
    float score(const Features& X, const Labels& y) const;

    int   depth() const;
    int   n_leaves() const;
    float feature_importance(int feature_idx) const;

    const std::vector<TreeNode>& nodes() const { return nodes_; }
    int n_classes() const { return n_classes_; }

private:
    TreeParams params_;
    std::vector<TreeNode> nodes_;
    std::vector<double> importance_accum_;  // unnormalized impurity-decrease sum per feature
    int n_features_ = 0;
    int n_classes_ = 0;
    mutable std::mt19937 rng_;

    // Grows the subtree rooted at `indices` (a subset of X/y's rows),
    // returns the index of the created node in nodes_. Recursive --
    // depth is bounded by max_depth, so stack depth is bounded too.
    int build_node(const Features& X, const Labels& y, std::vector<int> indices, int depth);

    double node_impurity(const std::vector<int>& class_counts, int total) const;

    // Scans every feature's sorted values for the best (feature,
    // threshold) split by impurity decrease. Returns false if no split
    // improves on the parent's impurity by at least min_impurity_decrease
    // (the node becomes a leaf).
    bool find_best_split(const Features& X, const Labels& y, const std::vector<int>& indices,
                          double parent_impurity, int& best_feature, float& best_threshold,
                          double& best_decrease, std::vector<int>& left_out, std::vector<int>& right_out) const;

    int predict_one(const std::vector<float>& row) const;
};
