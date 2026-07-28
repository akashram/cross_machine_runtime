#include "regression_tree.h"

#include <algorithm>
#include <cmath>
#include <numeric>

namespace {

// L1 (alpha) soft-thresholding on a gradient sum -- the standard
// XGBoost-style closed-form Newton leaf weight under an L1 penalty:
// zero region of width 2*alpha around G=0, linear shrinkage outside it.
double softthresh(double g_sum, double alpha) {
    if (g_sum > alpha) return g_sum - alpha;
    if (g_sum < -alpha) return g_sum + alpha;
    return 0.0;
}

}  // namespace

GBRegressionTree::GBRegressionTree(GBTreeParams params) : params_(params) {}

bool GBRegressionTree::find_best_split(const Features& X, const std::vector<float>& g,
                                        const std::vector<float>& h, const std::vector<int>& feature_subset,
                                        const std::vector<int>& indices, double G, double H, int& best_feature,
                                        float& best_threshold, double& best_gain, std::vector<int>& left_out,
                                        std::vector<int>& right_out) const {
    const int n = static_cast<int>(indices.size());
    best_gain = static_cast<double>(params_.min_split_gain);
    best_feature = -1;

    const double parent_term = std::pow(softthresh(G, static_cast<double>(params_.l1_reg)), 2) /
                                (H + static_cast<double>(params_.l2_reg));

    // (feature_value, gradient, hessian) triples, re-used per feature to
    // avoid reallocating -- same sort-then-linear-sweep shape as
    // decision_tree's find_best_split, generalized from class-count
    // histograms to running gradient/Hessian sums.
    std::vector<std::tuple<float, float, float>> sorted;
    sorted.reserve(static_cast<std::size_t>(n));

    for (int feature : feature_subset) {
        sorted.clear();
        for (int idx : indices)
            sorted.emplace_back(X[static_cast<std::size_t>(idx)][static_cast<std::size_t>(feature)],
                                 g[static_cast<std::size_t>(idx)], h[static_cast<std::size_t>(idx)]);
        std::sort(sorted.begin(), sorted.end(),
                  [](const auto& a, const auto& b) { return std::get<0>(a) < std::get<0>(b); });

        double left_G = 0.0, left_H = 0.0;
        int left_total = 0;
        for (int i = 0; i < n - 1; ++i) {
            left_G += static_cast<double>(std::get<1>(sorted[static_cast<std::size_t>(i)]));
            left_H += static_cast<double>(std::get<2>(sorted[static_cast<std::size_t>(i)]));
            ++left_total;
            if (std::get<0>(sorted[static_cast<std::size_t>(i)]) == std::get<0>(sorted[static_cast<std::size_t>(i + 1)]))
                continue;
            int right_total = n - left_total;
            if (left_total < params_.min_samples_leaf || right_total < params_.min_samples_leaf) continue;

            double right_G = G - left_G;
            double right_H = H - left_H;

            double left_term = std::pow(softthresh(left_G, static_cast<double>(params_.l1_reg)), 2) /
                                (left_H + static_cast<double>(params_.l2_reg));
            double right_term = std::pow(softthresh(right_G, static_cast<double>(params_.l1_reg)), 2) /
                                 (right_H + static_cast<double>(params_.l2_reg));
            double gain = 0.5 * (left_term + right_term - parent_term);

            if (gain > best_gain) {
                best_gain = gain;
                best_feature = feature;
                best_threshold = (std::get<0>(sorted[static_cast<std::size_t>(i)]) +
                                   std::get<0>(sorted[static_cast<std::size_t>(i + 1)])) /
                                  2.0f;
            }
        }
    }

    if (best_feature < 0) return false;

    left_out.clear();
    right_out.clear();
    for (int idx : indices) {
        if (X[static_cast<std::size_t>(idx)][static_cast<std::size_t>(best_feature)] <= best_threshold)
            left_out.push_back(idx);
        else
            right_out.push_back(idx);
    }
    return true;
}

int GBRegressionTree::build_node(const Features& X, const std::vector<float>& g, const std::vector<float>& h,
                                  const std::vector<int>& feature_subset, std::vector<int> indices, int depth) {
    double G = 0.0, H = 0.0;
    for (int idx : indices) {
        G += static_cast<double>(g[static_cast<std::size_t>(idx)]);
        H += static_cast<double>(h[static_cast<std::size_t>(idx)]);
    }
    int n = static_cast<int>(indices.size());

    GBTreeNode node;
    node.is_leaf = true;
    node.leaf_value =
        static_cast<float>(-softthresh(G, static_cast<double>(params_.l1_reg)) / (H + static_cast<double>(params_.l2_reg)));

    bool can_split = (params_.max_depth < 0 || depth < params_.max_depth) && n >= params_.min_samples_split;

    int best_feature = -1;
    float best_threshold = 0.0f;
    double best_gain = 0.0;
    std::vector<int> left_indices, right_indices;

    if (can_split) {
        can_split = find_best_split(X, g, h, feature_subset, indices, G, H, best_feature, best_threshold,
                                     best_gain, left_indices, right_indices);
    }

    int this_idx = static_cast<int>(nodes_.size());
    nodes_.push_back(node);

    if (!can_split) return this_idx;

    int left_child = build_node(X, g, h, feature_subset, std::move(left_indices), depth + 1);
    int right_child = build_node(X, g, h, feature_subset, std::move(right_indices), depth + 1);

    nodes_[static_cast<std::size_t>(this_idx)].is_leaf = false;
    nodes_[static_cast<std::size_t>(this_idx)].feature_idx = best_feature;
    nodes_[static_cast<std::size_t>(this_idx)].threshold = best_threshold;
    nodes_[static_cast<std::size_t>(this_idx)].left = left_child;
    nodes_[static_cast<std::size_t>(this_idx)].right = right_child;
    return this_idx;
}

void GBRegressionTree::fit(const Features& X, const std::vector<float>& gradients,
                            const std::vector<float>& hessians, const std::vector<int>& indices,
                            const std::vector<int>& feature_subset) {
    nodes_.clear();
    n_features_ = X.empty() ? 0 : static_cast<int>(X[0].size());

    std::vector<int> all_features;
    const std::vector<int>* features = &feature_subset;
    if (feature_subset.empty()) {
        all_features.resize(static_cast<std::size_t>(n_features_));
        std::iota(all_features.begin(), all_features.end(), 0);
        features = &all_features;
    }

    build_node(X, gradients, hessians, *features, indices, 0);
}

float GBRegressionTree::predict_one(const std::vector<float>& row) const {
    int node_idx = 0;
    while (!nodes_[static_cast<std::size_t>(node_idx)].is_leaf) {
        const GBTreeNode& node = nodes_[static_cast<std::size_t>(node_idx)];
        node_idx = (row[static_cast<std::size_t>(node.feature_idx)] <= node.threshold) ? node.left : node.right;
    }
    return nodes_[static_cast<std::size_t>(node_idx)].leaf_value;
}
