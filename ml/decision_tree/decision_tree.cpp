#include "decision_tree.h"

#include <algorithm>
#include <cmath>
#include <numeric>

DecisionTree::DecisionTree(TreeParams params) : params_(params) {}

double DecisionTree::node_impurity(const std::vector<int>& class_counts, int total) const {
    if (total == 0) return 0.0;
    double impurity = 0.0;
    if (params_.criterion == TreeParams::Criterion::GINI) {
        double sum_sq = 0.0;
        for (int c : class_counts) {
            double p = static_cast<double>(c) / total;
            sum_sq += p * p;
        }
        impurity = 1.0 - sum_sq;
    } else {
        for (int c : class_counts) {
            if (c == 0) continue;
            double p = static_cast<double>(c) / total;
            impurity -= p * std::log2(p);
        }
    }
    return impurity;
}

bool DecisionTree::find_best_split(const Features& X, const Labels& y, const std::vector<int>& indices,
                                    double parent_impurity, int& best_feature, float& best_threshold,
                                    double& best_decrease, std::vector<int>& left_out,
                                    std::vector<int>& right_out) const {
    const int n = static_cast<int>(indices.size());
    best_decrease = 0.0;
    best_feature = -1;

    std::vector<int> total_counts(static_cast<std::size_t>(n_classes_), 0);
    for (int idx : indices) total_counts[static_cast<std::size_t>(y[static_cast<std::size_t>(idx)])]++;

    // (feature_value, class) pairs, re-used per feature to avoid
    // reallocating -- the sort is the O(n log n) cost per feature; the
    // subsequent left-to-right sweep accumulating class-count histograms
    // is a single linear pass over contiguous arrays, the "vectorized
    // split search" shape PLAN.md asks for (auto-vectorizable at -O3 —
    // no hand-written AVX intrinsics here; see README.md's Design note).
    std::vector<std::pair<float, int>> sorted;
    sorted.reserve(static_cast<std::size_t>(n));

    for (int feature = 0; feature < n_features_; ++feature) {
        sorted.clear();
        for (int idx : indices)
            sorted.emplace_back(X[static_cast<std::size_t>(idx)][static_cast<std::size_t>(feature)],
                                 static_cast<int>(y[static_cast<std::size_t>(idx)]));
        std::sort(sorted.begin(), sorted.end());

        std::vector<int> left_counts(static_cast<std::size_t>(n_classes_), 0);
        int left_total = 0;
        for (int i = 0; i < n - 1; ++i) {
            left_counts[static_cast<std::size_t>(sorted[static_cast<std::size_t>(i)].second)]++;
            ++left_total;
            // Only a valid split point where the feature value actually
            // changes -- splitting between two samples with the same
            // value can't separate them.
            if (sorted[static_cast<std::size_t>(i)].first == sorted[static_cast<std::size_t>(i + 1)].first) continue;
            int right_total = n - left_total;
            if (left_total < params_.min_samples_leaf || right_total < params_.min_samples_leaf) continue;

            std::vector<int> right_counts(static_cast<std::size_t>(n_classes_));
            for (int c = 0; c < n_classes_; ++c)
                right_counts[static_cast<std::size_t>(c)] = total_counts[static_cast<std::size_t>(c)] - left_counts[static_cast<std::size_t>(c)];

            double left_imp = node_impurity(left_counts, left_total);
            double right_imp = node_impurity(right_counts, right_total);
            double weighted = (static_cast<double>(left_total) * left_imp + static_cast<double>(right_total) * right_imp) / n;
            double decrease = parent_impurity - weighted;

            if (decrease > best_decrease) {
                best_decrease = decrease;
                best_feature = feature;
                best_threshold = (sorted[static_cast<std::size_t>(i)].first + sorted[static_cast<std::size_t>(i + 1)].first) / 2.0f;
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

int DecisionTree::build_node(const Features& X, const Labels& y, std::vector<int> indices, int depth) {
    std::vector<int> class_counts(static_cast<std::size_t>(n_classes_), 0);
    for (int idx : indices) class_counts[static_cast<std::size_t>(y[static_cast<std::size_t>(idx)])]++;
    int n = static_cast<int>(indices.size());
    double impurity = node_impurity(class_counts, n);

    int majority_class = static_cast<int>(std::max_element(class_counts.begin(), class_counts.end()) - class_counts.begin());

    TreeNode node;
    node.is_leaf = true;
    node.predicted_class = majority_class;
    node.n_samples = n;
    node.impurity = static_cast<float>(impurity);

    bool can_split = (params_.max_depth < 0 || depth < params_.max_depth) &&
                      n >= params_.min_samples_split && impurity > 0.0;

    int best_feature = -1;
    float best_threshold = 0.0f;
    double best_decrease = 0.0;
    std::vector<int> left_indices, right_indices;

    if (can_split) {
        can_split = find_best_split(X, y, indices, impurity, best_feature, best_threshold, best_decrease,
                                     left_indices, right_indices) &&
                    best_decrease >= static_cast<double>(params_.min_impurity_decrease);
    }

    int this_idx = static_cast<int>(nodes_.size());
    nodes_.push_back(node);  // placeholder; filled in below if this becomes an internal node

    if (!can_split) return this_idx;

    importance_accum_[static_cast<std::size_t>(best_feature)] += static_cast<double>(n) * best_decrease;

    int left_child = build_node(X, y, std::move(left_indices), depth + 1);
    int right_child = build_node(X, y, std::move(right_indices), depth + 1);

    nodes_[static_cast<std::size_t>(this_idx)].is_leaf = false;
    nodes_[static_cast<std::size_t>(this_idx)].feature_idx = best_feature;
    nodes_[static_cast<std::size_t>(this_idx)].threshold = best_threshold;
    nodes_[static_cast<std::size_t>(this_idx)].left = left_child;
    nodes_[static_cast<std::size_t>(this_idx)].right = right_child;
    return this_idx;
}

void DecisionTree::fit(const Features& X, const Labels& y) {
    nodes_.clear();
    n_features_ = X.empty() ? 0 : static_cast<int>(X[0].size());
    int max_label = 0;
    for (float v : y) max_label = std::max(max_label, static_cast<int>(v));
    n_classes_ = max_label + 1;
    importance_accum_.assign(static_cast<std::size_t>(n_features_), 0.0);

    std::vector<int> indices(y.size());
    std::iota(indices.begin(), indices.end(), 0);
    build_node(X, y, std::move(indices), 0);
}

int DecisionTree::predict_one(const std::vector<float>& row) const {
    int node_idx = 0;
    while (!nodes_[static_cast<std::size_t>(node_idx)].is_leaf) {
        const TreeNode& node = nodes_[static_cast<std::size_t>(node_idx)];
        node_idx = (row[static_cast<std::size_t>(node.feature_idx)] <= node.threshold) ? node.left : node.right;
    }
    return nodes_[static_cast<std::size_t>(node_idx)].predicted_class;
}

Labels DecisionTree::predict(const Features& X) const {
    Labels out;
    out.reserve(X.size());
    for (const auto& row : X) out.push_back(static_cast<float>(predict_one(row)));
    return out;
}

float DecisionTree::score(const Features& X, const Labels& y) const {
    if (y.empty()) return 0.0f;
    Labels pred = predict(X);
    int correct = 0;
    for (std::size_t i = 0; i < y.size(); ++i)
        if (pred[i] == y[i]) ++correct;
    return static_cast<float>(correct) / static_cast<float>(y.size());
}

int DecisionTree::depth() const {
    // Iterative BFS/DFS over the flat node array -- avoids a second
    // recursive walk mirroring build_node's recursion.
    std::vector<int> stack{0};
    std::vector<int> node_depth(nodes_.size(), 0);
    int max_depth = 0;
    while (!stack.empty()) {
        int idx = stack.back();
        stack.pop_back();
        max_depth = std::max(max_depth, node_depth[static_cast<std::size_t>(idx)]);
        const TreeNode& node = nodes_[static_cast<std::size_t>(idx)];
        if (!node.is_leaf) {
            node_depth[static_cast<std::size_t>(node.left)] = node_depth[static_cast<std::size_t>(idx)] + 1;
            node_depth[static_cast<std::size_t>(node.right)] = node_depth[static_cast<std::size_t>(idx)] + 1;
            stack.push_back(node.left);
            stack.push_back(node.right);
        }
    }
    return max_depth;
}

int DecisionTree::n_leaves() const {
    int count = 0;
    for (const auto& node : nodes_)
        if (node.is_leaf) ++count;
    return count;
}

float DecisionTree::feature_importance(int feature_idx) const {
    double total = 0.0;
    for (double v : importance_accum_) total += v;
    if (total <= 0.0) return 0.0f;
    return static_cast<float>(importance_accum_[static_cast<std::size_t>(feature_idx)] / total);
}
