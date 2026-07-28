#include "gbt.h"

#include <algorithm>
#include <cmath>
#include <numeric>

GradientBoostedTrees::GradientBoostedTrees(GBTParams params) : params_(params), rng_(params.random_state) {}

void GradientBoostedTrees::fit(const Features& X, const Labels& y) {
    int n = static_cast<int>(X.size());
    n_features_ = n > 0 ? static_cast<int>(X[0].size()) : 0;

    double pos = 0.0;
    for (float v : y) pos += static_cast<double>(v);
    double p0 = std::clamp(pos / n, 1e-6, 1.0 - 1e-6);
    base_score_ = static_cast<float>(std::log(p0 / (1.0 - p0)));

    std::vector<float> F(static_cast<std::size_t>(n), base_score_);

    trees_.clear();
    trees_.reserve(static_cast<std::size_t>(params_.n_estimators));

    int n_feat_sub = std::max(1, static_cast<int>(std::round(static_cast<double>(params_.colsample) * n_features_)));
    int n_row_sub = std::max(1, static_cast<int>(std::round(static_cast<double>(params_.subsample) * n)));

    std::vector<int> all_indices(static_cast<std::size_t>(n));
    std::iota(all_indices.begin(), all_indices.end(), 0);
    std::vector<int> all_features(static_cast<std::size_t>(n_features_));
    std::iota(all_features.begin(), all_features.end(), 0);

    for (int t = 0; t < params_.n_estimators; ++t) {
        // Gradient/Hessian of binary logistic loss w.r.t. the running
        // raw score F -- the (g, h) pairs this round's tree fits to.
        std::vector<float> g(static_cast<std::size_t>(n)), h(static_cast<std::size_t>(n));
        for (int i = 0; i < n; ++i) {
            double p = 1.0 / (1.0 + std::exp(-static_cast<double>(F[static_cast<std::size_t>(i)])));
            g[static_cast<std::size_t>(i)] = static_cast<float>(p - static_cast<double>(y[static_cast<std::size_t>(i)]));
            h[static_cast<std::size_t>(i)] = static_cast<float>(p * (1.0 - p));
        }

        // Row subsample without replacement (Friedman 2002 stochastic
        // gradient boosting) -- a fresh draw every round.
        std::vector<int> row_indices = all_indices;
        std::shuffle(row_indices.begin(), row_indices.end(), rng_);
        row_indices.resize(static_cast<std::size_t>(n_row_sub));

        // Column subsample, fixed for the whole tree (XGBoost-style
        // colsample_bytree) -- deliberately different from
        // random_forest's per-split-node resampling.
        std::vector<int> feature_subset = all_features;
        std::shuffle(feature_subset.begin(), feature_subset.end(), rng_);
        feature_subset.resize(static_cast<std::size_t>(n_feat_sub));

        GBTreeParams tp;
        tp.max_depth = params_.max_depth;
        tp.l1_reg = params_.l1_reg;
        tp.l2_reg = params_.l2_reg;

        GBRegressionTree tree(tp);
        tree.fit(X, g, h, row_indices, feature_subset);

        for (int i = 0; i < n; ++i)
            F[static_cast<std::size_t>(i)] +=
                params_.learning_rate * tree.predict_one(X[static_cast<std::size_t>(i)]);

        trees_.push_back(std::move(tree));
    }
}

float GradientBoostedTrees::raw_score(const std::vector<float>& row, int n_trees) const {
    float score = base_score_;
    int limit = std::min(n_trees, static_cast<int>(trees_.size()));
    for (int t = 0; t < limit; ++t)
        score += params_.learning_rate * trees_[static_cast<std::size_t>(t)].predict_one(row);
    return score;
}

std::vector<float> GradientBoostedTrees::predict_proba(const Features& X) const {
    std::vector<float> out;
    out.reserve(X.size());
    for (const auto& row : X) {
        float raw = raw_score(row, static_cast<int>(trees_.size()));
        out.push_back(static_cast<float>(1.0 / (1.0 + std::exp(-static_cast<double>(raw)))));
    }
    return out;
}

std::vector<float> GradientBoostedTrees::staged_predict(const Features& X, int k) const {
    std::vector<float> out;
    out.reserve(X.size());
    for (const auto& row : X) {
        float raw = raw_score(row, k);
        out.push_back(static_cast<float>(1.0 / (1.0 + std::exp(-static_cast<double>(raw)))));
    }
    return out;
}

Labels GradientBoostedTrees::predict(const Features& X) const {
    Labels out;
    out.reserve(X.size());
    for (float p : predict_proba(X)) out.push_back(p >= 0.5f ? 1.0f : 0.0f);
    return out;
}

float GradientBoostedTrees::score(const Features& X, const Labels& y) const {
    if (y.empty()) return 0.0f;
    Labels pred = predict(X);
    int correct = 0;
    for (std::size_t i = 0; i < y.size(); ++i)
        if (pred[i] == y[i]) ++correct;
    return static_cast<float>(correct) / static_cast<float>(y.size());
}
