#include "random_forest.h"
#include "ws_pool/ws_pool.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>

RandomForest::RandomForest(RFParams params) : params_(params) {}

void RandomForest::fit(const Features& X, const Labels& y) {
    X_ = X;
    y_ = y;
    int n = static_cast<int>(X.size());
    n_features_ = n > 0 ? static_cast<int>(X[0].size()) : 0;
    int max_label = 0;
    for (float v : y) max_label = std::max(max_label, static_cast<int>(v));
    n_classes_ = max_label + 1;

    // max_features resolution: >=1 absolute count, (0,1) fraction, -1 ->
    // sqrt(n_features) -- the classification default PLAN.md names.
    int mtry;
    if (params_.max_features < 0.0f) {
        mtry = std::max(1, static_cast<int>(std::round(std::sqrt(static_cast<double>(n_features_)))));
    } else if (params_.max_features < 1.0f) {
        mtry = std::max(1, static_cast<int>(std::round(params_.max_features * static_cast<float>(n_features_))));
    } else {
        mtry = std::max(1, static_cast<int>(std::round(params_.max_features)));
    }
    mtry = std::min(mtry, n_features_);

    trees_.assign(static_cast<std::size_t>(params_.n_estimators), DecisionTree{});
    in_bag_.assign(static_cast<std::size_t>(params_.n_estimators), std::vector<char>(static_cast<std::size_t>(n), 0));

    foundation::WorkStealingPool pool;
    pool.parallel_for(static_cast<std::size_t>(params_.n_estimators), [&](std::size_t t) {
        std::mt19937 rng(static_cast<unsigned>(params_.random_state) + static_cast<unsigned>(t));
        std::uniform_int_distribution<int> sample_dist(0, n - 1);

        Features Xt;
        Labels yt;
        Xt.reserve(static_cast<std::size_t>(n));
        yt.reserve(static_cast<std::size_t>(n));
        for (int s = 0; s < n; ++s) {
            int idx = sample_dist(rng);
            Xt.push_back(X[static_cast<std::size_t>(idx)]);
            yt.push_back(y[static_cast<std::size_t>(idx)]);
            in_bag_[t][static_cast<std::size_t>(idx)] = 1;
        }

        TreeParams tp;
        tp.max_depth = params_.max_depth;
        tp.min_samples_leaf = params_.min_samples_leaf;
        tp.max_features = mtry;
        // Distinct seed per tree, offset well clear of the bootstrap
        // sampler's own RNG stream above so the two don't correlate.
        tp.random_state = static_cast<unsigned>(params_.random_state) + static_cast<unsigned>(t) + 1000000u;
        trees_[t] = DecisionTree(tp);
        trees_[t].fit(Xt, yt);
    });

    std::vector<int> identity(static_cast<std::size_t>(n));
    std::iota(identity.begin(), identity.end(), 0);
    oob_error_ = 1.0f - oob_accuracy_with_permutation(-1, identity);
}

float RandomForest::oob_accuracy_with_permutation(int permute_feature, const std::vector<int>& permutation) const {
    int n = static_cast<int>(X_.size());
    int correct = 0, total = 0;
    for (int i = 0; i < n; ++i) {
        std::vector<int> votes(static_cast<std::size_t>(n_classes_), 0);
        bool any_oob = false;
        for (std::size_t t = 0; t < trees_.size(); ++t) {
            if (in_bag_[t][static_cast<std::size_t>(i)]) continue;  // only OOB trees vote
            any_oob = true;
            std::vector<float> row = X_[static_cast<std::size_t>(i)];
            if (permute_feature >= 0)
                row[static_cast<std::size_t>(permute_feature)] =
                    X_[static_cast<std::size_t>(permutation[static_cast<std::size_t>(i)])][static_cast<std::size_t>(permute_feature)];
            Labels pred = trees_[t].predict(Features{row});
            votes[static_cast<std::size_t>(pred[0])]++;
        }
        if (!any_oob) continue;
        int predicted = static_cast<int>(std::max_element(votes.begin(), votes.end()) - votes.begin());
        ++total;
        if (predicted == static_cast<int>(y_[static_cast<std::size_t>(i)])) ++correct;
    }
    return total > 0 ? static_cast<float>(correct) / static_cast<float>(total) : 0.0f;
}

std::vector<float> RandomForest::feature_importances() const {
    std::vector<int> identity(X_.size());
    std::iota(identity.begin(), identity.end(), 0);
    float baseline = oob_accuracy_with_permutation(-1, identity);

    std::vector<float> importances(static_cast<std::size_t>(n_features_), 0.0f);
    std::mt19937 rng(12345);
    for (int f = 0; f < n_features_; ++f) {
        std::vector<int> perm = identity;
        std::shuffle(perm.begin(), perm.end(), rng);
        float permuted_acc = oob_accuracy_with_permutation(f, perm);
        importances[static_cast<std::size_t>(f)] = baseline - permuted_acc;
    }
    return importances;
}

Labels RandomForest::predict(const Features& X) const {
    Labels out;
    out.reserve(X.size());
    for (const auto& row : X) {
        std::vector<int> votes(static_cast<std::size_t>(n_classes_), 0);
        for (const auto& tree : trees_) {
            Labels pred = tree.predict(Features{row});
            votes[static_cast<std::size_t>(pred[0])]++;
        }
        out.push_back(static_cast<float>(std::max_element(votes.begin(), votes.end()) - votes.begin()));
    }
    return out;
}

float RandomForest::score(const Features& X, const Labels& y) const {
    if (y.empty()) return 0.0f;
    Labels pred = predict(X);
    int correct = 0;
    for (std::size_t i = 0; i < y.size(); ++i)
        if (pred[i] == y[i]) ++correct;
    return static_cast<float>(correct) / static_cast<float>(y.size());
}
