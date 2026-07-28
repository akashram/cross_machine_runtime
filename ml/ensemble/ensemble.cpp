#include "ensemble.h"

#include <algorithm>
#include <numeric>
#include <random>

std::vector<int> k_fold_assignment(std::size_t n, int k_folds, unsigned random_state) {
    std::vector<std::size_t> indices(n);
    std::iota(indices.begin(), indices.end(), 0);
    std::mt19937 rng(random_state);
    std::shuffle(indices.begin(), indices.end(), rng);

    std::vector<int> fold(n);
    for (std::size_t i = 0; i < n; ++i) fold[indices[i]] = static_cast<int>(i % static_cast<std::size_t>(k_folds));
    return fold;
}

Features stacking_oof_features(const Features& X, const Labels& y, const std::vector<FitPredictFn>& base_models, int k_folds,
                                unsigned random_state) {
    std::size_t n = X.size();
    std::vector<int> fold = k_fold_assignment(n, k_folds, random_state);
    Features meta(n, std::vector<float>(base_models.size(), 0.0f));

    for (std::size_t m = 0; m < base_models.size(); ++m) {
        for (int f = 0; f < k_folds; ++f) {
            Features X_train_fold, X_val_fold;
            Labels y_train_fold;
            std::vector<std::size_t> val_indices;
            for (std::size_t i = 0; i < n; ++i) {
                if (fold[i] == f) {
                    X_val_fold.push_back(X[i]);
                    val_indices.push_back(i);
                } else {
                    X_train_fold.push_back(X[i]);
                    y_train_fold.push_back(y[i]);
                }
            }
            Labels preds = base_models[m](X_train_fold, y_train_fold, X_val_fold);
            for (std::size_t j = 0; j < val_indices.size(); ++j) meta[val_indices[j]][m] = preds[j];
        }
    }
    return meta;
}

Features stacking_test_features(const Features& X_train, const Labels& y_train, const Features& X_test,
                                 const std::vector<FitPredictFn>& base_models) {
    Features meta(X_test.size(), std::vector<float>(base_models.size(), 0.0f));
    for (std::size_t m = 0; m < base_models.size(); ++m) {
        Labels preds = base_models[m](X_train, y_train, X_test);
        for (std::size_t i = 0; i < X_test.size(); ++i) meta[i][m] = preds[i];
    }
    return meta;
}

Labels majority_vote(const Features& X_train, const Labels& y_train, const Features& X_test, const std::vector<FitPredictFn>& base_models) {
    Features meta = stacking_test_features(X_train, y_train, X_test, base_models);
    Labels out(X_test.size());
    for (std::size_t i = 0; i < X_test.size(); ++i) {
        float votes = 0.0f;
        for (std::size_t m = 0; m < base_models.size(); ++m) votes += meta[i][m];
        out[i] = votes > static_cast<float>(base_models.size()) / 2.0f ? 1.0f : 0.0f;
    }
    return out;
}
