// ensemble_bench.cpp — PLAN.md Phase 12b step 13: real diverse-vs-
// correlated stacking experiment on real OpenML data (reusing
// ml/openml_bench's committed datasets + loader). Run manually (same
// convention as openml_bench/hyperparam_sweep -- see README.md's Design
// note); results captured into README.md.
#include "arff_loader.h"
#include "decision_tree.h"
#include "ensemble.h"
#include "knn.h"
#include "linear_model.h"

#include <cmath>
#include <cstdio>
#include <string>

namespace {

void standardize(TrainTestSplit& split) {
    std::size_t d = split.X_train[0].size();
    std::vector<float> mean(d, 0.0f), stddev(d, 0.0f);
    for (const auto& row : split.X_train)
        for (std::size_t f = 0; f < d; ++f) mean[f] += row[f];
    for (std::size_t f = 0; f < d; ++f) mean[f] /= static_cast<float>(split.X_train.size());
    for (const auto& row : split.X_train)
        for (std::size_t f = 0; f < d; ++f) stddev[f] += (row[f] - mean[f]) * (row[f] - mean[f]);
    for (std::size_t f = 0; f < d; ++f) stddev[f] = std::sqrt(stddev[f] / static_cast<float>(split.X_train.size()));
    auto apply = [&](Features& X) {
        for (auto& row : X)
            for (std::size_t f = 0; f < d; ++f)
                if (stddev[f] > 1e-8f) row[f] = (row[f] - mean[f]) / stddev[f];
    };
    apply(split.X_train);
    apply(split.X_test);
}

float accuracy(const Labels& pred, const Labels& truth) {
    int correct = 0;
    for (std::size_t i = 0; i < truth.size(); ++i)
        if (pred[i] == truth[i]) ++correct;
    return static_cast<float>(correct) / static_cast<float>(truth.size());
}

FitPredictFn decision_tree_fn(int max_depth) {
    return [max_depth](const Features& X_train, const Labels& y_train, const Features& X_query) {
        TreeParams p;
        p.max_depth = max_depth;
        DecisionTree model(p);
        model.fit(X_train, y_train);
        return model.predict(X_query);
    };
}

FitPredictFn knn_fn(int k) {
    return [k](const Features& X_train, const Labels& y_train, const Features& X_query) {
        KNNParams p;
        p.k = k;
        KNNClassifier model(p);
        model.fit(X_train, y_train);
        return model.predict(X_query);
    };
}

FitPredictFn linear_model_fn() {
    return [](const Features& X_train, const Labels& y_train, const Features& X_query) {
        LinearModelParams p;
        p.loss = LinearLoss::LOGISTIC;
        p.optimizer = LinearOptimizer::LBFGS;
        p.alpha = 1e-3f;
        p.max_iter = 100;
        LinearModel model(p);
        model.fit(X_train, y_train);
        return model.predict(X_query);
    };
}

void run_experiment(const std::string& file, const std::string& name) {
    std::printf("\n=== %s ===\n", name.c_str());
    OpenMLDataset d = load_arff(std::string(OPENML_DATA_DIR) + "/" + file, name);
    TrainTestSplit split = split_train_test(d.X, d.y, 0.2f, 42);
    standardize(split);

    struct Ensemble {
        const char* label;
        std::vector<FitPredictFn> models;
        std::vector<const char*> member_labels;
    };
    std::vector<Ensemble> ensembles = {
        {"diverse (DecisionTree + KNN + LogisticRegression)",
         {decision_tree_fn(10), knn_fn(5), linear_model_fn()},
         {"DecisionTree(depth=10)", "KNN(k=5)", "LogisticRegression"}},
        {"correlated (three DecisionTrees, varying depth)",
         {decision_tree_fn(8), decision_tree_fn(10), decision_tree_fn(12)},
         {"DecisionTree(depth=8)", "DecisionTree(depth=10)", "DecisionTree(depth=12)"}},
    };

    for (const auto& ens : ensembles) {
        std::printf("  %s:\n", ens.label);
        float best_individual = 0.0f;
        for (std::size_t m = 0; m < ens.models.size(); ++m) {
            Labels pred = ens.models[m](split.X_train, split.y_train, split.X_test);
            float acc = accuracy(pred, split.y_test);
            best_individual = std::max(best_individual, acc);
            std::printf("    %-28s test_acc=%.4f\n", ens.member_labels[m], static_cast<double>(acc));
        }

        Labels vote_pred = majority_vote(split.X_train, split.y_train, split.X_test, ens.models);
        float vote_acc = accuracy(vote_pred, split.y_test);
        std::printf("    %-28s test_acc=%.4f (vs best individual %.4f)\n", "majority_vote", static_cast<double>(vote_acc),
                    static_cast<double>(best_individual));

        Features meta_train = stacking_oof_features(split.X_train, split.y_train, ens.models, 5, 42);
        LinearModelParams meta_p;
        meta_p.loss = LinearLoss::LOGISTIC;
        meta_p.optimizer = LinearOptimizer::LBFGS;
        meta_p.alpha = 1e-3f;
        meta_p.max_iter = 100;
        LinearModel meta_model(meta_p);
        meta_model.fit(meta_train, split.y_train);

        Features meta_test = stacking_test_features(split.X_train, split.y_train, split.X_test, ens.models);
        Labels stack_pred = meta_model.predict(meta_test);
        float stack_acc = accuracy(stack_pred, split.y_test);
        std::printf("    %-28s test_acc=%.4f (vs best individual %.4f)\n", "stacking(meta=LogReg)", static_cast<double>(stack_acc),
                    static_cast<double>(best_individual));
    }
}

}  // namespace

int main() {
    run_experiment("blood-transfusion.arff", "blood-transfusion");
    run_experiment("breast-w.arff", "breast-w");
    return 0;
}
