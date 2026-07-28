// hyperparam_sweep.cpp — PLAN.md Phase 12b step 12: for each Phase 12a
// algorithm, sweep a key hyperparameter on a real OpenML dataset (reusing
// ml/openml_bench's committed data + loader) and record accuracy vs.
// parameter. Run manually (same convention as openml_bench: not a
// ctest target -- see CMakeLists.txt); results + mechanism explanations
// captured into README.md.
#include "arff_loader.h"
#include "decision_tree.h"
#include "random_forest.h"
#include "gbt.h"
#include "svm.h"
#include "knn.h"
#include "kmeans.h"
#include "pca.h"
#include "linear_model.h"

#include <cmath>
#include <cstdio>
#include <map>
#include <vector>

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

std::string data_path(const std::string& file) { return std::string(OPENML_DATA_DIR) + "/" + file; }

void sweep_decision_tree_max_depth() {
    std::printf("\n=== DecisionTree: max_depth (diabetes) ===\n");
    std::printf("%-10s %12s %12s %10s\n", "max_depth", "train_acc", "test_acc", "gap");
    OpenMLDataset d = load_arff(data_path("diabetes.arff"), "diabetes");
    TrainTestSplit split = split_train_test(d.X, d.y, 0.2f, 42);
    for (int depth : {1, 2, 3, 5, 8, 12, 20}) {
        TreeParams p;
        p.max_depth = depth;
        DecisionTree model(p);
        model.fit(split.X_train, split.y_train);
        float train_acc = model.score(split.X_train, split.y_train);
        float test_acc = model.score(split.X_test, split.y_test);
        std::printf("%-10d %12.4f %12.4f %10.4f\n", depth, static_cast<double>(train_acc), static_cast<double>(test_acc),
                    static_cast<double>(train_acc - test_acc));
    }
}

void sweep_random_forest_n_estimators() {
    std::printf("\n=== RandomForest: n_estimators (vehicle) ===\n");
    std::printf("%-12s %12s\n", "n_estimators", "test_acc");
    OpenMLDataset d = load_arff(data_path("vehicle.arff"), "vehicle");
    TrainTestSplit split = split_train_test(d.X, d.y, 0.2f, 42);
    for (int n : {1, 5, 10, 25, 50, 100, 200}) {
        RFParams p;
        p.n_estimators = n;
        RandomForest model(p);
        model.fit(split.X_train, split.y_train);
        float test_acc = model.score(split.X_test, split.y_test);
        std::printf("%-12d %12.4f\n", n, static_cast<double>(test_acc));
    }
}

void sweep_gbt_n_estimators() {
    std::printf("\n=== GBT: n_estimators (blood-transfusion) ===\n");
    std::printf("%-12s %12s %12s\n", "n_estimators", "train_acc", "test_acc");
    OpenMLDataset d = load_arff(data_path("blood-transfusion.arff"), "blood-transfusion");
    TrainTestSplit split = split_train_test(d.X, d.y, 0.2f, 42);
    for (int n : {5, 10, 25, 50, 100, 200, 400}) {
        GBTParams p;
        p.n_estimators = n;
        GradientBoostedTrees model(p);
        model.fit(split.X_train, split.y_train);
        float train_acc = model.score(split.X_train, split.y_train);
        float test_acc = model.score(split.X_test, split.y_test);
        std::printf("%-12d %12.4f %12.4f\n", n, static_cast<double>(train_acc), static_cast<double>(test_acc));
    }
}

void sweep_svm_gamma() {
    std::printf("\n=== SVM: gamma (wdbc, RBF kernel, standardized) ===\n");
    std::printf("%-10s %12s %12s\n", "gamma", "train_acc", "test_acc");
    OpenMLDataset d = load_arff(data_path("wdbc.arff"), "wdbc");
    TrainTestSplit split = split_train_test(d.X, d.y, 0.2f, 42);
    standardize(split);
    Labels y_train(split.y_train.size()), y_test(split.y_test.size());
    for (std::size_t i = 0; i < split.y_train.size(); ++i) y_train[i] = split.y_train[i] == 0.0f ? -1.0f : 1.0f;
    for (std::size_t i = 0; i < split.y_test.size(); ++i) y_test[i] = split.y_test[i] == 0.0f ? -1.0f : 1.0f;
    for (float gamma : {0.001f, 0.01f, 0.1f, 1.0f, 10.0f}) {
        SVMParams p;
        p.kernel = KernelType::RBF;
        p.gamma = gamma;
        p.max_iter = 200;
        SVM model(p);
        model.fit(split.X_train, y_train);
        float train_acc = model.score(split.X_train, y_train);
        float test_acc = model.score(split.X_test, y_test);
        std::printf("%-10.3f %12.4f %12.4f\n", static_cast<double>(gamma), static_cast<double>(train_acc), static_cast<double>(test_acc));
    }
}

void sweep_knn_k() {
    std::printf("\n=== KNN: k (breast-w, standardized) ===\n");
    std::printf("%-6s %12s %12s\n", "k", "train_acc", "test_acc");
    OpenMLDataset d = load_arff(data_path("breast-w.arff"), "breast-w");
    TrainTestSplit split = split_train_test(d.X, d.y, 0.2f, 42);
    standardize(split);
    for (int k : {1, 3, 5, 10, 20, 50}) {
        KNNParams p;
        p.k = k;
        KNNClassifier model(p);
        model.fit(split.X_train, split.y_train);
        float train_acc = model.score(split.X_train, split.y_train);
        float test_acc = model.score(split.X_test, split.y_test);
        std::printf("%-6d %12.4f %12.4f\n", k, static_cast<double>(train_acc), static_cast<double>(test_acc));
    }
}

void sweep_kmeans_k() {
    std::printf("\n=== KMeans: k vs. purity against true labels (balance-scale, 3 true classes) ===\n");
    std::printf("%-6s %12s %12s\n", "k", "inertia", "purity");
    OpenMLDataset d = load_arff(data_path("balance-scale.arff"), "balance-scale");
    for (int k : {2, 3, 4, 5, 6, 8}) {
        KMeansParams p;
        p.k = k;
        p.random_state = 7;
        KMeans model(p);
        model.fit(d.X);

        // Purity against the true (unused-in-training) labels: for each
        // cluster, count the majority true-label among its members.
        std::vector<std::map<int, int>> votes(static_cast<std::size_t>(k));
        for (std::size_t i = 0; i < d.X.size(); ++i) ++votes[static_cast<std::size_t>(model.labels()[i])][static_cast<int>(d.y[i])];
        int correct = 0;
        for (int c = 0; c < k; ++c) {
            int best = 0;
            for (const auto& kv : votes[static_cast<std::size_t>(c)]) best = std::max(best, kv.second);
            correct += best;
        }
        float purity = static_cast<float>(correct) / static_cast<float>(d.X.size());
        std::printf("%-6d %12.2f %12.4f\n", k, static_cast<double>(model.inertia()), static_cast<double>(purity));
    }
}

void sweep_pca_n_components() {
    std::printf("\n=== PCA: n_components vs. cumulative explained variance (wdbc) ===\n");
    std::printf("%-14s %14s\n", "n_components", "cum_var_ratio");
    OpenMLDataset d = load_arff(data_path("wdbc.arff"), "wdbc");
    for (int n : {1, 2, 3, 5, 10, 15, 20, 30}) {
        PCAParams p;
        p.n_components = n;
        p.random_state = 3;
        PCA model(p);
        model.fit(d.X);
        float cum = 0.0f;
        for (float r : model.explained_variance_ratio()) cum += r;
        std::printf("%-14d %14.4f\n", n, static_cast<double>(cum));
    }
}

void sweep_linear_model_alpha() {
    std::printf("\n=== LinearModel: alpha (diabetes, ridge logistic regression) ===\n");
    std::printf("%-10s %12s %12s\n", "alpha", "train_acc", "test_acc");
    OpenMLDataset d = load_arff(data_path("diabetes.arff"), "diabetes");
    TrainTestSplit split = split_train_test(d.X, d.y, 0.2f, 42);
    standardize(split);
    for (float alpha : {1e-5f, 1e-3f, 1e-2f, 0.1f, 1.0f, 10.0f}) {
        LinearModelParams p;
        p.loss = LinearLoss::LOGISTIC;
        p.optimizer = LinearOptimizer::LBFGS;
        p.alpha = alpha;
        p.max_iter = 100;
        LinearModel model(p);
        model.fit(split.X_train, split.y_train);
        float train_acc = model.score(split.X_train, split.y_train);
        float test_acc = model.score(split.X_test, split.y_test);
        std::printf("%-10.5f %12.4f %12.4f\n", static_cast<double>(alpha), static_cast<double>(train_acc), static_cast<double>(test_acc));
    }
}

}  // namespace

int main() {
    sweep_decision_tree_max_depth();
    sweep_random_forest_n_estimators();
    sweep_gbt_n_estimators();
    sweep_svm_gamma();
    sweep_knn_k();
    sweep_kmeans_k();
    sweep_pca_n_components();
    sweep_linear_model_alpha();
    return 0;
}
