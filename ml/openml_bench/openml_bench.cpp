// openml_bench.cpp — PLAN.md Phase 12a step 9: automated evaluation
// harness. Loads all 8 selected OpenML CC-18 datasets, trains every
// classifier from Phase 12a on each, records accuracy + training time +
// inference latency. Run manually (see README.md's Design note on why
// this isn't a `ctest` target); results captured into README.md.
//
// PLAN.md's step 4 (SVM) calls for "binary + multiclass (one-vs-rest)"
// but svm.h/gbt.h/LinearModel's logistic loss are all binary-only in
// their own modules (see each header's own scope note). Rather than
// change those modules, this harness adds the standard one-vs-rest
// reduction here (see one_vs_rest_* helpers below) so every algorithm
// runs on every dataset, including the 3 multiclass ones. KMeans and
// PCA are unsupervised and excluded -- OpenML-CC18 is defined as a
// classification suite, so there is no accuracy number to compare them
// against.
#include "arff_loader.h"
#include "decision_tree.h"
#include "random_forest.h"
#include "gbt.h"
#include "svm.h"
#include "knn.h"
#include "linear_model.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <functional>
#include <string>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;
double elapsed_seconds(Clock::time_point start) {
    return std::chrono::duration<double>(Clock::now() - start).count();
}

struct Result {
    std::string algorithm;
    double train_seconds = 0.0;
    double predict_seconds_per_sample = 0.0;
    float accuracy = 0.0f;
};

// One-vs-rest argmax over per-class real-valued scores (see file header
// note on why this lives here rather than in svm.h/gbt.h/linear_model.h).
Labels argmax_ovr(const std::vector<std::vector<float>>& scores_per_class) {
    std::size_t n_rows = scores_per_class[0].size();
    Labels out(n_rows);
    for (std::size_t i = 0; i < n_rows; ++i) {
        int best_class = 0;
        float best_score = scores_per_class[0][i];
        for (std::size_t c = 1; c < scores_per_class.size(); ++c) {
            if (scores_per_class[c][i] > best_score) {
                best_score = scores_per_class[c][i];
                best_class = static_cast<int>(c);
            }
        }
        out[i] = static_cast<float>(best_class);
    }
    return out;
}

float accuracy(const Labels& pred, const Labels& truth) {
    int correct = 0;
    for (std::size_t i = 0; i < truth.size(); ++i)
        if (pred[i] == truth[i]) ++correct;
    return static_cast<float>(correct) / static_cast<float>(truth.size());
}

// Z-score standardization, fit on the train split only (no test-set
// leakage) and applied to both. Trees (DecisionTree/RandomForest/GBT)
// are scale-invariant, so this doesn't change their results, but
// distance/margin-based methods (SVM, KNN, LogisticRegression) are
// genuinely scale-sensitive -- an unscaled first run of this harness
// showed SVM at 27-31% accuracy (near/below chance) on vehicle and
// mfeat-morphological purely from unnormalized feature scales, not any
// real algorithmic weakness. Standardizing uniformly before benchmarking
// removes that confound so the cross-method comparison (step 10)
// reflects real algorithm differences, not a missing preprocessing step.
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

Result bench_decision_tree(const TrainTestSplit& split) {
    Result r{"DecisionTree"};
    DecisionTree model;
    auto t0 = Clock::now();
    model.fit(split.X_train, split.y_train);
    r.train_seconds = elapsed_seconds(t0);
    auto t1 = Clock::now();
    Labels pred = model.predict(split.X_test);
    r.predict_seconds_per_sample = elapsed_seconds(t1) / static_cast<double>(split.X_test.size());
    r.accuracy = accuracy(pred, split.y_test);
    return r;
}

Result bench_random_forest(const TrainTestSplit& split) {
    Result r{"RandomForest"};
    RFParams p;
    p.n_estimators = 50;  // kept modest so the full 8-dataset sweep finishes quickly; see README
    RandomForest model(p);
    auto t0 = Clock::now();
    model.fit(split.X_train, split.y_train);
    r.train_seconds = elapsed_seconds(t0);
    auto t1 = Clock::now();
    Labels pred = model.predict(split.X_test);
    r.predict_seconds_per_sample = elapsed_seconds(t1) / static_cast<double>(split.X_test.size());
    r.accuracy = accuracy(pred, split.y_test);
    return r;
}

// GBT is binary-only (gbt.h). Multiclass datasets get the one-vs-rest
// reduction: one binary GBT per class, argmax over predict_proba.
Result bench_gbt(const TrainTestSplit& split, int n_classes) {
    Result r{"GBT"};
    GBTParams p;
    p.n_estimators = 50;  // kept modest; see README

    auto t0 = Clock::now();
    if (n_classes == 2) {
        GradientBoostedTrees model(p);
        model.fit(split.X_train, split.y_train);
        r.train_seconds = elapsed_seconds(t0);
        auto t1 = Clock::now();
        Labels pred = model.predict(split.X_test);
        r.predict_seconds_per_sample = elapsed_seconds(t1) / static_cast<double>(split.X_test.size());
        r.accuracy = accuracy(pred, split.y_test);
    } else {
        std::vector<std::vector<float>> scores(static_cast<std::size_t>(n_classes));
        for (int c = 0; c < n_classes; ++c) {
            Labels binary_y(split.y_train.size());
            for (std::size_t i = 0; i < split.y_train.size(); ++i) binary_y[i] = (split.y_train[i] == static_cast<float>(c)) ? 1.0f : 0.0f;
            GradientBoostedTrees model(p);
            model.fit(split.X_train, binary_y);
            scores[static_cast<std::size_t>(c)] = model.predict_proba(split.X_test);
        }
        r.train_seconds = elapsed_seconds(t0);
        auto t1 = Clock::now();
        Labels pred = argmax_ovr(scores);
        r.predict_seconds_per_sample = elapsed_seconds(t1) / static_cast<double>(split.X_test.size());
        r.accuracy = accuracy(pred, split.y_test);
    }
    return r;
}

// SVM is binary-only (y in {-1,+1}, svm.h). Multiclass: one-vs-rest via
// decision_function() (added to svm.h specifically to support this).
Result bench_svm(const TrainTestSplit& split, int n_classes) {
    Result r{"SVM"};
    SVMParams p;
    p.kernel = KernelType::RBF;
    p.max_iter = 200;  // kept modest; see README

    auto t0 = Clock::now();
    if (n_classes == 2) {
        Labels binary_y(split.y_train.size());
        for (std::size_t i = 0; i < split.y_train.size(); ++i) binary_y[i] = split.y_train[i] == 0.0f ? -1.0f : 1.0f;
        SVM model(p);
        model.fit(split.X_train, binary_y);
        r.train_seconds = elapsed_seconds(t0);
        auto t1 = Clock::now();
        std::vector<float> scores = model.decision_function(split.X_test);
        Labels pred(scores.size());
        for (std::size_t i = 0; i < scores.size(); ++i) pred[i] = scores[i] >= 0.0f ? 1.0f : 0.0f;
        r.predict_seconds_per_sample = elapsed_seconds(t1) / static_cast<double>(split.X_test.size());
        r.accuracy = accuracy(pred, split.y_test);
    } else {
        std::vector<std::vector<float>> scores(static_cast<std::size_t>(n_classes));
        for (int c = 0; c < n_classes; ++c) {
            Labels binary_y(split.y_train.size());
            for (std::size_t i = 0; i < split.y_train.size(); ++i) binary_y[i] = (split.y_train[i] == static_cast<float>(c)) ? 1.0f : -1.0f;
            SVM model(p);
            model.fit(split.X_train, binary_y);
            scores[static_cast<std::size_t>(c)] = model.decision_function(split.X_test);
        }
        r.train_seconds = elapsed_seconds(t0);
        auto t1 = Clock::now();
        Labels pred = argmax_ovr(scores);
        r.predict_seconds_per_sample = elapsed_seconds(t1) / static_cast<double>(split.X_test.size());
        r.accuracy = accuracy(pred, split.y_test);
    }
    return r;
}

Result bench_knn(const TrainTestSplit& split) {
    Result r{"KNN"};
    KNNParams p;
    p.k = 5;
    p.structure = NeighborStructure::KD_TREE;
    KNNClassifier model(p);
    auto t0 = Clock::now();
    model.fit(split.X_train, split.y_train);
    r.train_seconds = elapsed_seconds(t0);
    auto t1 = Clock::now();
    Labels pred = model.predict(split.X_test);
    r.predict_seconds_per_sample = elapsed_seconds(t1) / static_cast<double>(split.X_test.size());
    r.accuracy = accuracy(pred, split.y_test);
    return r;
}

// LinearModel's LOGISTIC loss is binary-only (y in {0,1}). Multiclass:
// one-vs-rest via predict_proba().
Result bench_linear_model(const TrainTestSplit& split, int n_classes) {
    Result r{"LogisticRegression"};
    LinearModelParams p;
    p.loss = LinearLoss::LOGISTIC;
    p.optimizer = LinearOptimizer::LBFGS;
    p.alpha = 1e-3f;
    p.max_iter = 100;

    auto t0 = Clock::now();
    if (n_classes == 2) {
        LinearModel model(p);
        model.fit(split.X_train, split.y_train);
        r.train_seconds = elapsed_seconds(t0);
        auto t1 = Clock::now();
        Labels pred = model.predict(split.X_test);
        r.predict_seconds_per_sample = elapsed_seconds(t1) / static_cast<double>(split.X_test.size());
        r.accuracy = accuracy(pred, split.y_test);
    } else {
        std::vector<std::vector<float>> scores(static_cast<std::size_t>(n_classes));
        for (int c = 0; c < n_classes; ++c) {
            Labels binary_y(split.y_train.size());
            for (std::size_t i = 0; i < split.y_train.size(); ++i) binary_y[i] = (split.y_train[i] == static_cast<float>(c)) ? 1.0f : 0.0f;
            LinearModel model(p);
            model.fit(split.X_train, binary_y);
            scores[static_cast<std::size_t>(c)] = model.predict_proba(split.X_test);
        }
        r.train_seconds = elapsed_seconds(t0);
        auto t1 = Clock::now();
        Labels pred = argmax_ovr(scores);
        r.predict_seconds_per_sample = elapsed_seconds(t1) / static_cast<double>(split.X_test.size());
        r.accuracy = accuracy(pred, split.y_test);
    }
    return r;
}

}  // namespace

int main() {
    struct DatasetSpec {
        std::string file;
        std::string name;
    };
    std::vector<DatasetSpec> specs = {
        {"balance-scale.arff", "balance-scale"},
        {"banknote-authentication.arff", "banknote-authentication"},
        {"blood-transfusion.arff", "blood-transfusion"},
        {"diabetes.arff", "diabetes"},
        {"breast-w.arff", "breast-w"},
        {"wdbc.arff", "wdbc"},
        {"vehicle.arff", "vehicle"},
        {"mfeat-morphological.arff", "mfeat-morphological"},
    };

    std::printf("%-24s %-20s %8s %10s %14s %10s\n", "dataset", "algorithm", "n", "d", "train(s)", "accuracy");
    std::printf("%s\n", std::string(90, '-').c_str());

    for (const auto& spec : specs) {
        OpenMLDataset dataset = load_arff(std::string(OPENML_DATA_DIR) + "/" + spec.file, spec.name);
        TrainTestSplit split = split_train_test(dataset.X, dataset.y, 0.2f, 42);
        standardize(split);
        int n_classes = static_cast<int>(dataset.class_names.size());

        std::vector<Result> results = {
            bench_decision_tree(split),
            bench_random_forest(split),
            bench_gbt(split, n_classes),
            bench_svm(split, n_classes),
            bench_knn(split),
            bench_linear_model(split, n_classes),
        };

        for (const auto& r : results) {
            std::printf("%-24s %-20s %8zu %10zu %14.4f %10.4f\n", dataset.name.c_str(), r.algorithm.c_str(), dataset.X.size(),
                        dataset.X[0].size(), r.train_seconds, static_cast<double>(r.accuracy));
        }
    }
    return 0;
}
