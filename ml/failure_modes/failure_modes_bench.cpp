// failure_modes_bench.cpp — PLAN.md Phase 12b step 14: for each Phase
// 12a algorithm, a concrete, measured example of it failing badly, and
// why. Run manually (same convention as openml_bench/hyperparam_sweep/
// ensemble_bench -- see README.md's Design note); results captured into
// README.md.
#include "decision_tree.h"
#include "random_forest.h"
#include "gbt.h"
#include "svm.h"
#include "knn.h"
#include "kmeans.h"
#include "pca.h"
#include "linear_model.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <random>
#include <set>

namespace {

using Clock = std::chrono::steady_clock;
double elapsed_seconds(Clock::time_point start) { return std::chrono::duration<double>(Clock::now() - start).count(); }

float accuracy(const Labels& pred, const Labels& truth) {
    int correct = 0;
    for (std::size_t i = 0; i < truth.size(); ++i)
        if (pred[i] == truth[i]) ++correct;
    return static_cast<float>(correct) / static_cast<float>(truth.size());
}

// --- 1. RandomForest on highly imbalanced classes -----------------------
// Real overlap between the classes (distance 3, std 1.2) plus scarce
// minority training examples (only 50 of 1000): a training-set
// evaluation would let unlimited-depth trees simply memorize the rare
// minority points (an artifact, not a real measurement of the failure),
// so this evaluates on a SEPARATE, class-balanced held-out test set
// drawn from the same distributions -- the fair way to expose how the
// model actually generalizes on the ambiguous, overlapping region
// between the two classes.
void random_forest_imbalanced_classes() {
    std::printf("\n=== RandomForest: highly imbalanced classes (95:5 train), held-out balanced test ===\n");
    std::mt19937 rng(1);
    std::normal_distribution<float> noise(0.0f, 1.2f);
    Features X_train;
    Labels y_train;
    for (int i = 0; i < 950; ++i) {
        X_train.push_back({0.0f + noise(rng), 0.0f + noise(rng)});
        y_train.push_back(0.0f);
    }
    for (int i = 0; i < 50; ++i) {
        X_train.push_back({3.0f + noise(rng), 3.0f + noise(rng)});
        y_train.push_back(1.0f);
    }

    Features X_test;
    Labels y_test;
    for (int i = 0; i < 200; ++i) {
        X_test.push_back({0.0f + noise(rng), 0.0f + noise(rng)});
        y_test.push_back(0.0f);
    }
    for (int i = 0; i < 200; ++i) {
        X_test.push_back({3.0f + noise(rng), 3.0f + noise(rng)});
        y_test.push_back(1.0f);
    }

    RFParams p;
    RandomForest model(p);
    model.fit(X_train, y_train);
    Labels pred = model.predict(X_test);

    int minority_total = 0, minority_correct = 0, majority_total = 0, majority_correct = 0;
    for (std::size_t i = 0; i < y_test.size(); ++i) {
        if (y_test[i] == 1.0f) {
            ++minority_total;
            if (pred[i] == 1.0f) ++minority_correct;
        } else {
            ++majority_total;
            if (pred[i] == 0.0f) ++majority_correct;
        }
    }
    float overall_acc = accuracy(pred, y_test);
    float minority_recall = static_cast<float>(minority_correct) / static_cast<float>(minority_total);
    float majority_recall = static_cast<float>(majority_correct) / static_cast<float>(majority_total);

    std::printf("  balanced-test-set overall accuracy=%.4f\n", static_cast<double>(overall_acc));
    std::printf("  majority-class recall=%.4f, minority-class recall=%.4f (the real failure, hidden by overall accuracy alone)\n",
                static_cast<double>(majority_recall), static_cast<double>(minority_recall));
}

// --- 2. GBT on noisy labels (controlled label-flip, GBT vs RandomForest) ---
void gbt_noisy_labels() {
    std::printf("\n=== GBT vs RandomForest: controlled label noise ===\n");
    std::mt19937 data_rng(2);
    std::normal_distribution<float> noise(0.0f, 1.0f);
    Features X_train, X_test;
    Labels y_train_clean, y_test;
    for (int i = 0; i < 400; ++i) {
        float cx = (i % 2 == 0) ? -3.0f : 3.0f;
        X_train.push_back({cx + noise(data_rng), noise(data_rng)});
        y_train_clean.push_back(i % 2 == 0 ? 0.0f : 1.0f);
    }
    for (int i = 0; i < 200; ++i) {
        float cx = (i % 2 == 0) ? -3.0f : 3.0f;
        X_test.push_back({cx + noise(data_rng), noise(data_rng)});
        y_test.push_back(i % 2 == 0 ? 0.0f : 1.0f);
    }

    std::mt19937 flip_rng(3);
    std::uniform_real_distribution<float> flip_roll(0.0f, 1.0f);
    for (float flip_rate : {0.0f, 0.3f}) {
        Labels y_train_noisy = y_train_clean;
        for (auto& label : y_train_noisy)
            if (flip_roll(flip_rng) < flip_rate) label = 1.0f - label;

        GBTParams gbt_p;
        gbt_p.n_estimators = 200;
        GradientBoostedTrees gbt(gbt_p);
        gbt.fit(X_train, y_train_noisy);
        float gbt_acc = gbt.score(X_test, y_test);

        RFParams rf_p;
        RandomForest rf(rf_p);
        rf.fit(X_train, y_train_noisy);
        float rf_acc = rf.score(X_test, y_test);

        std::printf("  flip_rate=%.1f: GBT test_acc=%.4f, RandomForest test_acc=%.4f\n", static_cast<double>(flip_rate),
                    static_cast<double>(gbt_acc), static_cast<double>(rf_acc));
    }
}

// --- 3. SVM at scale (O(n^2) kernel matrix, timing sweep) -----------------
void svm_at_scale() {
    std::printf("\n=== SVM: training time vs. n (O(n^2) kernel matrix) ===\n");
    std::mt19937 rng(4);
    std::normal_distribution<float> noise(0.0f, 1.0f);
    double prev_time = 0.0;
    int prev_n = 0;
    for (int n : {200, 400, 800, 1600}) {
        Features X;
        Labels y;
        for (int i = 0; i < n; ++i) {
            std::vector<float> row(10);
            for (auto& v : row) v = noise(rng);
            X.push_back(row);
            y.push_back(i % 2 == 0 ? -1.0f : 1.0f);
        }
        SVMParams p;
        p.max_iter = 100;
        SVM model(p);
        auto t0 = Clock::now();
        model.fit(X, y);
        double t = elapsed_seconds(t0);
        double ratio = prev_time > 0.0 ? t / prev_time : 0.0;
        double n_ratio_sq = prev_n > 0 ? std::pow(static_cast<double>(n) / static_cast<double>(prev_n), 2.0) : 0.0;
        std::printf("  n=%-6d train_time=%.4fs  (time ratio=%.2f, n-ratio^2=%.2f)\n", n, t, ratio, n_ratio_sq);
        prev_time = t;
        prev_n = n;
    }
}

// --- 4. KNN in high dimensions (curse of dimensionality) ------------------
// A modest training set (120 rows) and modest class separation (the two
// classes overlap somewhat even in 2D) so that as pure-noise dimensions
// are added, their contribution to Euclidean distance increasingly
// swamps the 2 informative dimensions -- exactly the mechanism behind
// KNN's well-known high-dimensional degradation. LinearModel is included
// as a contrast: its learned per-feature weights can drive irrelevant
// dimensions toward zero instead of treating every dimension as equally
// distance-relevant.
void knn_high_dimensions() {
    std::printf("\n=== KNN vs LinearModel: accuracy as irrelevant noise dimensions grow ===\n");
    std::mt19937 rng(5);
    std::normal_distribution<float> class_noise(0.0f, 1.2f), pure_noise(0.0f, 1.5f);
    for (int n_noise_dims : {0, 20, 100, 300}) {
        Features X_train, X_test;
        Labels y_train, y_test;
        auto make_row = [&](bool is_class1) {
            std::vector<float> row;
            row.push_back((is_class1 ? 1.5f : -1.5f) + class_noise(rng));
            row.push_back(class_noise(rng));
            for (int d = 0; d < n_noise_dims; ++d) row.push_back(pure_noise(rng));
            return row;
        };
        for (int i = 0; i < 120; ++i) {
            bool c1 = i % 2 == 0;
            X_train.push_back(make_row(c1));
            y_train.push_back(c1 ? 1.0f : 0.0f);
        }
        for (int i = 0; i < 150; ++i) {
            bool c1 = i % 2 == 0;
            X_test.push_back(make_row(c1));
            y_test.push_back(c1 ? 1.0f : 0.0f);
        }

        KNNParams knn_p;
        knn_p.k = 5;
        KNNClassifier knn(knn_p);
        knn.fit(X_train, y_train);
        float knn_acc = knn.score(X_test, y_test);

        LinearModelParams lm_p;
        lm_p.loss = LinearLoss::LOGISTIC;
        lm_p.optimizer = LinearOptimizer::LBFGS;
        lm_p.alpha = 1e-2f;
        LinearModel lm(lm_p);
        lm.fit(X_train, y_train);
        float lm_acc = lm.score(X_test, y_test);

        std::printf("  noise_dims=%-4d (total d=%-4d) KNN test_acc=%.4f, LogisticRegression test_acc=%.4f\n", n_noise_dims,
                    n_noise_dims + 2, static_cast<double>(knn_acc), static_cast<double>(lm_acc));
    }
}

// --- 5. KMeans on non-convex clusters (two interleaved moons) -------------
void kmeans_non_convex_clusters() {
    std::printf("\n=== KMeans: purity on two interleaved moons (non-convex clusters) ===\n");
    std::mt19937 rng(6);
    std::normal_distribution<float> noise(0.0f, 0.1f);
    Features X;
    std::vector<int> true_label;
    int n_per_moon = 150;
    for (int i = 0; i < n_per_moon; ++i) {
        float theta = static_cast<float>(M_PI) * static_cast<float>(i) / static_cast<float>(n_per_moon);
        X.push_back({std::cos(theta) + noise(rng), std::sin(theta) + noise(rng)});
        true_label.push_back(0);
    }
    for (int i = 0; i < n_per_moon; ++i) {
        float theta = static_cast<float>(M_PI) * static_cast<float>(i) / static_cast<float>(n_per_moon);
        X.push_back({1.0f - std::cos(theta) + noise(rng), 0.5f - std::sin(theta) + noise(rng)});
        true_label.push_back(1);
    }

    KMeansParams p;
    p.k = 2;
    p.random_state = 7;
    KMeans model(p);
    model.fit(X);

    int correct_as_is = 0, correct_flipped = 0;
    for (std::size_t i = 0; i < X.size(); ++i) {
        if (model.labels()[i] == true_label[i]) ++correct_as_is;
        if (model.labels()[i] == (1 - true_label[i])) ++correct_flipped;
    }
    float purity = static_cast<float>(std::max(correct_as_is, correct_flipped)) / static_cast<float>(X.size());
    std::printf("  purity against true moon labels=%.4f (0.5 = chance for k=2)\n", static_cast<double>(purity));
}

// --- 6. PCA discarding the discriminative (low-variance) direction --------
void pca_discards_discriminative_direction() {
    std::printf("\n=== PCA + LogisticRegression: high-variance noise dim swamps low-variance signal ===\n");
    std::mt19937 rng(8);
    std::normal_distribution<float> big_noise(0.0f, 10.0f), small_signal(0.0f, 0.3f);
    Features X;
    Labels y;
    for (int i = 0; i < 400; ++i) {
        bool c1 = i % 2 == 0;
        X.push_back({big_noise(rng), (c1 ? 1.0f : -1.0f) + small_signal(rng)});
        y.push_back(c1 ? 1.0f : 0.0f);
    }

    LinearModelParams lm_p;
    lm_p.loss = LinearLoss::LOGISTIC;
    lm_p.optimizer = LinearOptimizer::LBFGS;
    lm_p.alpha = 1e-3f;

    // Raw 2D features (both dimensions available to the classifier).
    LinearModel lm_raw(lm_p);
    lm_raw.fit(X, y);
    float raw_acc = lm_raw.score(X, y);

    // PCA to 1 component, then classify on that single component.
    PCAParams pca_p;
    pca_p.n_components = 1;
    pca_p.random_state = 9;
    PCA pca(pca_p);
    Features X_reduced = pca.fit_transform(X);
    LinearModel lm_pca(lm_p);
    lm_pca.fit(X_reduced, y);
    float pca_acc = lm_pca.score(X_reduced, y);

    std::printf("  component[0] direction=[%.3f, %.3f] (near [1,0] means it kept the noise dim, not the signal dim)\n",
                static_cast<double>(pca.components()[0][0]), static_cast<double>(pca.components()[0][1]));
    std::printf("  classifier on raw 2D features: train_acc=%.4f\n", static_cast<double>(raw_acc));
    std::printf("  classifier on PCA(n_components=1): train_acc=%.4f\n", static_cast<double>(pca_acc));
}

// --- 7. DecisionTree instability (high variance across resamples) --------
// Heavy class overlap (means 2.0 apart, std 2.5) and a small dataset
// (120 rows): enough genuine ambiguity near the decision boundary that
// which points happen to land in each bootstrap resample can flip which
// splits look best. On too-clean, too-large data (the first attempt at
// this experiment), both a single tree and a 100-tree forest converge to
// nearly the same boundary regardless of resampling, masking the real
// difference -- this setup is tuned specifically so bagging's variance
// reduction has room to actually show up.
void decision_tree_instability() {
    std::printf("\n=== DecisionTree vs RandomForest: prediction instability across bootstrap resamples ===\n");
    std::mt19937 rng(10);
    std::normal_distribution<float> noise(0.0f, 2.5f), extra_noise(0.0f, 1.0f);
    Features X;
    Labels y;
    // 3 extra pure-noise features alongside the 2 informative ones (5
    // total) -- RandomForest's per-split feature subsampling defaults to
    // sqrt(5)~=2 features per split here, a regime where different
    // random subsets can actually differ meaningfully. With only the 2
    // informative features and no extras (the first version of this
    // experiment), sqrt(2)~=1 feature per split meant every tree in the
    // forest was already maximally randomized per split, and averaging
    // many maximally-randomized trees didn't reduce disagreement between
    // two forests any more than a single tree's own resampling variance
    // -- a real, honest finding in its own right (documented in
    // README.md), but not what this section is trying to isolate.
    for (int i = 0; i < 120; ++i) {
        bool c1 = i % 2 == 0;
        X.push_back({(c1 ? 2.0f : -2.0f) + noise(rng), noise(rng), extra_noise(rng), extra_noise(rng), extra_noise(rng)});
        y.push_back(c1 ? 1.0f : 0.0f);
    }

    std::uniform_int_distribution<std::size_t> pick(0, X.size() - 1);
    auto bootstrap = [&](std::mt19937& r) {
        Features Xb;
        Labels yb;
        for (std::size_t i = 0; i < X.size(); ++i) {
            std::size_t idx = pick(r);
            Xb.push_back(X[idx]);
            yb.push_back(y[idx]);
        }
        return std::make_pair(Xb, yb);
    };

    std::mt19937 boot_rng_a(11), boot_rng_b(12);
    auto [Xa, ya] = bootstrap(boot_rng_a);
    auto [Xb, yb] = bootstrap(boot_rng_b);

    DecisionTree tree_a, tree_b;
    tree_a.fit(Xa, ya);
    tree_b.fit(Xb, yb);
    Labels pred_a = tree_a.predict(X);
    Labels pred_b = tree_b.predict(X);
    int disagreements = 0;
    for (std::size_t i = 0; i < X.size(); ++i)
        if (pred_a[i] != pred_b[i]) ++disagreements;
    float disagreement_rate = static_cast<float>(disagreements) / static_cast<float>(X.size());

    RFParams rf_p;
    RandomForest rf_a(rf_p), rf_b(rf_p);
    rf_a.fit(Xa, ya);
    rf_b.fit(Xb, yb);
    Labels rf_pred_a = rf_a.predict(X);
    Labels rf_pred_b = rf_b.predict(X);
    int rf_disagreements = 0;
    for (std::size_t i = 0; i < X.size(); ++i)
        if (rf_pred_a[i] != rf_pred_b[i]) ++rf_disagreements;
    float rf_disagreement_rate = static_cast<float>(rf_disagreements) / static_cast<float>(X.size());

    std::printf("  two DecisionTrees on two bootstrap resamples of the same data disagree on %.4f of predictions\n",
                static_cast<double>(disagreement_rate));
    std::printf("  two RandomForests (100 trees each) on the same two resamples disagree on %.4f of predictions\n",
                static_cast<double>(rf_disagreement_rate));
}

// --- 8. LinearModel on non-linearly-separable data (XOR) ------------------
void linear_model_xor() {
    std::printf("\n=== LinearModel vs SVM(RBF): XOR (no linear decision boundary exists) ===\n");
    std::mt19937 rng(13);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    Features X;
    Labels y_binary;
    Labels y_pm1;
    for (int i = 0; i < 300; ++i) {
        float x0 = dist(rng), x1 = dist(rng);
        X.push_back({x0, x1});
        bool label = (x0 > 0) != (x1 > 0);
        y_binary.push_back(label ? 1.0f : 0.0f);
        y_pm1.push_back(label ? 1.0f : -1.0f);
    }

    float best_lm_acc = 0.0f;
    for (float alpha : {1e-5f, 1e-3f, 1e-1f, 1.0f, 10.0f}) {
        LinearModelParams p;
        p.loss = LinearLoss::LOGISTIC;
        p.optimizer = LinearOptimizer::LBFGS;
        p.alpha = alpha;
        LinearModel lm(p);
        lm.fit(X, y_binary);
        best_lm_acc = std::max(best_lm_acc, lm.score(X, y_binary));
    }

    SVMParams svm_p;
    svm_p.kernel = KernelType::RBF;
    svm_p.gamma = 2.0f;
    SVM svm(svm_p);
    svm.fit(X, y_pm1);
    float svm_acc = svm.score(X, y_pm1);

    std::printf("  best LogisticRegression accuracy across an alpha sweep=%.4f (no alpha setting escapes chance-level)\n",
                static_cast<double>(best_lm_acc));
    std::printf("  SVM(RBF) accuracy=%.4f (kernel trick captures the nonlinear boundary directly)\n", static_cast<double>(svm_acc));
}

}  // namespace

int main() {
    random_forest_imbalanced_classes();
    gbt_noisy_labels();
    svm_at_scale();
    knn_high_dimensions();
    kmeans_non_convex_clusters();
    pca_discards_discriminative_direction();
    decision_tree_instability();
    linear_model_xor();
    return 0;
}
