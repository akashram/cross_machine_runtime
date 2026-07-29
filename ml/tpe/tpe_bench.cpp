// tpe_bench.cpp — PLAN.md Phase 12c step 16: TPE benchmarked against
// GP-based Bayesian optimization (ml/bayesian_opt) and random search,
// on the identical real hyperparameter-tuning task (SVM's RBF `gamma`,
// and jointly `C`+`gamma`) bayesian_opt_bench.cpp uses, plus the same
// needle-in-haystack synthetic experiment, for a direct, apples-to-
// apples comparison. Run manually (same convention as the other Phase
// 12b/12c bench executables); results captured into README.md.
#include "arff_loader.h"
#include "bayesian_opt.h"
#include "svm.h"
#include "tpe.h"

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

void needle_in_haystack_multiseed() {
    std::printf("\n=== Synthetic needle-in-haystack: TPE vs GP-based BO vs random search, 10 seeds ===\n");
    ObjectiveFn needle = [](const std::vector<float>& x) {
        float d = x[0] - 3.0f;
        return 10.0f * std::exp(-(d * d) / (2.0f * 0.05f * 0.05f)) + 0.1f * std::sin(x[0]);
    };
    Bounds bounds = {{-5.0f, 5.0f}};
    int budget = 15;
    const float found_threshold = 1.0f;

    int tpe_found = 0, bo_found = 0, rs_found = 0;
    for (unsigned seed = 0; seed < 10; ++seed) {
        TPEParams tpe_p;
        tpe_p.n_initial_random = 5;
        tpe_p.n_iterations = 10;
        tpe_p.random_state = seed;
        TPEOptimizer tpe(bounds, tpe_p);
        auto [tx, tv] = tpe.optimize(needle);

        BOParams bo_p;
        bo_p.n_initial_random = 5;
        bo_p.n_iterations = 10;
        bo_p.random_state = seed;
        BayesianOptimizer bo(bounds, bo_p);
        auto [bx, bv] = bo.optimize(needle);

        auto [rx, rv] = random_search(bounds, budget, needle, seed);

        if (tv > found_threshold) ++tpe_found;
        if (bv > found_threshold) ++bo_found;
        if (rv > found_threshold) ++rs_found;
        std::printf("  seed=%u  TPE=%.4f%s  GP-BO=%.4f%s  random=%.4f%s\n", seed, static_cast<double>(tv),
                    tv > found_threshold ? "*" : "", static_cast<double>(bv), bv > found_threshold ? "*" : "", static_cast<double>(rv),
                    rv > found_threshold ? "*" : "");
    }
    std::printf("  spike found (* above): TPE %d/10, GP-BO %d/10, random search %d/10\n", tpe_found, bo_found, rs_found);
}

void run_experiment_1d(const std::string& file, const std::string& name) {
    std::printf("\n=== %s: tuning SVM(RBF) gamma only (log10 space, bound [-4, 2]) ===\n", name.c_str());
    OpenMLDataset d = load_arff(std::string(OPENML_DATA_DIR) + "/" + file, name);
    TrainTestSplit split = split_train_test(d.X, d.y, 0.2f, 42);
    standardize(split);

    Labels y_train(split.y_train.size()), y_val(split.y_test.size());
    for (std::size_t i = 0; i < split.y_train.size(); ++i) y_train[i] = split.y_train[i] == 0.0f ? -1.0f : 1.0f;
    for (std::size_t i = 0; i < split.y_test.size(); ++i) y_val[i] = split.y_test[i] == 0.0f ? -1.0f : 1.0f;

    ObjectiveFn objective = [&](const std::vector<float>& x) {
        SVMParams p;
        p.kernel = KernelType::RBF;
        p.gamma = std::pow(10.0f, x[0]);
        p.max_iter = 150;
        SVM model(p);
        model.fit(split.X_train, y_train);
        return model.score(split.X_test, y_val);
    };

    Bounds bounds = {{-4.0f, 2.0f}};
    const int total_budget = 15;

    TPEParams tpe_p;
    tpe_p.n_initial_random = 5;
    tpe_p.n_iterations = total_budget - tpe_p.n_initial_random;
    tpe_p.random_state = 42;
    TPEOptimizer tpe(bounds, tpe_p);
    auto [tpe_x, tpe_acc] = tpe.optimize(objective);

    BOParams bo_p;
    bo_p.n_initial_random = 5;
    bo_p.n_iterations = total_budget - bo_p.n_initial_random;
    bo_p.random_state = 42;
    BayesianOptimizer bo(bounds, bo_p);
    auto [bo_x, bo_acc] = bo.optimize(objective);

    auto [rs_x, rs_acc] = random_search(bounds, total_budget, objective, 42);

    std::printf("  TPE:                    gamma=%.5f, val_acc=%.4f (%d evaluations)\n", static_cast<double>(std::pow(10.0f, tpe_x[0])),
                static_cast<double>(tpe_acc), total_budget);
    std::printf("  GP-based BO:            gamma=%.5f, val_acc=%.4f (%d evaluations)\n", static_cast<double>(std::pow(10.0f, bo_x[0])),
                static_cast<double>(bo_acc), total_budget);
    std::printf("  Random search:          gamma=%.5f, val_acc=%.4f (%d evaluations)\n", static_cast<double>(std::pow(10.0f, rs_x[0])),
                static_cast<double>(rs_acc), total_budget);
}

void run_experiment_2d(const std::string& file, const std::string& name) {
    std::printf("\n=== %s: tuning SVM(RBF) C and gamma jointly (log10 space) ===\n", name.c_str());
    OpenMLDataset d = load_arff(std::string(OPENML_DATA_DIR) + "/" + file, name);
    TrainTestSplit split = split_train_test(d.X, d.y, 0.2f, 42);
    standardize(split);

    Labels y_train(split.y_train.size()), y_val(split.y_test.size());
    for (std::size_t i = 0; i < split.y_train.size(); ++i) y_train[i] = split.y_train[i] == 0.0f ? -1.0f : 1.0f;
    for (std::size_t i = 0; i < split.y_test.size(); ++i) y_val[i] = split.y_test[i] == 0.0f ? -1.0f : 1.0f;

    ObjectiveFn objective = [&](const std::vector<float>& x) {
        SVMParams p;
        p.kernel = KernelType::RBF;
        p.C = std::pow(10.0f, x[0]);
        p.gamma = std::pow(10.0f, x[1]);
        p.max_iter = 150;
        SVM model(p);
        model.fit(split.X_train, y_train);
        return model.score(split.X_test, y_val);
    };

    Bounds bounds = {{-2.0f, 3.0f}, {-4.0f, 2.0f}};

    TPEParams tpe_p;
    tpe_p.n_initial_random = 6;
    tpe_p.n_iterations = 10;
    tpe_p.random_state = 42;
    TPEOptimizer tpe(bounds, tpe_p);
    auto [tpe_x, tpe_acc] = tpe.optimize(objective);

    BOParams bo_p;
    bo_p.n_initial_random = 6;
    bo_p.n_iterations = 10;
    bo_p.random_state = 42;
    BayesianOptimizer bo(bounds, bo_p);
    auto [bo_x, bo_acc] = bo.optimize(objective);

    auto [rs_x, rs_acc] = random_search(bounds, 16, objective, 42);

    std::printf("  TPE:          C=%.4f gamma=%.5f, val_acc=%.4f (16 evaluations)\n", static_cast<double>(std::pow(10.0f, tpe_x[0])),
                static_cast<double>(std::pow(10.0f, tpe_x[1])), static_cast<double>(tpe_acc));
    std::printf("  GP-based BO:  C=%.4f gamma=%.5f, val_acc=%.4f (16 evaluations)\n", static_cast<double>(std::pow(10.0f, bo_x[0])),
                static_cast<double>(std::pow(10.0f, bo_x[1])), static_cast<double>(bo_acc));
    std::printf("  Random:       C=%.4f gamma=%.5f, val_acc=%.4f (16 evaluations)\n", static_cast<double>(std::pow(10.0f, rs_x[0])),
                static_cast<double>(std::pow(10.0f, rs_x[1])), static_cast<double>(rs_acc));
}

}  // namespace

int main() {
    needle_in_haystack_multiseed();
    run_experiment_1d("blood-transfusion.arff", "blood-transfusion");
    run_experiment_1d("breast-w.arff", "breast-w");
    run_experiment_1d("wdbc.arff", "wdbc");
    run_experiment_2d("blood-transfusion.arff", "blood-transfusion");
    run_experiment_2d("breast-w.arff", "breast-w");
    run_experiment_2d("wdbc.arff", "wdbc");
    return 0;
}
