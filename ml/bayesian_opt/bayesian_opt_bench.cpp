// bayesian_opt_bench.cpp — PLAN.md Phase 12c step 15: Bayesian
// optimization benchmarked against random search and grid search, on a
// real hyperparameter-tuning task (SVM's RBF `gamma`, tuned in log10
// space) across 3 real OpenML datasets reused from ml/openml_bench. Run
// manually (same convention as openml_bench/hyperparam_sweep/
// ensemble_bench/failure_modes_bench); results captured into README.md.
#include "arff_loader.h"
#include "bayesian_opt.h"
#include "svm.h"

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

void run_experiment_1d(const std::string& file, const std::string& name) {
    std::printf("\n=== %s: tuning SVM(RBF) gamma only (log10 space, bound [-4, 2]) ===\n", name.c_str());
    OpenMLDataset d = load_arff(std::string(OPENML_DATA_DIR) + "/" + file, name);
    TrainTestSplit split = split_train_test(d.X, d.y, 0.2f, 42);
    standardize(split);

    Labels y_train(split.y_train.size()), y_val(split.y_test.size());
    for (std::size_t i = 0; i < split.y_train.size(); ++i) y_train[i] = split.y_train[i] == 0.0f ? -1.0f : 1.0f;
    for (std::size_t i = 0; i < split.y_test.size(); ++i) y_val[i] = split.y_test[i] == 0.0f ? -1.0f : 1.0f;

    ObjectiveFn objective = [&](const std::vector<float>& x) {
        float gamma = std::pow(10.0f, x[0]);
        SVMParams p;
        p.kernel = KernelType::RBF;
        p.gamma = gamma;
        p.max_iter = 150;
        SVM model(p);
        model.fit(split.X_train, y_train);
        return model.score(split.X_test, y_val);
    };

    Bounds bounds = {{-4.0f, 2.0f}};
    const int total_budget = 15;

    BOParams bo_p;
    bo_p.n_initial_random = 5;
    bo_p.n_iterations = total_budget - bo_p.n_initial_random;
    bo_p.random_state = 42;
    BayesianOptimizer bo(bounds, bo_p);
    auto [bo_x, bo_acc] = bo.optimize(objective);

    auto [rs_x, rs_acc] = random_search(bounds, total_budget, objective, 42);
    auto [gs_x, gs_acc] = grid_search(bounds, total_budget, objective);

    std::printf("  Bayesian optimization: gamma=%.5f, val_acc=%.4f (%d evaluations)\n", static_cast<double>(std::pow(10.0f, bo_x[0])),
                static_cast<double>(bo_acc), total_budget);
    std::printf("  Random search:         gamma=%.5f, val_acc=%.4f (%d evaluations)\n", static_cast<double>(std::pow(10.0f, rs_x[0])),
                static_cast<double>(rs_acc), total_budget);
    std::printf("  Grid search:           gamma=%.5f, val_acc=%.4f (%d evaluations)\n", static_cast<double>(std::pow(10.0f, gs_x[0])),
                static_cast<double>(gs_acc), total_budget);
}

// A 2D joint tuning task (C and gamma both, log10 space): the regime
// where BO's sample efficiency over random/grid search is expected to
// actually show up. A 15-point 1D grid is dense; a naive extension to
// 2D at the same per-dimension density would need 15^2=225 evaluations,
// so under a fixed, modest total budget, grid search in 2D is
// necessarily much coarser (4x4=16 points) -- exactly the situation a
// model-based search should have an edge in.
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

    Bounds bounds = {{-2.0f, 3.0f}, {-4.0f, 2.0f}};  // log10(C), log10(gamma)

    BOParams bo_p;
    bo_p.n_initial_random = 6;
    bo_p.n_iterations = 10;  // total budget = 16, matching grid's 4x4
    bo_p.random_state = 42;
    BayesianOptimizer bo(bounds, bo_p);
    auto [bo_x, bo_acc] = bo.optimize(objective);

    auto [rs_x, rs_acc] = random_search(bounds, 16, objective, 42);
    auto [gs_x, gs_acc] = grid_search(bounds, 4, objective);  // 4x4 = 16 evaluations

    std::printf("  Bayesian optimization: C=%.4f gamma=%.5f, val_acc=%.4f (16 evaluations)\n", static_cast<double>(std::pow(10.0f, bo_x[0])),
                static_cast<double>(std::pow(10.0f, bo_x[1])), static_cast<double>(bo_acc));
    std::printf("  Random search:         C=%.4f gamma=%.5f, val_acc=%.4f (16 evaluations)\n", static_cast<double>(std::pow(10.0f, rs_x[0])),
                static_cast<double>(std::pow(10.0f, rs_x[1])), static_cast<double>(rs_acc));
    std::printf("  Grid search:           C=%.4f gamma=%.5f, val_acc=%.4f (16 evaluations)\n", static_cast<double>(std::pow(10.0f, gs_x[0])),
                static_cast<double>(std::pow(10.0f, gs_x[1])), static_cast<double>(gs_acc));
}

// A synthetic "needle in a haystack" function: a narrow Gaussian spike
// (std 0.05) at x=3 in a [-5,5] range, plus a tiny sinusoidal baseline
// so the function isn't literally flat elsewhere. Grid search's 15
// evenly-spaced points (spacing ~0.71) essentially can't land inside a
// spike this narrow; random search has some but low chance per draw.
// Run across 10 seeds to measure how often each method actually finds
// the spike, rather than reporting one (possibly lucky or unlucky) run.
void needle_in_haystack_multiseed() {
    std::printf("\n=== Synthetic needle-in-haystack: narrow spike at x=3, range [-5,5], 15-evaluation budget, 10 seeds ===\n");
    ObjectiveFn needle = [](const std::vector<float>& x) {
        float d = x[0] - 3.0f;
        return 10.0f * std::exp(-(d * d) / (2.0f * 0.05f * 0.05f)) + 0.1f * std::sin(x[0]);
    };
    Bounds bounds = {{-5.0f, 5.0f}};
    int budget = 15;
    const float found_threshold = 1.0f;  // baseline is ~0.1; anything above 1.0 means the spike was actually located

    int bo_found = 0, rs_found = 0;
    auto [gx, gv] = grid_search(bounds, budget, needle);
    std::printf("  grid search:  x=%.3f value=%.4f (found=%d) -- same result every time, no randomness\n", static_cast<double>(gx[0]),
                static_cast<double>(gv), gv > found_threshold ? 1 : 0);

    for (unsigned seed = 0; seed < 10; ++seed) {
        BOParams bo_p;
        bo_p.n_initial_random = 5;
        bo_p.n_iterations = 10;
        bo_p.random_state = seed;
        BayesianOptimizer bo(bounds, bo_p);
        auto [bx, bv] = bo.optimize(needle);
        auto [rx, rv] = random_search(bounds, budget, needle, seed);
        if (bv > found_threshold) ++bo_found;
        if (rv > found_threshold) ++rs_found;
        std::printf("  seed=%u  BO value=%.4f%s  random-search value=%.4f%s\n", seed, static_cast<double>(bv),
                    bv > found_threshold ? " (found spike)" : "", static_cast<double>(rv), rv > found_threshold ? " (found spike)" : "");
    }
    std::printf("  spike found: BO %d/10, random search %d/10, grid search %d/1\n", bo_found, rs_found, gv > found_threshold ? 1 : 0);
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
