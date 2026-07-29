// hyperband_bench.cpp — PLAN.md Phase 12c step 17: Successive Halving,
// Hyperband, and ASHA benchmarked on a real hyperparameter-tuning task
// -- GBT's (learning_rate, max_depth) config, with resource = number of
// boosting rounds (n_estimators). Each `evaluate()` call below actually
// retrains a fresh GBT with n_estimators=resource -- paying the real
// training cost every time (not reusing staged_predict() to fake cheap
// partial evaluation) so the resource-consumption numbers reported
// genuinely reflect the compute Hyperband/ASHA's early stopping saves
// vs. training every sampled config to the max resource. Run manually
// (same convention as the other Phase 12b/12c bench executables);
// results captured into README.md.
#include "arff_loader.h"
#include "gbt.h"
#include "hyperband.h"

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

int total_resource(const std::vector<EvalRecord>& history) {
    int total = 0;
    for (const auto& rec : history) total += rec.resource;
    return total;
}

void run_experiment(const std::string& file, const std::string& name) {
    std::printf("\n=== %s: tuning GBT (learning_rate, max_depth), resource=n_estimators ===\n", name.c_str());
    OpenMLDataset d = load_arff(std::string(OPENML_DATA_DIR) + "/" + file, name);
    TrainTestSplit split = split_train_test(d.X, d.y, 0.2f, 42);
    standardize(split);

    const int min_resource = 5, max_resource = 45;  // rungs at eta=3: 5, 15, 45
    const float eta = 3.0f;
    const int n_configs_full = 9;  // 27/3 -- matches Hyperband's total-budget-per-bracket scale

    ResourceEvalFn evaluate = [&](const Config& c, int resource) {
        GBTParams p;
        p.learning_rate = c[0];
        p.max_depth = static_cast<int>(c[1]);
        p.n_estimators = resource;
        GradientBoostedTrees model(p);
        model.fit(split.X_train, split.y_train);
        return model.score(split.X_test, split.y_test);
    };

    ConfigSampler sampler = [](std::mt19937& rng) {
        std::uniform_real_distribution<float> lr_dist(0.01f, 0.5f);
        std::uniform_int_distribution<int> depth_dist(2, 8);
        return Config{lr_dist(rng), static_cast<float>(depth_dist(rng))};
    };

    // "Full training" baseline: n_configs_full random configs, each
    // trained straight to max_resource -- no early stopping at all, the
    // naive approach Hyperband/ASHA are meant to beat on total cost.
    std::mt19937 baseline_rng(42);
    float baseline_best = -1.0f;
    Config baseline_best_config;
    int baseline_total_resource = 0;
    for (int i = 0; i < n_configs_full; ++i) {
        Config c = sampler(baseline_rng);
        float s = evaluate(c, max_resource);
        baseline_total_resource += max_resource;
        if (s > baseline_best) {
            baseline_best = s;
            baseline_best_config = c;
        }
    }

    std::mt19937 sha_rng(42);
    std::vector<Config> sha_configs;
    for (int i = 0; i < n_configs_full; ++i) sha_configs.push_back(sampler(sha_rng));
    SHAResult sha_result = successive_halving(sha_configs, min_resource, max_resource, eta, evaluate);

    SHAResult hb_result = hyperband(sampler, max_resource, eta, evaluate, 42);

    SHAResult asha_result = asha(sampler, min_resource, max_resource, eta, n_configs_full, evaluate, 42);

    std::printf("  Full training (baseline): best_val_acc=%.4f, total_resource=%d (lr=%.3f depth=%d)\n",
                static_cast<double>(baseline_best), baseline_total_resource, static_cast<double>(baseline_best_config[0]),
                static_cast<int>(baseline_best_config[1]));
    std::printf("  Successive Halving:       best_val_acc=%.4f, total_resource=%d (%.1f%% of baseline)\n",
                static_cast<double>(sha_result.best_score), total_resource(sha_result.history),
                100.0 * total_resource(sha_result.history) / baseline_total_resource);
    std::printf("  Hyperband:                best_val_acc=%.4f, total_resource=%d\n", static_cast<double>(hb_result.best_score),
                total_resource(hb_result.history));
    std::printf("  ASHA:                     best_val_acc=%.4f, total_resource=%d (%.1f%% of baseline)\n",
                static_cast<double>(asha_result.best_score), total_resource(asha_result.history),
                100.0 * total_resource(asha_result.history) / baseline_total_resource);
}

}  // namespace

int main() {
    run_experiment("blood-transfusion.arff", "blood-transfusion");
    run_experiment("breast-w.arff", "breast-w");
    return 0;
}
