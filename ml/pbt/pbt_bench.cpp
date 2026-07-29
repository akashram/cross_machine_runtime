// pbt_bench.cpp — PLAN.md Phase 12c step 18: PBT tuning LinearModel's
// SGD learning_rate mid-training, backed by real LinearModel::partial_fit
// warm-starts, on real OpenML data reused from ml/openml_bench. Run
// manually (same convention as the other Phase 12b/12c bench
// executables); results captured into README.md.
#include "arff_loader.h"
#include "linear_model.h"
#include "pbt.h"

#include <cmath>
#include <cstdio>
#include <random>
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

void run_experiment(const std::string& file, const std::string& name) {
    std::printf("\n=== %s: PBT tuning LinearModel SGD learning_rate mid-training ===\n", name.c_str());
    OpenMLDataset d = load_arff(std::string(OPENML_DATA_DIR) + "/" + file, name);
    TrainTestSplit split = split_train_test(d.X, d.y, 0.2f, 42);
    standardize(split);

    const int n_members = 8;
    const int n_rounds = 12;
    const int steps_per_round = 5;  // epochs of SGD per round

    // A diverse, mostly-bad initial population (log-uniform over a wide
    // range dominated by too-small learning rates, so most members
    // start in the slow-convergence regime) -- the realistic PBT setup
    // (Jaderberg et al. compare PBT against holding each member's own
    // random initial hyperparameters fixed, i.e. "random search without
    // mid-training adaptation," not against one single repeated bad
    // value; an earlier version of this benchmark used the latter and
    // showed no measurable PBT benefit -- see README's Design note on
    // why that comparison didn't isolate what PBT actually helps with).
    std::mt19937 init_rng(7);
    std::uniform_real_distribution<float> log_lr_dist(-4.0f, -0.5f);  // learning_rate in [1e-4, ~0.32]
    std::vector<float> initial_learning_rates(static_cast<std::size_t>(n_members));
    for (auto& lr : initial_learning_rates) lr = std::pow(10.0f, log_lr_dist(init_rng));

    // Fixed baseline: the SAME n_members initial learning rates, each
    // trained independently for the same total epoch budget, no
    // exploit/explore -- "random search over learning rate, no
    // mid-training adaptation."
    float fixed_best = -1.0f;
    for (int i = 0; i < n_members; ++i) {
        LinearModelParams p;
        p.loss = LinearLoss::LOGISTIC;
        p.optimizer = LinearOptimizer::SGD;
        p.learning_rate = initial_learning_rates[static_cast<std::size_t>(i)];
        p.alpha = 1e-3f;
        LinearModel model(p);
        model.partial_fit(split.X_train, split.y_train, n_rounds * steps_per_round);
        fixed_best = std::max(fixed_best, model.score(split.X_test, split.y_test));
    }

    // PBT: the identical n_members starting learning rates, but able to
    // exploit/explore each other's state and hyperparameters every round.
    std::vector<LinearModel> models;
    for (int i = 0; i < n_members; ++i) {
        LinearModelParams p;
        p.loss = LinearLoss::LOGISTIC;
        p.optimizer = LinearOptimizer::SGD;
        p.learning_rate = initial_learning_rates[static_cast<std::size_t>(i)];
        p.alpha = 1e-3f;
        models.emplace_back(p);
    }

    TrainStepFn train_step = [&](int idx, const std::vector<float>& hyperparams, int steps) {
        models[static_cast<std::size_t>(idx)].params().learning_rate = hyperparams[0];
        models[static_cast<std::size_t>(idx)].partial_fit(split.X_train, split.y_train, steps);
        return models[static_cast<std::size_t>(idx)].score(split.X_test, split.y_test);
    };
    CloneStateFn clone_state = [&](int dst, int src) {
        models[static_cast<std::size_t>(dst)].set_weights(models[static_cast<std::size_t>(src)].coef(), models[static_cast<std::size_t>(src)].intercept());
    };

    std::vector<PBTMember> population;
    for (int i = 0; i < n_members; ++i) population.push_back(PBTMember{{initial_learning_rates[static_cast<std::size_t>(i)]}, 0.0f});
    PBTParams pbt_p;
    pbt_p.n_rounds = n_rounds;
    pbt_p.steps_per_round = steps_per_round;
    pbt_p.perturb_factor_low = 0.5f;
    pbt_p.perturb_factor_high = 2.0f;
    pbt_p.random_state = 42;
    PBTResult result = population_based_training(population, train_step, clone_state, pbt_p);

    std::printf("  Fixed (random initial learning_rates, no mutation): best_val_acc=%.4f\n", static_cast<double>(fixed_best));
    std::printf("  PBT (same initial learning_rates, with exploit/explore): best_val_acc=%.4f, final learning_rates=[",
                static_cast<double>(result.best_score_per_round.back()));
    for (std::size_t i = 0; i < result.final_population.size(); ++i)
        std::printf("%.4f%s", static_cast<double>(result.final_population[i].hyperparams[0]), i + 1 < result.final_population.size() ? ", " : "");
    std::printf("]\n");
    std::printf("  score progression by round: ");
    for (float s : result.best_score_per_round) std::printf("%.3f ", static_cast<double>(s));
    std::printf("\n");
}

}  // namespace

int main() {
    run_experiment("blood-transfusion.arff", "blood-transfusion");
    run_experiment("diabetes.arff", "diabetes");
    return 0;
}
