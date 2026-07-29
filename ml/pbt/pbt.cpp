#include "pbt.h"

#include <algorithm>
#include <numeric>
#include <random>

PBTResult population_based_training(std::vector<PBTMember> population, const TrainStepFn& train_step, const CloneStateFn& clone_state,
                                     PBTParams params) {
    PBTResult result;
    std::mt19937 rng(params.random_state);
    int n = static_cast<int>(population.size());

    for (int round = 0; round < params.n_rounds; ++round) {
        for (int i = 0; i < n; ++i) population[static_cast<std::size_t>(i)].score = train_step(i, population[static_cast<std::size_t>(i)].hyperparams, params.steps_per_round);

        std::vector<int> order(static_cast<std::size_t>(n));
        std::iota(order.begin(), order.end(), 0);
        std::sort(order.begin(), order.end(), [&](int a, int b) { return population[static_cast<std::size_t>(a)].score > population[static_cast<std::size_t>(b)].score; });
        result.best_score_per_round.push_back(population[static_cast<std::size_t>(order[0])].score);

        // Exploit/explore: skip on the final round -- there is no
        // subsequent training step to benefit from a mutation, so
        // mutating now would only throw away the final, already-
        // measured state of the weaker members for no reason.
        if (round + 1 < params.n_rounds) {
            int n_bottom = std::min(n, std::max(1, static_cast<int>(static_cast<float>(n) * params.exploit_bottom_fraction)));
            int n_top = std::min(n, std::max(1, static_cast<int>(static_cast<float>(n) * params.exploit_top_fraction)));
            std::uniform_int_distribution<int> top_pick(0, n_top - 1);
            std::uniform_real_distribution<float> perturb(params.perturb_factor_low, params.perturb_factor_high);

            for (int rank = n - n_bottom; rank < n; ++rank) {
                int dst = order[static_cast<std::size_t>(rank)];
                int src = order[static_cast<std::size_t>(top_pick(rng))];
                if (src == dst) continue;  // only possible for tiny populations where the top/bottom fractions overlap in rank

                clone_state(dst, src);
                population[static_cast<std::size_t>(dst)].hyperparams = population[static_cast<std::size_t>(src)].hyperparams;
                for (float& h : population[static_cast<std::size_t>(dst)].hyperparams) h *= perturb(rng);
                population[static_cast<std::size_t>(dst)].score = population[static_cast<std::size_t>(src)].score;  // pending re-evaluation next round
            }
        }
    }

    result.final_population = population;
    int best = 0;
    for (int i = 1; i < n; ++i)
        if (population[static_cast<std::size_t>(i)].score > population[static_cast<std::size_t>(best)].score) best = i;
    result.best_member_idx = best;
    return result;
}
