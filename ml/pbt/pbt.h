#pragma once
#include <functional>
#include <vector>

// PLAN.md Phase 12c step 18: Population-based training (Jaderberg et
// al. 2017) -- exploit/explore schedule, mutation of hyperparameters
// mid-training.
//
// Unlike ml/bayesian_opt, ml/tpe, and ml/hyperband (all of which
// restart training from scratch for every config), PBT's whole point
// is that training CONTINUES: poorly performing population members
// copy a stronger member's TRAINED STATE (exploit), then perturb the
// copied hyperparameters (explore) and keep training from there. PBT is
// generic over what "state" and "training" mean, so this library
// doesn't know about model weights directly -- callers provide:

// One training step for population member `member_idx`: continues
// training that member's underlying model for `steps_per_round` more
// steps using `hyperparams`, and returns its current validation score
// (higher is better). Where the model's state actually lives is the
// caller's business -- see ml/pbt/pbt_bench.cpp, which backs each
// member with a real LinearModel and calls its partial_fit().
using TrainStepFn = std::function<float(int member_idx, const std::vector<float>& hyperparams, int steps_per_round)>;

// The "exploit" half: copies member `src_member_idx`'s trained state
// onto member `dst_member_idx` (e.g. LinearModel::set_weights from
// src's coef()/intercept()).
using CloneStateFn = std::function<void(int dst_member_idx, int src_member_idx)>;

struct PBTMember {
    std::vector<float> hyperparams;
    float score = -1e30f;
};

struct PBTParams {
    int n_rounds               = 10;
    int steps_per_round         = 5;     // training steps between exploit/explore checkpoints
    float exploit_bottom_fraction = 0.25f;  // fraction of the population (by score, worst-first) that exploits each round
    float exploit_top_fraction    = 0.25f;  // fraction (best-first) exploiters copy from
    float perturb_factor_low      = 0.8f;   // explore: each copied hyperparameter is multiplied by Uniform(low, high)
    float perturb_factor_high     = 1.2f;
    unsigned random_state          = 0;
};

struct PBTResult {
    std::vector<PBTMember> final_population;
    std::vector<float> best_score_per_round;  // population-best score after each round -- a convergence curve
    int best_member_idx = 0;
};

PBTResult population_based_training(std::vector<PBTMember> initial_population, const TrainStepFn& train_step, const CloneStateFn& clone_state,
                                     PBTParams params = {});
