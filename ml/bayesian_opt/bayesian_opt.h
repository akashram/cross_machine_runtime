#pragma once
#include <functional>
#include <utility>
#include <vector>

#include "gp.h"

// PLAN.md Phase 12c step 15: Bayesian optimization -- GP surrogate,
// Expected Improvement acquisition, upper confidence bound.
// Benchmarked against random search and grid search.
//
// Design note: the original stub sketched a minimization-oriented,
// named-SearchSpace API ("lower metric is better"). Replaced with a
// maximization-oriented Bounds/ObjectiveFn design instead -- every
// benchmark in this repo (accuracy, purity, explained variance, ...)
// is a higher-is-better metric, and `ml/hyperparam_sensitivity`'s real
// sweeps are exactly the objective this optimizer tunes, so
// maximization keeps this step consistent with the rest of Phase 12
// rather than requiring negation at every call site.
//
// All objectives here are MAXIMIZED.

using ObjectiveFn = std::function<float(const std::vector<float>&)>;
using Bounds = std::vector<std::pair<float, float>>;  // one (lo, hi) pair per dimension
using EvalHistory = std::vector<std::pair<std::vector<float>, float>>;

// Expected Improvement (Mockus 1978), maximization form: `improvement =
// mean - best_so_far - xi`; EI = improvement*Phi(z) + std*phi(z) where
// z = improvement/std. xi trades exploitation (xi=0) for exploration
// (xi>0) by requiring a margin of improvement before rewarding it.
float expected_improvement(float mean, float std_dev, float best_so_far, float xi = 0.01f);

// Upper Confidence Bound: mean + kappa*std -- kappa directly controls
// the exploration/exploitation trade-off (larger kappa explores more).
float upper_confidence_bound(float mean, float std_dev, float kappa = 2.0f);

enum class AcquisitionType { EXPECTED_IMPROVEMENT, UPPER_CONFIDENCE_BOUND };

struct BOParams {
    int n_initial_random  = 5;    // random points evaluated before the GP has any data
    int n_iterations      = 10;   // GP-guided iterations after the initial random ones
    int n_candidates       = 500; // random candidates evaluated per iteration to (approximately) maximize acquisition
    AcquisitionType acquisition = AcquisitionType::EXPECTED_IMPROVEMENT;
    float xi               = 0.01f;
    float kappa            = 2.0f;
    unsigned random_state   = 0;
    GPParams gp_params;
};

class BayesianOptimizer {
public:
    BayesianOptimizer(Bounds bounds, BOParams params = {});

    // Runs the full initial-random + GP-guided budget, returns the best
    // (point, value) found.
    std::pair<std::vector<float>, float> optimize(const ObjectiveFn& objective);

    const EvalHistory& history() const { return history_; }

private:
    Bounds bounds_;
    BOParams params_;
    EvalHistory history_;
};

// Baselines step 15 is benchmarked against.
std::pair<std::vector<float>, float> random_search(const Bounds& bounds, int n_evaluations, const ObjectiveFn& objective,
                                                    unsigned random_state, EvalHistory* history_out = nullptr);

// N-dimensional grid search: the full Cartesian product of
// n_points_per_dim evenly-spaced values per dimension (so total
// evaluations = n_points_per_dim ^ bounds.size() -- keep both small for
// dimensions > 1).
std::pair<std::vector<float>, float> grid_search(const Bounds& bounds, int n_points_per_dim, const ObjectiveFn& objective,
                                                  EvalHistory* history_out = nullptr);
