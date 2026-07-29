#pragma once
#include <functional>
#include <random>
#include <utility>
#include <vector>

// PLAN.md Phase 12c step 16: Tree-structured Parzen Estimator (Bergstra
// et al. 2011, the Hyperopt/Optuna default) -- models p(x|good) and
// p(x|bad) separately via per-dimension Parzen-window (Gaussian KDE)
// density estimates, rather than bayesian_opt's single joint GP
// surrogate. More scalable than GP-based BO: each iteration's cost is
// linear in the number of observations (rebuilding a 1D KDE per
// dimension) rather than the GP's O(n^3) Cholesky factorization.
//
// All objectives are MAXIMIZED, matching ml/bayesian_opt's convention.

using ObjectiveFn = std::function<float(const std::vector<float>&)>;
using Bounds = std::vector<std::pair<float, float>>;
using EvalHistory = std::vector<std::pair<std::vector<float>, float>>;

// A 1D Parzen-window density estimate: a mixture of Gaussians, one per
// observed point, with the Bergstra et al. neighbor-distance bandwidth
// heuristic (each point's bandwidth is the larger of its gaps to its
// left/right neighbors, clipped to [min_bw, max_bw]) -- avoids both
// degenerate zero-bandwidth spikes and over-smoothing across the whole
// range.
struct Parzen1D {
    std::vector<float> points;
    std::vector<float> bandwidths;
};

Parzen1D build_parzen1d(const std::vector<float>& values, float lo, float hi);
float parzen1d_density(const Parzen1D& p, float x);
float parzen1d_sample(const Parzen1D& p, std::mt19937& rng, float lo, float hi);

struct TPEParams {
    int n_initial_random = 5;    // random points evaluated before any (good, bad) split exists
    int n_iterations     = 10;   // TPE-guided iterations after the initial random ones
    float gamma            = 0.25f;  // fraction of observations (by value, best-first) treated as "good"
    int n_candidates       = 100; // candidates sampled from l(x) per iteration, scored by l(x)/g(x)
    unsigned random_state   = 0;
};

class TPEOptimizer {
public:
    TPEOptimizer(Bounds bounds, TPEParams params = {});

    std::pair<std::vector<float>, float> optimize(const ObjectiveFn& objective);
    const EvalHistory& history() const { return history_; }

private:
    Bounds bounds_;
    TPEParams params_;
    EvalHistory history_;
};
