#pragma once
#include <vector>
#include <functional>
// TODO: implement GP surrogate + Expected Improvement acquisition function

struct SearchSpace {
    struct Param {
        std::string name;
        float       lo, hi;       // continuous range
        bool        log_scale;    // search in log space
    };
    std::vector<Param> params;
};

class BayesianOptimizer {
public:
    explicit BayesianOptimizer(int random_seed = 42, float xi = 0.01f);

    // Suggest next parameters to evaluate (via Expected Improvement)
    std::vector<float> suggest(const SearchSpace& space);

    // Record an observation (lower metric is better)
    void observe(const std::vector<float>& params, double metric);

    // Return best parameters seen so far
    std::vector<float> best_params() const;
    double             best_metric() const;
};
