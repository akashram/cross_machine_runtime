#pragma once
#include <functional>
#include <random>
#include <vector>

// PLAN.md Phase 12c step 17: Hyperband / ASHA -- successive halving,
// early stopping of unpromising configs, async version for parallel
// workers.

using Config = std::vector<float>;
using ConfigSampler = std::function<Config(std::mt19937&)>;

// Evaluates a config at a given resource level (e.g. number of training
// rounds). Higher-is-better, matching this repo's other Phase 12c
// optimizers.
using ResourceEvalFn = std::function<float(const Config&, int resource)>;

struct EvalRecord {
    Config config;
    int resource;
    float score;
};

struct SHAResult {
    Config best_config;
    float best_score = -1e30f;
    std::vector<EvalRecord> history;  // every (config, resource, score) evaluation actually performed -- sum of `resource` here is the real total training cost paid
};

// Successive Halving (Jamieson & Talwalkar 2016): evaluate all configs
// at `min_resource`, keep the top 1/eta fraction, multiply resource by
// eta, repeat until `max_resource` is reached or one config remains.
SHAResult successive_halving(const std::vector<Config>& configs, int min_resource, int max_resource, float eta, const ResourceEvalFn& evaluate);

// Hyperband (Li et al. 2016): runs multiple Successive Halving
// brackets with different (n_configs, initial_resource) trade-offs --
// hedges against not knowing in advance whether more configs or more
// resource-per-config is the better use of a fixed budget.
SHAResult hyperband(const ConfigSampler& sampler, int max_resource, float eta, const ResourceEvalFn& evaluate, unsigned random_state);

// ASHA (Li et al. 2018): the asynchronous version -- no synchronization
// barrier waiting for an entire rung to finish before promoting anyone.
// Simulated single-threaded here (see README's honest scope note: this
// captures ASHA's real promotion-eligibility logic -- promote as soon
// as a config ranks in the top 1/eta of its rung's results *so far* --
// not literal multi-worker wall-clock parallelism, which needs real
// parallel hardware this benchmark doesn't claim to measure).
SHAResult asha(const ConfigSampler& sampler, int min_resource, int max_resource, float eta, int n_total_configs, const ResourceEvalFn& evaluate,
                unsigned random_state);
