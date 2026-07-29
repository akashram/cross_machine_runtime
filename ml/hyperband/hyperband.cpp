#include "hyperband.h"

#include <algorithm>
#include <cmath>

SHAResult successive_halving(const std::vector<Config>& configs, int min_resource, int max_resource, float eta, const ResourceEvalFn& evaluate) {
    SHAResult result;
    std::vector<Config> current = configs;
    int resource = min_resource;

    while (true) {
        std::vector<std::pair<Config, float>> scored;
        scored.reserve(current.size());
        for (const auto& c : current) {
            float s = evaluate(c, resource);
            result.history.push_back({c, resource, s});
            scored.emplace_back(c, s);
        }
        std::sort(scored.begin(), scored.end(), [](const auto& a, const auto& b) { return a.second > b.second; });

        if (resource >= max_resource || current.size() <= 1) {
            result.best_config = scored.front().first;
            result.best_score = scored.front().second;
            break;
        }

        std::size_t n_keep = std::max<std::size_t>(1, current.size() / static_cast<std::size_t>(eta));
        current.clear();
        for (std::size_t i = 0; i < n_keep; ++i) current.push_back(scored[i].first);
        resource = std::min(static_cast<int>(static_cast<float>(resource) * eta), max_resource);
    }
    return result;
}

SHAResult hyperband(const ConfigSampler& sampler, int max_resource, float eta, const ResourceEvalFn& evaluate, unsigned random_state) {
    std::mt19937 rng(random_state);
    int R = max_resource;
    int s_max = static_cast<int>(std::floor(std::log(static_cast<float>(R)) / std::log(eta)));

    SHAResult overall;
    for (int s = s_max; s >= 0; --s) {
        int n = static_cast<int>(std::ceil((static_cast<float>(s_max + 1) / static_cast<float>(s + 1)) * std::pow(eta, static_cast<float>(s))));
        int r = std::max(1, static_cast<int>(static_cast<float>(R) * std::pow(eta, static_cast<float>(-s))));

        std::vector<Config> configs;
        configs.reserve(static_cast<std::size_t>(n));
        for (int i = 0; i < n; ++i) configs.push_back(sampler(rng));

        SHAResult bracket = successive_halving(configs, r, R, eta, evaluate);
        overall.history.insert(overall.history.end(), bracket.history.begin(), bracket.history.end());
        if (bracket.best_score > overall.best_score) {
            overall.best_score = bracket.best_score;
            overall.best_config = bracket.best_config;
        }
    }
    return overall;
}

SHAResult asha(const ConfigSampler& sampler, int min_resource, int max_resource, float eta, int n_total_configs, const ResourceEvalFn& evaluate,
                unsigned random_state) {
    std::mt19937 rng(random_state);

    std::vector<int> rung_resources;
    int r = min_resource;
    while (true) {
        rung_resources.push_back(r);
        if (r >= max_resource) break;
        r = std::min(static_cast<int>(static_cast<float>(r) * eta), max_resource);
    }
    std::size_t n_rungs = rung_resources.size();

    std::vector<std::vector<std::pair<Config, float>>> rung_results(n_rungs);
    std::vector<std::vector<bool>> promoted(n_rungs);

    SHAResult result;
    int configs_started = 0;

    while (true) {
        // Try to promote: scan rungs from highest-promotable down to the
        // base rung, promote the first eligible (not-yet-promoted, ranks
        // in the top 1/eta of its rung's results so far) config found --
        // this is what makes ASHA asynchronous: a config can be promoted
        // the moment it's eligible, without waiting for every other
        // config at its rung to finish.
        bool promoted_something = false;
        for (std::size_t rung = 0; rung + 1 < n_rungs && !promoted_something; ++rung) {
            auto& results = rung_results[rung];
            auto& prom = promoted[rung];
            // A rung only has enough information to determine a
            // meaningful top-1/eta cutoff once at least eta configs
            // have been evaluated there -- with fewer, every result
            // would trivially count as "top 1/eta" and promote itself
            // immediately, defeating the point of successive halving.
            // A real bug found via hyperband_test.cpp's hand-computable
            // ASHA scenario: without this guard, a single early result
            // promoted itself right away.
            if (results.size() < static_cast<std::size_t>(eta)) continue;

            std::vector<float> scores;
            for (const auto& rp : results) scores.push_back(rp.second);
            std::vector<float> sorted_scores = scores;
            std::sort(sorted_scores.begin(), sorted_scores.end(), std::greater<float>());
            std::size_t top_k = std::max<std::size_t>(1, results.size() / static_cast<std::size_t>(eta));
            float threshold = sorted_scores[top_k - 1];

            for (std::size_t i = 0; i < results.size(); ++i) {
                if (prom[i]) continue;
                if (results[i].second >= threshold) {
                    prom[i] = true;
                    int next_resource = rung_resources[rung + 1];
                    float s = evaluate(results[i].first, next_resource);
                    result.history.push_back({results[i].first, next_resource, s});
                    rung_results[rung + 1].emplace_back(results[i].first, s);
                    promoted[rung + 1].push_back(false);
                    promoted_something = true;
                    break;
                }
            }
        }

        if (promoted_something) continue;

        if (configs_started >= n_total_configs) break;  // nothing left to promote and no new configs to start

        Config c = sampler(rng);
        float s = evaluate(c, rung_resources[0]);
        result.history.push_back({c, rung_resources[0], s});
        rung_results[0].emplace_back(c, s);
        promoted[0].push_back(false);
        ++configs_started;
    }

    for (const auto& rec : result.history) {
        if (rec.score > result.best_score) {
            result.best_score = rec.score;
            result.best_config = rec.config;
        }
    }
    return result;
}
