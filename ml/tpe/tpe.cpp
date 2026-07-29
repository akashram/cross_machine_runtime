#include "tpe.h"

#include <algorithm>
#include <cmath>
#include <numeric>

Parzen1D build_parzen1d(const std::vector<float>& values, float lo, float hi) {
    Parzen1D p;
    std::vector<float> sorted = values;
    std::sort(sorted.begin(), sorted.end());
    std::size_t n = sorted.size();

    float range = hi - lo;
    float min_bw = std::max(range * 0.01f, 1e-6f);
    float max_bw = std::max(range, min_bw);

    p.points = sorted;
    p.bandwidths.resize(n);
    for (std::size_t i = 0; i < n; ++i) {
        float bw;
        if (n == 1)
            bw = range;
        else if (i == 0)
            bw = sorted[i + 1] - sorted[i];
        else if (i == n - 1)
            bw = sorted[i] - sorted[i - 1];
        else
            bw = std::max(sorted[i] - sorted[i - 1], sorted[i + 1] - sorted[i]);
        p.bandwidths[i] = std::clamp(bw, min_bw, max_bw);
    }
    return p;
}

float parzen1d_density(const Parzen1D& p, float x) {
    if (p.points.empty()) return 1e-6f;  // no observations in this group: treat as a small uniform-ish density
    float sum = 0.0f;
    for (std::size_t i = 0; i < p.points.size(); ++i) {
        float d = x - p.points[i];
        float bw = p.bandwidths[i];
        sum += std::exp(-0.5f * (d * d) / (bw * bw)) / (bw * std::sqrt(2.0f * static_cast<float>(M_PI)));
    }
    return sum / static_cast<float>(p.points.size());
}

float parzen1d_sample(const Parzen1D& p, std::mt19937& rng, float lo, float hi) {
    if (p.points.empty()) {
        std::uniform_real_distribution<float> dist(lo, hi);
        return dist(rng);
    }
    std::uniform_int_distribution<std::size_t> pick(0, p.points.size() - 1);
    std::size_t idx = pick(rng);
    std::normal_distribution<float> gauss(p.points[idx], p.bandwidths[idx]);
    return std::clamp(gauss(rng), lo, hi);
}

TPEOptimizer::TPEOptimizer(Bounds bounds, TPEParams params) : bounds_(std::move(bounds)), params_(params) {}

std::pair<std::vector<float>, float> TPEOptimizer::optimize(const ObjectiveFn& objective) {
    std::mt19937 rng(params_.random_state);
    history_.clear();
    std::size_t d = bounds_.size();

    auto random_point = [&]() {
        std::vector<float> point(d);
        for (std::size_t i = 0; i < d; ++i) {
            std::uniform_real_distribution<float> dist(bounds_[i].first, bounds_[i].second);
            point[i] = dist(rng);
        }
        return point;
    };

    for (int i = 0; i < params_.n_initial_random; ++i) {
        std::vector<float> x = random_point();
        history_.emplace_back(x, objective(x));
    }

    for (int iter = 0; iter < params_.n_iterations; ++iter) {
        std::vector<std::size_t> order(history_.size());
        std::iota(order.begin(), order.end(), 0);
        std::sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) { return history_[a].second > history_[b].second; });

        std::size_t n_good = std::max<std::size_t>(1, static_cast<std::size_t>(params_.gamma * static_cast<float>(history_.size())));
        if (history_.size() > 1) n_good = std::min(n_good, history_.size() - 1);  // keep at least 1 "bad" point when possible

        std::vector<Parzen1D> good_parzens(d), bad_parzens(d);
        for (std::size_t dim = 0; dim < d; ++dim) {
            std::vector<float> good_vals, bad_vals;
            for (std::size_t k = 0; k < order.size(); ++k) {
                float v = history_[order[k]].first[dim];
                if (k < n_good)
                    good_vals.push_back(v);
                else
                    bad_vals.push_back(v);
            }
            good_parzens[dim] = build_parzen1d(good_vals, bounds_[dim].first, bounds_[dim].second);
            bad_parzens[dim] = build_parzen1d(bad_vals, bounds_[dim].first, bounds_[dim].second);
        }

        // Sample candidates from l(x) (the "good" density) and score by
        // l(x)/g(x) -- Bergstra et al.'s result that this ratio is
        // (up to a monotone transform) equivalent to Expected
        // Improvement under the TPE model, without needing a joint GP.
        std::vector<float> best_candidate;
        float best_score = -1.0f;
        for (int c = 0; c < params_.n_candidates; ++c) {
            std::vector<float> candidate(d);
            for (std::size_t dim = 0; dim < d; ++dim) candidate[dim] = parzen1d_sample(good_parzens[dim], rng, bounds_[dim].first, bounds_[dim].second);

            float l = 1.0f, g = 1.0f;
            for (std::size_t dim = 0; dim < d; ++dim) {
                l *= parzen1d_density(good_parzens[dim], candidate[dim]);
                g *= parzen1d_density(bad_parzens[dim], candidate[dim]);
            }
            float score = l / std::max(g, 1e-12f);
            if (score > best_score) {
                best_score = score;
                best_candidate = candidate;
            }
        }

        history_.emplace_back(best_candidate, objective(best_candidate));
    }

    auto best_it = std::max_element(history_.begin(), history_.end(),
                                     [](const auto& a, const auto& b) { return a.second < b.second; });
    return *best_it;
}
