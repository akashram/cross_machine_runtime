#include "bayesian_opt.h"

#include <algorithm>
#include <cmath>
#include <random>

namespace {

float normal_cdf(float z) { return 0.5f * (1.0f + std::erff(z / std::sqrt(2.0f))); }
float normal_pdf(float z) { return static_cast<float>(1.0 / std::sqrt(2.0 * M_PI)) * std::exp(-0.5f * z * z); }

std::vector<float> random_point(const Bounds& bounds, std::mt19937& rng) {
    std::vector<float> point(bounds.size());
    for (std::size_t d = 0; d < bounds.size(); ++d) {
        std::uniform_real_distribution<float> dist(bounds[d].first, bounds[d].second);
        point[d] = dist(rng);
    }
    return point;
}

}  // namespace

float expected_improvement(float mean, float std_dev, float best_so_far, float xi) {
    float improvement = mean - best_so_far - xi;
    if (std_dev < 1e-9f) return improvement > 0.0f ? improvement : 0.0f;
    float z = improvement / std_dev;
    return improvement * normal_cdf(z) + std_dev * normal_pdf(z);
}

float upper_confidence_bound(float mean, float std_dev, float kappa) { return mean + kappa * std_dev; }

BayesianOptimizer::BayesianOptimizer(Bounds bounds, BOParams params) : bounds_(std::move(bounds)), params_(params) {}

std::pair<std::vector<float>, float> BayesianOptimizer::optimize(const ObjectiveFn& objective) {
    std::mt19937 rng(params_.random_state);
    history_.clear();

    for (int i = 0; i < params_.n_initial_random; ++i) {
        std::vector<float> x = random_point(bounds_, rng);
        history_.emplace_back(x, objective(x));
    }

    for (int iter = 0; iter < params_.n_iterations; ++iter) {
        std::vector<std::vector<float>> X;
        std::vector<float> y;
        for (const auto& [point, value] : history_) {
            X.push_back(point);
            y.push_back(value);
        }
        GaussianProcess gp(params_.gp_params);
        gp.fit(X, y);

        float best_so_far = *std::max_element(y.begin(), y.end());

        std::vector<float> best_candidate;
        float best_acquisition = -1e30f;
        for (int c = 0; c < params_.n_candidates; ++c) {
            std::vector<float> candidate = random_point(bounds_, rng);
            float mean, std_dev;
            gp.predict(candidate, mean, std_dev);
            float acq = params_.acquisition == AcquisitionType::EXPECTED_IMPROVEMENT
                            ? expected_improvement(mean, std_dev, best_so_far, params_.xi)
                            : upper_confidence_bound(mean, std_dev, params_.kappa);
            if (acq > best_acquisition) {
                best_acquisition = acq;
                best_candidate = candidate;
            }
        }

        history_.emplace_back(best_candidate, objective(best_candidate));
    }

    auto best_it = std::max_element(history_.begin(), history_.end(),
                                     [](const auto& a, const auto& b) { return a.second < b.second; });
    return *best_it;
}

std::pair<std::vector<float>, float> random_search(const Bounds& bounds, int n_evaluations, const ObjectiveFn& objective,
                                                    unsigned random_state, EvalHistory* history_out) {
    std::mt19937 rng(random_state);
    EvalHistory history;
    for (int i = 0; i < n_evaluations; ++i) {
        std::vector<float> x = random_point(bounds, rng);
        history.emplace_back(x, objective(x));
    }
    if (history_out) *history_out = history;
    auto best_it = std::max_element(history.begin(), history.end(), [](const auto& a, const auto& b) { return a.second < b.second; });
    return *best_it;
}

namespace {

void grid_search_recurse(const Bounds& bounds, int n_points_per_dim, std::size_t dim, std::vector<float>& point,
                          const ObjectiveFn& objective, EvalHistory& history) {
    if (dim == bounds.size()) {
        history.emplace_back(point, objective(point));
        return;
    }
    for (int i = 0; i < n_points_per_dim; ++i) {
        float t = n_points_per_dim > 1 ? static_cast<float>(i) / static_cast<float>(n_points_per_dim - 1) : 0.0f;
        point[dim] = bounds[dim].first + t * (bounds[dim].second - bounds[dim].first);
        grid_search_recurse(bounds, n_points_per_dim, dim + 1, point, objective, history);
    }
}

}  // namespace

std::pair<std::vector<float>, float> grid_search(const Bounds& bounds, int n_points_per_dim, const ObjectiveFn& objective,
                                                  EvalHistory* history_out) {
    EvalHistory history;
    std::vector<float> point(bounds.size());
    grid_search_recurse(bounds, n_points_per_dim, 0, point, objective, history);
    if (history_out) *history_out = history;
    auto best_it = std::max_element(history.begin(), history.end(), [](const auto& a, const auto& b) { return a.second < b.second; });
    return *best_it;
}
