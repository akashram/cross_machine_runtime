#include "linear_model.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>

namespace {

// Numerically stable logistic sigmoid -- avoids overflow in exp() for
// large-magnitude negative inputs.
float sigmoid(float z) {
    if (z >= 0.0f) return 1.0f / (1.0f + std::exp(-z));
    float e = std::exp(z);
    return e / (1.0f + e);
}

}  // namespace

LinearModel::LinearModel(LinearModelParams params) : params_(params) {}

float LinearModel::predict_raw(const std::vector<float>& x, const std::vector<float>& w, float b) const {
    float sum = b;
    for (std::size_t j = 0; j < x.size(); ++j) sum += w[j] * x[j];
    return sum;
}

void LinearModel::fit(const Features& X, const Labels& y) {
    weights_.assign(X[0].size(), 0.0f);
    bias_ = 0.0f;
    if (params_.optimizer == LinearOptimizer::SGD)
        fit_sgd(X, y);
    else
        fit_lbfgs(X, y);
}

// Proximal (ISTA-style) elastic-net SGD: each step takes an ordinary
// gradient step on the smooth part (data loss + L2 term), then applies
// soft-thresholding for the L1 term -- the standard way to handle SGD
// with a non-differentiable-at-zero L1 penalty (what sklearn's
// SGDClassifier/Regressor do under penalty='elasticnet').
void LinearModel::fit_sgd(const Features& X, const Labels& y) {
    std::size_t n = X.size(), d = X[0].size();
    std::vector<std::size_t> order(n);
    std::iota(order.begin(), order.end(), 0);
    std::mt19937 rng(params_.random_state);

    float l2 = params_.alpha * (1.0f - params_.l1_ratio);
    float l1 = params_.alpha * params_.l1_ratio;

    int step = 0;
    for (int epoch = 0; epoch < params_.max_iter; ++epoch) {
        std::shuffle(order.begin(), order.end(), rng);
        for (std::size_t idx : order) {
            float eta = params_.learning_rate / (1.0f + 0.0001f * static_cast<float>(step));
            ++step;

            const auto& x = X[idx];
            float raw = predict_raw(x, weights_, bias_);
            float error;
            if (params_.loss == LinearLoss::LOGISTIC)
                error = sigmoid(raw) - y[idx];  // d/dz of log-loss
            else
                error = raw - y[idx];  // d/dz of (1/2)(raw - y)^2

            for (std::size_t j = 0; j < d; ++j) {
                float grad = error * x[j] + l2 * weights_[j];
                weights_[j] -= eta * grad;
            }
            bias_ -= eta * error;  // bias is never regularized

            if (l1 > 0.0f) {
                float shrink = eta * l1;
                for (std::size_t j = 0; j < d; ++j) {
                    float w = weights_[j];
                    weights_[j] = (w > 0.0f) ? std::max(0.0f, w - shrink) : std::min(0.0f, w + shrink);
                }
            }
        }
    }
}

float LinearModel::loss_and_gradient(const Features& X, const Labels& y, const std::vector<float>& w, float b,
                                      std::vector<float>& grad_w, float& grad_b) const {
    std::size_t n = X.size(), d = w.size();
    float l2 = params_.alpha * (1.0f - params_.l1_ratio);

    grad_w.assign(d, 0.0f);
    grad_b = 0.0f;
    float loss = 0.0f;

    for (std::size_t i = 0; i < n; ++i) {
        float raw = predict_raw(X[i], w, b);
        float error;
        if (params_.loss == LinearLoss::LOGISTIC) {
            float p = sigmoid(raw);
            float p_clamped = std::clamp(p, 1e-7f, 1.0f - 1e-7f);
            loss -= y[i] * std::log(p_clamped) + (1.0f - y[i]) * std::log(1.0f - p_clamped);
            error = p - y[i];
        } else {
            float diff = raw - y[i];
            loss += 0.5f * diff * diff;
            error = diff;
        }
        for (std::size_t j = 0; j < d; ++j) grad_w[j] += error * X[i][j];
        grad_b += error;
    }

    loss /= static_cast<float>(n);
    grad_b /= static_cast<float>(n);
    for (std::size_t j = 0; j < d; ++j) grad_w[j] = grad_w[j] / static_cast<float>(n) + l2 * w[j];
    for (std::size_t j = 0; j < d; ++j) loss += 0.5f * l2 * w[j] * w[j];

    return loss;
}

// L-BFGS (Nocedal & Wright, Algorithm 7.4/7.5): two-loop recursion over
// the last `lbfgs_memory` (s, y) correction pairs to approximate the
// inverse Hessian-vector product, plus backtracking Armijo line search.
// L2-only regularization (see header): a non-differentiable L1 term
// would break the smoothness the quasi-Newton curvature estimate
// assumes, so LBFGS always regularizes with alpha*(1-l1_ratio) as an L2
// penalty regardless of l1_ratio -- a real, documented scope limit.
void LinearModel::fit_lbfgs(const Features& X, const Labels& y) {
    std::size_t d = weights_.size();
    std::vector<float> w = weights_;
    float b = bias_;

    std::vector<float> grad;
    float grad_b;
    float loss = loss_and_gradient(X, y, w, b, grad, grad_b);

    // Combined (weights, bias) vector so the bias participates in the
    // same quasi-Newton update as the weights.
    std::vector<float> x_vec(d + 1);
    for (std::size_t j = 0; j < d; ++j) x_vec[j] = w[j];
    x_vec[d] = b;
    std::vector<float> g_vec(d + 1);
    for (std::size_t j = 0; j < d; ++j) g_vec[j] = grad[j];
    g_vec[d] = grad_b;

    std::vector<std::vector<float>> s_history, y_history;
    std::vector<float> rho_history;

    for (int iter = 0; iter < params_.max_iter; ++iter) {
        float grad_norm = 0.0f;
        for (float g : g_vec) grad_norm += g * g;
        grad_norm = std::sqrt(grad_norm);
        if (grad_norm < params_.tol) break;

        // Two-loop recursion.
        std::vector<float> q = g_vec;
        std::size_t m = s_history.size();
        std::vector<float> alpha_hist(m);
        for (std::size_t idx = m; idx-- > 0;) {
            float dot_sq = 0.0f;
            for (std::size_t j = 0; j < q.size(); ++j) dot_sq += s_history[idx][j] * q[j];
            alpha_hist[idx] = rho_history[idx] * dot_sq;
            for (std::size_t j = 0; j < q.size(); ++j) q[j] -= alpha_hist[idx] * y_history[idx][j];
        }
        float gamma = 1.0f;
        if (m > 0) {
            float sy = 0.0f, yy = 0.0f;
            for (std::size_t j = 0; j < q.size(); ++j) {
                sy += s_history[m - 1][j] * y_history[m - 1][j];
                yy += y_history[m - 1][j] * y_history[m - 1][j];
            }
            if (yy > 1e-12f) gamma = sy / yy;
        }
        std::vector<float> r(q.size());
        for (std::size_t j = 0; j < q.size(); ++j) r[j] = gamma * q[j];
        for (std::size_t idx = 0; idx < m; ++idx) {
            float dot_yr = 0.0f;
            for (std::size_t j = 0; j < r.size(); ++j) dot_yr += y_history[idx][j] * r[j];
            float beta = rho_history[idx] * dot_yr;
            for (std::size_t j = 0; j < r.size(); ++j) r[j] += s_history[idx][j] * (alpha_hist[idx] - beta);
        }
        std::vector<float> direction(r.size());
        for (std::size_t j = 0; j < r.size(); ++j) direction[j] = -r[j];

        // Backtracking Armijo line search.
        float directional_deriv = 0.0f;
        for (std::size_t j = 0; j < direction.size(); ++j) directional_deriv += g_vec[j] * direction[j];

        float step = 1.0f;
        std::vector<float> new_x(x_vec.size());
        float new_loss = loss;
        const float c1 = 1e-4f;
        for (int ls = 0; ls < 50; ++ls) {
            for (std::size_t j = 0; j < x_vec.size(); ++j) new_x[j] = x_vec[j] + step * direction[j];
            std::vector<float> new_w(new_x.begin(), new_x.begin() + static_cast<long>(d));
            std::vector<float> dummy_grad;
            float dummy_grad_b;
            new_loss = loss_and_gradient(X, y, new_w, new_x[d], dummy_grad, dummy_grad_b);
            if (new_loss <= loss + c1 * step * directional_deriv) break;
            step *= 0.5f;
        }

        std::vector<float> new_w(new_x.begin(), new_x.begin() + static_cast<long>(d));
        std::vector<float> new_grad;
        float new_grad_b;
        loss_and_gradient(X, y, new_w, new_x[d], new_grad, new_grad_b);
        std::vector<float> new_g_vec(d + 1);
        for (std::size_t j = 0; j < d; ++j) new_g_vec[j] = new_grad[j];
        new_g_vec[d] = new_grad_b;

        std::vector<float> s_k(x_vec.size()), y_k(x_vec.size());
        for (std::size_t j = 0; j < x_vec.size(); ++j) {
            s_k[j] = new_x[j] - x_vec[j];
            y_k[j] = new_g_vec[j] - g_vec[j];
        }
        float sy = 0.0f;
        for (std::size_t j = 0; j < s_k.size(); ++j) sy += s_k[j] * y_k[j];
        if (sy > 1e-10f) {
            s_history.push_back(s_k);
            y_history.push_back(y_k);
            rho_history.push_back(1.0f / sy);
            if (static_cast<int>(s_history.size()) > params_.lbfgs_memory) {
                s_history.erase(s_history.begin());
                y_history.erase(y_history.begin());
                rho_history.erase(rho_history.begin());
            }
        }

        x_vec = new_x;
        g_vec = new_g_vec;
        loss = new_loss;
    }

    weights_.assign(x_vec.begin(), x_vec.begin() + static_cast<long>(d));
    bias_ = x_vec[d];
}

Labels LinearModel::predict(const Features& X) const {
    Labels out;
    out.reserve(X.size());
    for (const auto& row : X) {
        float raw = predict_raw(row, weights_, bias_);
        out.push_back(params_.loss == LinearLoss::LOGISTIC ? (sigmoid(raw) >= 0.5f ? 1.0f : 0.0f) : raw);
    }
    return out;
}

std::vector<float> LinearModel::predict_proba(const Features& X) const {
    std::vector<float> out;
    out.reserve(X.size());
    for (const auto& row : X) out.push_back(sigmoid(predict_raw(row, weights_, bias_)));
    return out;
}

float LinearModel::score(const Features& X, const Labels& y) const {
    if (y.empty()) return 0.0f;
    Labels pred = predict(X);

    if (params_.loss == LinearLoss::LOGISTIC) {
        int correct = 0;
        for (std::size_t i = 0; i < y.size(); ++i)
            if (pred[i] == y[i]) ++correct;
        return static_cast<float>(correct) / static_cast<float>(y.size());
    }

    float mean = 0.0f;
    for (float v : y) mean += v;
    mean /= static_cast<float>(y.size());
    float ss_tot = 0.0f, ss_res = 0.0f;
    for (std::size_t i = 0; i < y.size(); ++i) {
        ss_tot += (y[i] - mean) * (y[i] - mean);
        ss_res += (y[i] - pred[i]) * (y[i] - pred[i]);
    }
    return ss_tot > 1e-12f ? 1.0f - ss_res / ss_tot : 0.0f;
}
