#include "gp.h"

#include <algorithm>
#include <cmath>

GaussianProcess::GaussianProcess(GPParams params) : params_(params) {}

double GaussianProcess::kernel(const std::vector<float>& a, const std::vector<float>& b) const {
    double sq_dist = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        double d = static_cast<double>(a[i]) - static_cast<double>(b[i]);
        sq_dist += d * d;
    }
    double ls = static_cast<double>(params_.length_scale);
    return static_cast<double>(params_.signal_variance) * std::exp(-sq_dist / (2.0 * ls * ls));
}

void GaussianProcess::fit(const std::vector<std::vector<float>>& X, const std::vector<float>& y) {
    X_ = X;
    std::size_t n = X.size();

    double sum = 0.0;
    for (float v : y) sum += static_cast<double>(v);
    mean_y_ = static_cast<float>(sum / static_cast<double>(n));

    std::vector<std::vector<double>> K(n, std::vector<double>(n));
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j) K[i][j] = kernel(X[i], X[j]) + (i == j ? static_cast<double>(params_.noise_variance) : 0.0);

    // Cholesky decomposition: K = L L^T, L lower-triangular.
    L_.assign(n, std::vector<double>(n, 0.0));
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j <= i; ++j) {
            double s = K[i][j];
            for (std::size_t k = 0; k < j; ++k) s -= L_[i][k] * L_[j][k];
            if (i == j)
                L_[i][j] = std::sqrt(std::max(s, 1e-12));
            else
                L_[i][j] = s / L_[j][j];
        }
    }

    // Solve L z = (y - mean_y) by forward substitution, then L^T alpha = z
    // by back substitution -- alpha = K^{-1}(y - mean_y), the GPML
    // Algorithm 2.1 approach (avoids explicitly inverting K).
    std::vector<double> b(n), z(n);
    for (std::size_t i = 0; i < n; ++i) b[i] = static_cast<double>(y[i]) - static_cast<double>(mean_y_);
    for (std::size_t i = 0; i < n; ++i) {
        double s = b[i];
        for (std::size_t k = 0; k < i; ++k) s -= L_[i][k] * z[k];
        z[i] = s / L_[i][i];
    }
    alpha_.assign(n, 0.0);
    for (std::size_t idx = n; idx-- > 0;) {
        double s = z[idx];
        for (std::size_t k = idx + 1; k < n; ++k) s -= L_[k][idx] * alpha_[k];
        alpha_[idx] = s / L_[idx][idx];
    }
}

void GaussianProcess::predict(const std::vector<float>& x, float& mean, float& std_dev) const {
    std::size_t n = X_.size();
    std::vector<double> k_star(n);
    for (std::size_t i = 0; i < n; ++i) k_star[i] = kernel(x, X_[i]);

    double mean_d = static_cast<double>(mean_y_);
    for (std::size_t i = 0; i < n; ++i) mean_d += k_star[i] * alpha_[i];
    mean = static_cast<float>(mean_d);

    // v = L^{-1} k_star (forward substitution); predictive variance is
    // k(x,x) - v.v (GPML Algorithm 2.1).
    std::vector<double> v(n);
    for (std::size_t i = 0; i < n; ++i) {
        double s = k_star[i];
        for (std::size_t k = 0; k < i; ++k) s -= L_[i][k] * v[k];
        v[i] = s / L_[i][i];
    }
    double variance = kernel(x, x);
    for (double vi : v) variance -= vi * vi;
    std_dev = static_cast<float>(std::sqrt(std::max(variance, 0.0)));
}
