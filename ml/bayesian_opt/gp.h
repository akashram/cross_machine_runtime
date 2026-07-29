#pragma once
#include <vector>

// PLAN.md Phase 12c step 15: Gaussian Process surrogate (Rasmussen &
// Williams, GPML Algorithm 2.1) -- RBF kernel, Cholesky-based exact
// inference. Internal linear algebra runs in double (see pca.cpp's own
// note on why: Cholesky factorization is numerically sensitive the same
// way randomized SVD is), public API is float.

struct GPParams {
    float length_scale    = 1.0f;   // RBF kernel length scale
    float signal_variance = 1.0f;   // RBF kernel output (amplitude) variance
    float noise_variance  = 1e-6f;  // observation noise / Cholesky jitter
};

class GaussianProcess {
public:
    explicit GaussianProcess(GPParams params = {});

    // X: observed points (n x d). y: observed (noisy) function values.
    void fit(const std::vector<std::vector<float>>& X, const std::vector<float>& y);

    // Posterior predictive mean and standard deviation at a query point.
    void predict(const std::vector<float>& x, float& mean, float& std_dev) const;

private:
    GPParams params_;
    std::vector<std::vector<float>> X_;
    float mean_y_ = 0.0f;
    std::vector<std::vector<double>> L_;  // Cholesky factor of (K + noise*I)
    std::vector<double> alpha_;           // K^{-1} (y - mean_y), via forward/back substitution

    double kernel(const std::vector<float>& a, const std::vector<float>& b) const;
};
