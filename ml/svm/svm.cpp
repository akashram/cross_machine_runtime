#include "svm.h"

#include <algorithm>
#include <cmath>
#include <random>

SVM::SVM(SVMParams params) : params_(params) {}

float SVM::kernel(const std::vector<float>& a, const std::vector<float>& b) const {
    switch (params_.kernel) {
        case KernelType::LINEAR: {
            float dot = 0.0f;
            for (std::size_t k = 0; k < a.size(); ++k) dot += a[k] * b[k];
            return dot;
        }
        case KernelType::POLY: {
            float dot = 0.0f;
            for (std::size_t k = 0; k < a.size(); ++k) dot += a[k] * b[k];
            return std::pow(effective_gamma_ * dot + params_.coef0, static_cast<float>(params_.degree));
        }
        case KernelType::RBF:
        default: {
            float sq = 0.0f;
            for (std::size_t k = 0; k < a.size(); ++k) {
                float d = a[k] - b[k];
                sq += d * d;
            }
            return std::exp(-effective_gamma_ * sq);
        }
    }
}

// Platt (1998) simplified SMO: repeatedly scan every alpha_i for a KKT
// violation, pick a random second multiplier alpha_j, solve the resulting
// 2-variable QP subproblem in closed form (clipped to the box + linear
// equality constraint), and update b to keep KKT conditions satisfied at
// the boundary. Stops once `max_passes_without_change` consecutive full
// scans make no update (the standard simplified-SMO stopping rule), or
// after `max_iter` scans, whichever comes first.
void SVM::fit(const Features& X, const Labels& y) {
    int n = static_cast<int>(X.size());
    int n_features = n > 0 ? static_cast<int>(X[0].size()) : 0;
    effective_gamma_ = params_.gamma > 0.0f ? params_.gamma : (n_features > 0 ? 1.0f / static_cast<float>(n_features) : 1.0f);

    std::vector<float> alpha(static_cast<std::size_t>(n), 0.0f);
    float b = 0.0f;

    // Full kernel matrix: O(n^2) memory, avoids recomputing K(i,j) on
    // every SMO scan. Fine at the dataset sizes this gets exercised on.
    std::vector<std::vector<float>> K(static_cast<std::size_t>(n), std::vector<float>(static_cast<std::size_t>(n)));
    for (int i = 0; i < n; ++i)
        for (int j = i; j < n; ++j) {
            float k = kernel(X[static_cast<std::size_t>(i)], X[static_cast<std::size_t>(j)]);
            K[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)] = k;
            K[static_cast<std::size_t>(j)][static_cast<std::size_t>(i)] = k;
        }

    auto decision = [&](int i) {
        float sum = 0.0f;
        for (int k = 0; k < n; ++k)
            sum += alpha[static_cast<std::size_t>(k)] * y[static_cast<std::size_t>(k)] *
                   K[static_cast<std::size_t>(k)][static_cast<std::size_t>(i)];
        return sum + b;
    };

    std::mt19937 rng(0);
    int passes_without_change = 0;
    const int max_passes_without_change = 10;
    int total_passes = 0;

    while (passes_without_change < max_passes_without_change && total_passes < params_.max_iter) {
        int num_changed = 0;
        for (int i = 0; i < n; ++i) {
            float E_i = decision(i) - y[static_cast<std::size_t>(i)];
            bool kkt_violated = (y[static_cast<std::size_t>(i)] * E_i < -params_.tol && alpha[static_cast<std::size_t>(i)] < params_.C) ||
                                 (y[static_cast<std::size_t>(i)] * E_i > params_.tol && alpha[static_cast<std::size_t>(i)] > 0.0f);
            if (!kkt_violated) continue;

            int j = i;
            if (n > 1)
                while (j == i) j = static_cast<int>(rng() % static_cast<unsigned>(n));
            else
                continue;

            float E_j = decision(j) - y[static_cast<std::size_t>(j)];
            float alpha_i_old = alpha[static_cast<std::size_t>(i)];
            float alpha_j_old = alpha[static_cast<std::size_t>(j)];

            float L, H;
            if (y[static_cast<std::size_t>(i)] != y[static_cast<std::size_t>(j)]) {
                L = std::max(0.0f, alpha_j_old - alpha_i_old);
                H = std::min(params_.C, params_.C + alpha_j_old - alpha_i_old);
            } else {
                L = std::max(0.0f, alpha_i_old + alpha_j_old - params_.C);
                H = std::min(params_.C, alpha_i_old + alpha_j_old);
            }
            if (L >= H) continue;

            float eta = 2.0f * K[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)] -
                        K[static_cast<std::size_t>(i)][static_cast<std::size_t>(i)] -
                        K[static_cast<std::size_t>(j)][static_cast<std::size_t>(j)];
            if (eta >= 0.0f) continue;

            float alpha_j_new = alpha_j_old - y[static_cast<std::size_t>(j)] * (E_i - E_j) / eta;
            alpha_j_new = std::clamp(alpha_j_new, L, H);
            if (std::fabs(alpha_j_new - alpha_j_old) < 1e-7f) continue;

            float alpha_i_new = alpha_i_old + y[static_cast<std::size_t>(i)] * y[static_cast<std::size_t>(j)] * (alpha_j_old - alpha_j_new);

            float b1 = b - E_i - y[static_cast<std::size_t>(i)] * (alpha_i_new - alpha_i_old) * K[static_cast<std::size_t>(i)][static_cast<std::size_t>(i)] -
                       y[static_cast<std::size_t>(j)] * (alpha_j_new - alpha_j_old) * K[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)];
            float b2 = b - E_j - y[static_cast<std::size_t>(i)] * (alpha_i_new - alpha_i_old) * K[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)] -
                       y[static_cast<std::size_t>(j)] * (alpha_j_new - alpha_j_old) * K[static_cast<std::size_t>(j)][static_cast<std::size_t>(j)];

            if (alpha_i_new > 0.0f && alpha_i_new < params_.C)
                b = b1;
            else if (alpha_j_new > 0.0f && alpha_j_new < params_.C)
                b = b2;
            else
                b = (b1 + b2) / 2.0f;

            alpha[static_cast<std::size_t>(i)] = alpha_i_new;
            alpha[static_cast<std::size_t>(j)] = alpha_j_new;
            ++num_changed;
        }
        ++total_passes;
        passes_without_change = (num_changed == 0) ? passes_without_change + 1 : 0;
    }

    X_sv_.clear();
    y_sv_.clear();
    alpha_sv_.clear();
    for (int i = 0; i < n; ++i) {
        if (alpha[static_cast<std::size_t>(i)] > 1e-7f) {
            X_sv_.push_back(X[static_cast<std::size_t>(i)]);
            y_sv_.push_back(y[static_cast<std::size_t>(i)]);
            alpha_sv_.push_back(alpha[static_cast<std::size_t>(i)]);
        }
    }
    b_ = b;
}

Labels SVM::predict(const Features& X) const {
    Labels out;
    out.reserve(X.size());
    for (const auto& row : X) {
        float sum = b_;
        for (std::size_t i = 0; i < X_sv_.size(); ++i) sum += alpha_sv_[i] * y_sv_[i] * kernel(X_sv_[i], row);
        out.push_back(sum >= 0.0f ? 1.0f : -1.0f);
    }
    return out;
}

float SVM::score(const Features& X, const Labels& y) const {
    if (y.empty()) return 0.0f;
    Labels pred = predict(X);
    int correct = 0;
    for (std::size_t i = 0; i < y.size(); ++i)
        if (pred[i] == y[i]) ++correct;
    return static_cast<float>(correct) / static_cast<float>(y.size());
}
