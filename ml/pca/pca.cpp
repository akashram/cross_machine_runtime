#include "pca.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>

namespace {

using Matrix = std::vector<std::vector<float>>;

Matrix transpose(const Matrix& M) {
    std::size_t rows = M.size(), cols = rows ? M[0].size() : 0;
    Matrix out(cols, std::vector<float>(rows));
    for (std::size_t i = 0; i < rows; ++i)
        for (std::size_t j = 0; j < cols; ++j) out[j][i] = M[i][j];
    return out;
}

Matrix matmul(const Matrix& A, const Matrix& B) {
    std::size_t n = A.size(), m = n ? A[0].size() : 0, p = B.size() ? B[0].size() : 0;
    Matrix out(n, std::vector<float>(p, 0.0f));
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t k = 0; k < m; ++k) {
            float a = A[i][k];
            if (a == 0.0f) continue;
            for (std::size_t j = 0; j < p; ++j) out[i][j] += a * B[k][j];
        }
    return out;
}

// Modified Gram-Schmidt QR: orthonormalizes Y's columns in place. Only Q
// is needed by randomized SVD (R is discarded), so this returns Q
// directly. Honest caveat, same convention as decision_tree/svm/knn: MGS
// rather than Householder QR -- adequate at the well-conditioned,
// modest-dimension scale this gets exercised on, not the most
// numerically robust QR available; a Householder version is a real,
// scoped-out follow-up.
Matrix orthonormalize_columns(Matrix Y) {
    std::size_t n = Y.size(), l = n ? Y[0].size() : 0;
    for (std::size_t j = 0; j < l; ++j) {
        for (std::size_t i = 0; i < j; ++i) {
            float dot = 0.0f;
            for (std::size_t r = 0; r < n; ++r) dot += Y[r][i] * Y[r][j];
            for (std::size_t r = 0; r < n; ++r) Y[r][j] -= dot * Y[r][i];
        }
        float norm = 0.0f;
        for (std::size_t r = 0; r < n; ++r) norm += Y[r][j] * Y[r][j];
        norm = std::sqrt(norm);
        if (norm < 1e-8f) continue;  // degenerate (near-zero) column: leave as-is rather than divide by ~0
        for (std::size_t r = 0; r < n; ++r) Y[r][j] /= norm;
    }
    return Y;
}

// Classic cyclic Jacobi eigenvalue algorithm (Golub & Van Loan) for a
// small symmetric matrix -- appropriate here since it operates on the
// (n_components + n_oversamples)-sized matrix B*B^T, not the full data
// matrix. Returns eigenvalues and eigenvectors (as columns of V), not
// sorted.
void jacobi_eigen(Matrix A, std::vector<float>& eigenvalues, Matrix& eigenvectors) {
    std::size_t n = A.size();
    eigenvectors.assign(n, std::vector<float>(n, 0.0f));
    for (std::size_t i = 0; i < n; ++i) eigenvectors[i][i] = 1.0f;

    const int max_sweeps = 100;
    for (int sweep = 0; sweep < max_sweeps; ++sweep) {
        float off = 0.0f;
        for (std::size_t p = 0; p < n; ++p)
            for (std::size_t q = p + 1; q < n; ++q) off += A[p][q] * A[p][q];
        if (off < 1e-12f) break;

        for (std::size_t p = 0; p < n; ++p) {
            for (std::size_t q = p + 1; q < n; ++q) {
                if (std::fabs(A[p][q]) < 1e-12f) continue;
                float theta = (A[q][q] - A[p][p]) / (2.0f * A[p][q]);
                float t = (theta >= 0 ? 1.0f : -1.0f) / (std::fabs(theta) + std::sqrt(theta * theta + 1.0f));
                float c = 1.0f / std::sqrt(t * t + 1.0f);
                float s = t * c;
                float tau = s / (1.0f + c);

                float apq = A[p][q];
                float app2 = A[p][p], aqq2 = A[q][q];
                A[p][p] = app2 - t * apq;
                A[q][q] = aqq2 + t * apq;
                A[p][q] = 0.0f;
                A[q][p] = 0.0f;

                for (std::size_t i = 0; i < n; ++i) {
                    if (i == p || i == q) continue;
                    float aip = A[i][p], aiq = A[i][q];
                    A[i][p] = A[p][i] = aip - s * (aiq + tau * aip);
                    A[i][q] = A[q][i] = aiq + s * (aip - tau * aiq);
                }
                for (std::size_t i = 0; i < n; ++i) {
                    float vip = eigenvectors[i][p], viq = eigenvectors[i][q];
                    eigenvectors[i][p] = vip - s * (viq + tau * vip);
                    eigenvectors[i][q] = viq + s * (vip - tau * viq);
                }
            }
        }
    }

    eigenvalues.resize(n);
    for (std::size_t i = 0; i < n; ++i) eigenvalues[i] = A[i][i];
}

}  // namespace

PCA::PCA(PCAParams params) : params_(params) {}

void PCA::fit(const Features& X) {
    std::size_t n = X.size();
    std::size_t d = n ? X[0].size() : 0;

    mean_.assign(d, 0.0f);
    for (const auto& row : X)
        for (std::size_t f = 0; f < d; ++f) mean_[f] += row[f];
    for (std::size_t f = 0; f < d; ++f) mean_[f] /= static_cast<float>(n);

    Matrix A(n, std::vector<float>(d));
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t f = 0; f < d; ++f) A[i][f] = X[i][f] - mean_[f];

    std::size_t max_rank = std::min(n, d);
    std::size_t n_components = static_cast<std::size_t>(std::max(1, params_.n_components));
    n_components = std::min(n_components, max_rank);
    std::size_t l = std::min(n_components + static_cast<std::size_t>(std::max(0, params_.n_oversamples)), max_rank);

    // Halko et al. 2011 randomized range finder: Y = A * Omega for a
    // random Gaussian Omega, re-orthonormalized between power
    // iterations (Y = A(A^T Y)) so the basis doesn't lose precision as
    // it sharpens onto the dominant singular subspace.
    std::mt19937 rng(params_.random_state);
    std::normal_distribution<float> gauss(0.0f, 1.0f);
    Matrix Omega(d, std::vector<float>(l));
    for (auto& row : Omega)
        for (auto& v : row) v = gauss(rng);

    Matrix Y = matmul(A, Omega);
    Matrix At = transpose(A);
    for (int q = 0; q < params_.n_power_iterations; ++q) {
        Y = orthonormalize_columns(Y);
        Matrix Z = matmul(At, Y);
        Y = matmul(A, Z);
    }
    Matrix Q = orthonormalize_columns(Y);  // n x l orthonormal basis for A's dominant column space

    Matrix B = matmul(transpose(Q), A);   // l x d, A projected into the small basis
    Matrix Bt = transpose(B);
    Matrix BBt = matmul(B, Bt);           // l x l symmetric -- BB^T = U_hat S^2 U_hat^T

    std::vector<float> eigenvalues;
    Matrix eigenvectors;
    jacobi_eigen(BBt, eigenvalues, eigenvectors);

    std::vector<std::size_t> order(l);
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) { return eigenvalues[a] > eigenvalues[b]; });

    singular_values_.assign(n_components, 0.0f);
    components_.assign(n_components, std::vector<float>(d, 0.0f));
    for (std::size_t c = 0; c < n_components; ++c) {
        std::size_t idx = order[c];
        float eigenvalue = std::max(eigenvalues[idx], 0.0f);
        float s = std::sqrt(eigenvalue);
        singular_values_[c] = s;
        if (s < 1e-8f) continue;  // negligible component (rank-deficient data): leave as zero vector

        // V[:,c] = (1/s) * B^T * U_hat[:,c] -- see README's Design note
        // for the derivation from B = U_hat S V^T.
        std::vector<float> v(d, 0.0f);
        for (std::size_t r = 0; r < l; ++r) {
            float u = eigenvectors[r][idx];
            if (u == 0.0f) continue;
            for (std::size_t f = 0; f < d; ++f) v[f] += u * B[r][f];
        }
        float norm = 0.0f;
        for (float x : v) norm += x * x;
        norm = std::sqrt(norm);
        if (norm > 1e-8f)
            for (float& x : v) x /= norm;
        components_[c] = v;
    }

    // Total variance uses ALL features (the standard sklearn-style
    // explained_variance_ratio_ definition), not just the retained
    // components -- otherwise the ratios would trivially sum to 1
    // regardless of how much of the data's actual variance they cover.
    float total_variance = 0.0f;
    for (std::size_t f = 0; f < d; ++f) {
        float var = 0.0f;
        for (std::size_t i = 0; i < n; ++i) var += A[i][f] * A[i][f];
        total_variance += var / static_cast<float>(n > 1 ? n - 1 : 1);
    }

    explained_variance_.assign(n_components, 0.0f);
    explained_variance_ratio_.assign(n_components, 0.0f);
    for (std::size_t c = 0; c < n_components; ++c) {
        explained_variance_[c] = singular_values_[c] * singular_values_[c] / static_cast<float>(n > 1 ? n - 1 : 1);
        explained_variance_ratio_[c] = total_variance > 1e-12f ? explained_variance_[c] / total_variance : 0.0f;
    }
}

Features PCA::transform(const Features& X) const {
    std::size_t d = mean_.size();
    Features out(X.size(), std::vector<float>(components_.size()));
    for (std::size_t i = 0; i < X.size(); ++i) {
        std::vector<float> centered(d);
        for (std::size_t f = 0; f < d; ++f) centered[f] = X[i][f] - mean_[f];
        for (std::size_t c = 0; c < components_.size(); ++c) {
            float proj = 0.0f;
            for (std::size_t f = 0; f < d; ++f) proj += centered[f] * components_[c][f];
            if (params_.whiten) {
                float denom = std::sqrt(std::max(explained_variance_[c], 1e-12f));
                proj /= denom;
            }
            out[i][c] = proj;
        }
    }
    return out;
}

Features PCA::fit_transform(const Features& X) {
    fit(X);
    return transform(X);
}
