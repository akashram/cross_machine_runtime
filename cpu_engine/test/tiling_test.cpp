#include "tiling/matmul.h"

#include <cmath>
#include <cstdio>
#include <cstddef>
#include <vector>

using namespace cpu_engine::tiling;

// Independent reference: pure scalar, no pragmas, easy to trust.
static void ref_matmul(const float* A, const float* B, float* C,
                       int M, int K, int N) {
    for (int i = 0; i < M; ++i)
        for (int k = 0; k < K; ++k)
            for (int j = 0; j < N; ++j)
                C[i * N + j] += A[i * K + k] * B[k * N + j];
}

static bool allclose(const float* ref, const float* got, int n,
                     float rtol = 1e-4f) {
    for (int i = 0; i < n; ++i) {
        float diff = std::fabs(ref[i] - got[i]);
        float scale = std::fabs(ref[i]) + 1.0f;
        if (diff > rtol * scale) {
            printf("  mismatch at [%d]: ref=%.6f got=%.6f\n",
                   i, static_cast<double>(ref[i]), static_cast<double>(got[i]));
            return false;
        }
    }
    return true;
}

static bool test_naive(int M, int K, int N) {
    auto sz_a = static_cast<std::size_t>(M * K);
    auto sz_b = static_cast<std::size_t>(K * N);
    auto sz_c = static_cast<std::size_t>(M * N);
    std::vector<float> A(sz_a), B(sz_b), C_ref(sz_c, 0), C_got(sz_c, 0);
    for (std::size_t i = 0; i < sz_a; ++i) A[i] = static_cast<float>((static_cast<int>(i) % 7) - 3) * 0.5f;
    for (std::size_t i = 0; i < sz_b; ++i) B[i] = static_cast<float>((static_cast<int>(i) % 5) - 2) * 0.3f;
    ref_matmul(A.data(), B.data(), C_ref.data(), M, K, N);
    matmul_naive_f32(A.data(), B.data(), C_got.data(), M, K, N);
    if (!allclose(C_ref.data(), C_got.data(), M * N)) {
        printf("FAIL: naive (%d,%d,%d)\n", M, K, N);
        return false;
    }
    return true;
}

static bool test_tiled(int M, int K, int N, int tile) {
    auto sz_a = static_cast<std::size_t>(M * K);
    auto sz_b = static_cast<std::size_t>(K * N);
    auto sz_c = static_cast<std::size_t>(M * N);
    std::vector<float> A(sz_a), B(sz_b), C_ref(sz_c, 0), C_got(sz_c, 0);
    for (std::size_t i = 0; i < sz_a; ++i) A[i] = static_cast<float>((static_cast<int>(i) % 7) - 3) * 0.5f;
    for (std::size_t i = 0; i < sz_b; ++i) B[i] = static_cast<float>((static_cast<int>(i) % 5) - 2) * 0.3f;
    ref_matmul(A.data(), B.data(), C_ref.data(), M, K, N);
    matmul_tiled_f32(A.data(), B.data(), C_got.data(), M, K, N, tile);
    if (!allclose(C_ref.data(), C_got.data(), M * N)) {
        printf("FAIL: tiled (%d,%d,%d) tile=%d\n", M, K, N, tile);
        return false;
    }
    return true;
}

int main() {
    bool ok = true;

    // Square power-of-two sizes
    for (int sz : {16, 32, 64, 128}) {
        ok &= test_naive(sz, sz, sz);
        for (int t : {8, 16, 32, 64})
            ok &= test_tiled(sz, sz, sz, t);
    }

    // Non-power-of-two to stress boundary handling
    ok &= test_naive(63, 47, 53);
    ok &= test_tiled(63, 47, 53, 16);
    ok &= test_tiled(63, 47, 53, 32);
    ok &= test_tiled(63, 47, 53, 64);

    // Tile larger than the matrix (degenerates to one tile)
    ok &= test_tiled(10, 10, 10, 64);
    ok &= test_tiled(17, 31, 13, 64);

    // Rectangular (M, K, N all different)
    ok &= test_naive(64, 128, 32);
    ok &= test_tiled(64, 128, 32, 16);
    ok &= test_tiled(64, 128, 32, 32);

    // Single row / column
    ok &= test_naive(1, 64, 64);
    ok &= test_tiled(1, 64, 64, 16);
    ok &= test_naive(64, 64, 1);
    ok &= test_tiled(64, 64, 1, 16);

    if (ok) {
        printf("PASS: all tiling correctness tests\n");
        return 0;
    }
    return 1;
}
