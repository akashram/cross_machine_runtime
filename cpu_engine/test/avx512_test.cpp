#include "avx512/kernels.h"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

using namespace cpu_engine::avx512;

static constexpr float kF32Tol = 1e-4f;   // tolerance for FP32 comparison
static constexpr float kSigTol = 1e-2f;   // tolerance for fast sigmoid approx

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
static bool near_f32(float a, float b, float tol = kF32Tol) {
    return std::fabs(a - b) <= tol * (1.0f + std::fabs(b));
}

// Generate reproducible random data
static std::vector<float> rand_f32(int n, float lo = -1.0f, float hi = 1.0f,
                                   uint32_t seed = 42) {
    std::mt19937 rng{seed};
    std::uniform_real_distribution<float> d{lo, hi};
    std::vector<float> v(static_cast<std::size_t>(n));
    for (auto& x : v) x = d(rng);
    return v;
}

static std::vector<int8_t> rand_i8(int n, uint32_t seed = 99) {
    std::mt19937 rng{seed};
    std::uniform_int_distribution<int> d{-127, 127};
    std::vector<int8_t> v(static_cast<std::size_t>(n));
    for (auto& x : v) x = static_cast<int8_t>(d(rng));
    return v;
}

// ---------------------------------------------------------------------------
// Test 1: dot_f32 — all tiers agree
// ---------------------------------------------------------------------------
static void test_dot_f32() {
    for (int n : {1, 7, 16, 17, 64, 65, 256, 1024, 4096}) {
        auto a = rand_f32(n, -1, 1, 1);
        auto b = rand_f32(n, -1, 1, 2);

        float s  = dot_f32_scalar (a.data(), b.data(), n);
        float av = dot_f32_autovec(a.data(), b.data(), n);

        assert(near_f32(s, av));

#ifdef __AVX512F__
        float a5 = dot_f32_avx512(a.data(), b.data(), n);
        assert(near_f32(s, a5));
#endif
    }
    printf("PASS  test_dot_f32  (sizes: 1..4096, all tiers agree)\n");
}

// ---------------------------------------------------------------------------
// Test 2: matvec_f32 — all tiers produce identical output vectors
// ---------------------------------------------------------------------------
static void test_matvec_f32() {
    for (auto [M, N] : std::initializer_list<std::pair<int,int>>{
            {1,1}, {4,4}, {16,16}, {7,13}, {64,128}, {128,64}}) {
        auto A  = rand_f32(M * N, -1, 1, 10);
        auto x  = rand_f32(N,     -1, 1, 11);
        std::vector<float> ys(static_cast<std::size_t>(M));
        std::vector<float> ya(static_cast<std::size_t>(M));

        matvec_f32_scalar (A.data(), x.data(), ys.data(), M, N);
        matvec_f32_autovec(A.data(), x.data(), ya.data(), M, N);

        for (int m = 0; m < M; ++m)
            assert(near_f32(ys[static_cast<std::size_t>(m)],
                            ya[static_cast<std::size_t>(m)]));

#ifdef __AVX512F__
        std::vector<float> y5(static_cast<std::size_t>(M));
        matvec_f32_avx512(A.data(), x.data(), y5.data(), M, N);
        for (int m = 0; m < M; ++m)
            assert(near_f32(ys[static_cast<std::size_t>(m)],
                            y5[static_cast<std::size_t>(m)]));
#endif
    }
    printf("PASS  test_matvec_f32\n");
}

// ---------------------------------------------------------------------------
// Test 3: element-wise add
// ---------------------------------------------------------------------------
static void test_eltwise_add() {
    const int n = 1025;  // non-multiple of 16 to exercise tail
    auto a  = rand_f32(n, -5, 5, 20);
    auto b  = rand_f32(n, -5, 5, 21);
    std::vector<float> os(static_cast<std::size_t>(n));
    std::vector<float> oa(static_cast<std::size_t>(n));

    eltwise_add_f32_scalar (a.data(), b.data(), os.data(), n);
    eltwise_add_f32_autovec(a.data(), b.data(), oa.data(), n);
    for (int i = 0; i < n; ++i)
        assert(near_f32(os[static_cast<std::size_t>(i)],
                        oa[static_cast<std::size_t>(i)]));

#ifdef __AVX512F__
    std::vector<float> o5(static_cast<std::size_t>(n));
    eltwise_add_f32_avx512(a.data(), b.data(), o5.data(), n);
    for (int i = 0; i < n; ++i)
        assert(near_f32(os[static_cast<std::size_t>(i)],
                        o5[static_cast<std::size_t>(i)]));
#endif
    printf("PASS  test_eltwise_add  (n=%d)\n", n);
}

// ---------------------------------------------------------------------------
// Test 4: element-wise relu
// ---------------------------------------------------------------------------
static void test_eltwise_relu() {
    const int n = 257;
    auto in = rand_f32(n, -3, 3, 30);
    std::vector<float> os(static_cast<std::size_t>(n));
    std::vector<float> oa(static_cast<std::size_t>(n));

    eltwise_relu_f32_scalar (in.data(), os.data(), n);
    eltwise_relu_f32_autovec(in.data(), oa.data(), n);

    for (int i = 0; i < n; ++i) {
        auto ii = static_cast<std::size_t>(i);
        assert(os[ii] >= 0.0f);
        assert(near_f32(os[ii], oa[ii]));
    }

#ifdef __AVX512F__
    std::vector<float> o5(static_cast<std::size_t>(n));
    eltwise_relu_f32_avx512(in.data(), o5.data(), n);
    for (int i = 0; i < n; ++i)
        assert(near_f32(os[static_cast<std::size_t>(i)],
                        o5[static_cast<std::size_t>(i)]));
#endif
    printf("PASS  test_eltwise_relu  (n=%d)\n", n);
}

// ---------------------------------------------------------------------------
// Test 5: fast sigmoid approximation is in (0,1) and monotone
// ---------------------------------------------------------------------------
static void test_eltwise_sigmoid() {
    const int n = 513;
    auto in = rand_f32(n, -5, 5, 40);
    std::vector<float> os(static_cast<std::size_t>(n));
    std::vector<float> oa(static_cast<std::size_t>(n));

    eltwise_sigmoid_f32_scalar (in.data(), os.data(), n);
    eltwise_sigmoid_f32_autovec(in.data(), oa.data(), n);

    for (int i = 0; i < n; ++i) {
        auto ii = static_cast<std::size_t>(i);
        assert(os[ii] > 0.0f && os[ii] < 1.0f);  // output in (0,1)
        assert(near_f32(os[ii], oa[ii], kSigTol));
    }

    // Monotonicity: sigmoid is strictly increasing
    auto in_sorted = in;
    std::sort(in_sorted.begin(), in_sorted.end());
    std::vector<float> sorted_out(static_cast<std::size_t>(n));
    eltwise_sigmoid_f32_scalar(in_sorted.data(), sorted_out.data(), n);
    for (int i = 1; i < n; ++i)
        assert(sorted_out[static_cast<std::size_t>(i)] >=
               sorted_out[static_cast<std::size_t>(i-1)]);

#ifdef __AVX512F__
    std::vector<float> o5(static_cast<std::size_t>(n));
    eltwise_sigmoid_f32_avx512(in.data(), o5.data(), n);
    for (int i = 0; i < n; ++i)
        assert(near_f32(os[static_cast<std::size_t>(i)],
                        o5[static_cast<std::size_t>(i)], kSigTol));
#endif
    printf("PASS  test_eltwise_sigmoid  (n=%d, monotone, in (0,1))\n", n);
}

// ---------------------------------------------------------------------------
// Test 6: dot_i8_i32 — all tiers agree (exact integer arithmetic)
// ---------------------------------------------------------------------------
static void test_dot_i8() {
    for (int n : {1, 7, 32, 64, 65, 128, 256, 1024}) {
        auto a = rand_i8(n, 1);
        auto b = rand_i8(n, 2);

        int32_t s  = dot_i8_i32_scalar (a.data(), b.data(), n);
        int32_t av = dot_i8_i32_autovec(a.data(), b.data(), n);
        assert(s == av);

#ifdef __AVX512F__
        int32_t a5 = dot_i8_i32_avx512(a.data(), b.data(), n);
        assert(s == a5);
#endif
    }
    printf("PASS  test_dot_i8_i32  (sizes 1..1024, all tiers exact)\n");
}

// ---------------------------------------------------------------------------
// Test 7: tier-selector wrappers forward to the best available tier
// ---------------------------------------------------------------------------
static void test_tier_selectors() {
    const int n = 64;
    auto a = rand_f32(n, -1, 1, 77);
    auto b = rand_f32(n, -1, 1, 78);

    float ref = dot_f32_scalar(a.data(), b.data(), n);
    float got = dot_f32(a.data(), b.data(), n);
    assert(near_f32(ref, got));

    std::vector<float> out_ref(static_cast<std::size_t>(n));
    std::vector<float> out(static_cast<std::size_t>(n));
    eltwise_relu_f32_scalar(a.data(), out_ref.data(), n);
    eltwise_relu_f32(a.data(), out.data(), n);
    for (int i = 0; i < n; ++i)
        assert(near_f32(out_ref[static_cast<std::size_t>(i)],
                        out[static_cast<std::size_t>(i)]));

    printf("PASS  test_tier_selectors  (ISA: %s)\n",
#ifdef __AVX512F__
           "AVX-512"
#elif defined(__AVX2__)
           "AVX2 (auto-vectorized)"
#else
           "scalar"
#endif
    );
}

int main() {
    printf("AVX-512 kernel tests\n");
    printf("ISA: ");
#ifdef __AVX512F__
    printf("AVX-512F available — testing all three tiers\n\n");
#elif defined(__AVX2__)
    printf("AVX2 (no AVX-512 on this Mac) — testing scalar + auto-vec tiers\n\n");
#else
    printf("scalar only\n\n");
#endif

    test_dot_f32();
    test_matvec_f32();
    test_eltwise_add();
    test_eltwise_relu();
    test_eltwise_sigmoid();
    test_dot_i8();
    test_tier_selectors();

    printf("\nAll avx512 kernel tests passed.\n");
}
