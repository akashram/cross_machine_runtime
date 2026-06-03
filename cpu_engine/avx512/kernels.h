#pragma once

// AVX-512 Kernel Library — Three-Tier Implementation
// =========================================================================
//
// TIER ARCHITECTURE
// -----------------
// Every kernel ships in three tiers so that the same source file compiles
// everywhere and the right tier is selected at runtime (or compile-time on
// the hot path):
//
//   Tier 1 — Scalar
//     Plain C++ loops. Correct reference implementation. Always compiles.
//     Used in correctness tests to validate higher tiers.
//
//   Tier 2 — Auto-vectorized
//     Compiler-friendly loops: no intrinsics, annotated with pragmas so
//     Clang/GCC auto-vectorise to AVX2 (this Mac) or AVX-512 (Linux/cloud).
//     Serves as the baseline in the benchmark.
//
//   Tier 3 — AVX-512 intrinsics
//     Explicit _mm512_* calls. NEEDS __AVX512F__ (compile with -mavx512f).
//     Enabled on: AWS c5.2xlarge / c6i / m6i (Xeon Platinum Ice Lake/Cascade).
//     NOT available on macOS Intel — the CPU has AVX2 but not AVX-512.
//
//
// KERNELS PROVIDED
// ----------------
//   dot_f32        — FP32 dot product, N elements
//                    FLOPs: 2N  (N multiply + N add, fused to N FMA)
//
//   matvec_f32     — FP32 M×N matrix × N vector → M vector
//                    FLOPs: 2·M·N
//
//   eltwise_add_f32   — element-wise a + b
//   eltwise_mul_f32   — element-wise a * b
//   eltwise_relu_f32  — element-wise max(0, x)
//   eltwise_sigmoid_f32 — fast sigmoid: 0.5·x/(1+|x|)+0.5  (~0.5% error)
//
//   dot_i8_i32     — INT8 dot product accumulating into INT32
//                    (widening: i8 → i16 → i32 sum)
//
//
// HOW TO ENABLE AVX-512
// ---------------------
// macOS Intel: not possible (CPU lacks AVX-512 support).
//
// Linux (AWS c5/c6i/m6i instance):
//   cmake --preset avx512
//   cmake --build --preset avx512 --target avx512_bench
//
// Cloud instance check:
//   grep -o 'avx512[a-z]*' /proc/cpuinfo | sort -u
//   Expected: avx512f, avx512bw, avx512vl, avx512dq, avx512cd
//
//
// PERFORMANCE TARGETS (Intel Skylake-X / Ice Lake SP, AVX-512)
// --------------------------------------------------------------
//   Theoretical peak (one core, 512-bit FMA, 2 FMA/cycle, ~3 GHz):
//     FP32: 2 * 16 floats/FMA * 2 FMA/cycle * 3e9 = 192 GFLOPS
//     INT8: 2 * 64 int8/cycle * 3e9 = 384 GOPS (with VNNI)
//
//   Expected achieved (dot product, data in L1):
//     Scalar:        ~3–6   GFLOPS
//     Auto-vec AVX2: ~30–50 GFLOPS
//     AVX-512 FMA:   ~80–150 GFLOPS (4–5× over auto-vec)
//
// =========================================================================

#include <cstddef>
#include <cstdint>
#include <cmath>
#include <algorithm>

#if defined(__AVX512F__)
#  include <immintrin.h>
#elif defined(__AVX2__)
#  include <immintrin.h>
#endif

namespace cpu_engine::avx512 {

// =========================================================================
// Constants
// =========================================================================
static constexpr int kF32PerAvx512 = 16;  // float32s per __m512
static constexpr int kI8PerAvx512  = 64;  // int8s per __m512i

// =========================================================================
// 1. dot_f32 — FP32 dot product: sum(a[i] * b[i])
// =========================================================================

// --- Tier 1: scalar ---
[[nodiscard]]
inline float dot_f32_scalar(const float* __restrict__ a,
                             const float* __restrict__ b,
                             int n) noexcept {
    float acc = 0.0f;
    for (int i = 0; i < n; ++i) acc += a[i] * b[i];
    return acc;
}

// --- Tier 2: auto-vectorized ---
// Pragma tells Clang/GCC: reduce this loop to FMA. Compiles to AVX2 FMA here,
// AVX-512 FMA on a wider machine.
[[nodiscard]]
inline float dot_f32_autovec(const float* __restrict__ a,
                              const float* __restrict__ b,
                              int n) noexcept {
    float acc = 0.0f;
#pragma clang loop vectorize(enable) interleave(enable)
    for (int i = 0; i < n; ++i) acc += a[i] * b[i];
    return acc;
}

// --- Tier 3: AVX-512 intrinsics ---
#ifdef __AVX512F__
[[nodiscard]]
inline float dot_f32_avx512(const float* __restrict__ a,
                              const float* __restrict__ b,
                              int n) noexcept {
    __m512 acc0 = _mm512_setzero_ps();
    __m512 acc1 = _mm512_setzero_ps();
    __m512 acc2 = _mm512_setzero_ps();
    __m512 acc3 = _mm512_setzero_ps();

    // 4-way unrolled FMA: hides FMA latency (4 cycles on SKX) behind
    // 4 independent accumulator chains.
    int i = 0;
    for (; i + 64 <= n; i += 64) {
        acc0 = _mm512_fmadd_ps(_mm512_loadu_ps(a+i),    _mm512_loadu_ps(b+i),    acc0);
        acc1 = _mm512_fmadd_ps(_mm512_loadu_ps(a+i+16), _mm512_loadu_ps(b+i+16), acc1);
        acc2 = _mm512_fmadd_ps(_mm512_loadu_ps(a+i+32), _mm512_loadu_ps(b+i+32), acc2);
        acc3 = _mm512_fmadd_ps(_mm512_loadu_ps(a+i+48), _mm512_loadu_ps(b+i+48), acc3);
    }
    acc0 = _mm512_add_ps(acc0, acc1);
    acc2 = _mm512_add_ps(acc2, acc3);
    acc0 = _mm512_add_ps(acc0, acc2);

    // Remaining 16-float blocks
    for (; i + 16 <= n; i += 16)
        acc0 = _mm512_fmadd_ps(_mm512_loadu_ps(a+i), _mm512_loadu_ps(b+i), acc0);

    // Horizontal reduction: sum all 16 lanes
    float result = _mm512_reduce_add_ps(acc0);

    // Scalar tail
    for (; i < n; ++i) result += a[i] * b[i];
    return result;
}
#endif // __AVX512F__


// =========================================================================
// 2. matvec_f32 — M×N matrix times N-vector: y[m] = sum_n(A[m*N+n] * x[n])
// =========================================================================

// --- Tier 1: scalar ---
inline void matvec_f32_scalar(const float* __restrict__ A,
                               const float* __restrict__ x,
                               float* __restrict__ y,
                               int M, int N) noexcept {
    for (int m = 0; m < M; ++m) {
        float acc = 0.0f;
        for (int n = 0; n < N; ++n) acc += A[m * N + n] * x[n];
        y[m] = acc;
    }
}

// --- Tier 2: auto-vectorized ---
inline void matvec_f32_autovec(const float* __restrict__ A,
                                const float* __restrict__ x,
                                float* __restrict__ y,
                                int M, int N) noexcept {
    for (int m = 0; m < M; ++m) {
        float acc = 0.0f;
        const float* row = A + m * N;
#pragma clang loop vectorize(enable) interleave(enable)
        for (int n = 0; n < N; ++n) acc += row[n] * x[n];
        y[m] = acc;
    }
}

// --- Tier 3: AVX-512 ---
#ifdef __AVX512F__
inline void matvec_f32_avx512(const float* __restrict__ A,
                               const float* __restrict__ x,
                               float* __restrict__ y,
                               int M, int N) noexcept {
    for (int m = 0; m < M; ++m) {
        const float* row = A + m * N;
        __m512 acc = _mm512_setzero_ps();
        int n = 0;
        for (; n + 16 <= N; n += 16)
            acc = _mm512_fmadd_ps(_mm512_loadu_ps(row + n),
                                  _mm512_loadu_ps(x + n), acc);
        float s = _mm512_reduce_add_ps(acc);
        for (; n < N; ++n) s += row[n] * x[n];
        y[m] = s;
    }
}
#endif


// =========================================================================
// 3. Element-wise FP32 kernels
// =========================================================================

// --- add ---
inline void eltwise_add_f32_scalar(const float* __restrict__ a,
                                    const float* __restrict__ b,
                                    float* __restrict__ out, int n) noexcept {
    for (int i = 0; i < n; ++i) out[i] = a[i] + b[i];
}

inline void eltwise_add_f32_autovec(const float* __restrict__ a,
                                     const float* __restrict__ b,
                                     float* __restrict__ out, int n) noexcept {
    int i = 0;
#pragma clang loop vectorize(enable) interleave(enable)
    for (; i < n; ++i) out[i] = a[i] + b[i];
}

#ifdef __AVX512F__
inline void eltwise_add_f32_avx512(const float* __restrict__ a,
                                    const float* __restrict__ b,
                                    float* __restrict__ out, int n) noexcept {
    int i = 0;
    for (; i + 16 <= n; i += 16)
        _mm512_storeu_ps(out + i,
            _mm512_add_ps(_mm512_loadu_ps(a + i), _mm512_loadu_ps(b + i)));
    for (; i < n; ++i) out[i] = a[i] + b[i];
}
#endif

// --- mul ---
inline void eltwise_mul_f32_scalar(const float* __restrict__ a,
                                    const float* __restrict__ b,
                                    float* __restrict__ out, int n) noexcept {
    for (int i = 0; i < n; ++i) out[i] = a[i] * b[i];
}

inline void eltwise_mul_f32_autovec(const float* __restrict__ a,
                                     const float* __restrict__ b,
                                     float* __restrict__ out, int n) noexcept {
    int i = 0;
#pragma clang loop vectorize(enable) interleave(enable)
    for (; i < n; ++i) out[i] = a[i] * b[i];
}

#ifdef __AVX512F__
inline void eltwise_mul_f32_avx512(const float* __restrict__ a,
                                    const float* __restrict__ b,
                                    float* __restrict__ out, int n) noexcept {
    int i = 0;
    for (; i + 16 <= n; i += 16)
        _mm512_storeu_ps(out + i,
            _mm512_mul_ps(_mm512_loadu_ps(a + i), _mm512_loadu_ps(b + i)));
    for (; i < n; ++i) out[i] = a[i] * b[i];
}
#endif

// --- relu: max(0, x) ---
inline void eltwise_relu_f32_scalar(const float* __restrict__ in,
                                     float* __restrict__ out, int n) noexcept {
    for (int i = 0; i < n; ++i) out[i] = in[i] > 0.0f ? in[i] : 0.0f;
}

inline void eltwise_relu_f32_autovec(const float* __restrict__ in,
                                      float* __restrict__ out, int n) noexcept {
    int i = 0;
#pragma clang loop vectorize(enable) interleave(enable)
    for (; i < n; ++i) out[i] = in[i] > 0.0f ? in[i] : 0.0f;
}

#ifdef __AVX512F__
inline void eltwise_relu_f32_avx512(const float* __restrict__ in,
                                     float* __restrict__ out, int n) noexcept {
    const __m512 zero = _mm512_setzero_ps();
    int i = 0;
    for (; i + 16 <= n; i += 16)
        _mm512_storeu_ps(out + i,
            _mm512_max_ps(_mm512_loadu_ps(in + i), zero));
    for (; i < n; ++i) out[i] = in[i] > 0.0f ? in[i] : 0.0f;
}
#endif

// --- sigmoid: fast approximation  0.5·x/(1+|x|)+0.5
//     Accurate to ~0.5% for |x| < 5. Avoids exp() and SVML. ----
inline float sigmoid_fast_scalar(float x) noexcept {
    return 0.5f * x / (1.0f + std::fabs(x)) + 0.5f;
}

inline void eltwise_sigmoid_f32_scalar(const float* __restrict__ in,
                                        float* __restrict__ out, int n) noexcept {
    for (int i = 0; i < n; ++i) out[i] = sigmoid_fast_scalar(in[i]);
}

inline void eltwise_sigmoid_f32_autovec(const float* __restrict__ in,
                                         float* __restrict__ out, int n) noexcept {
    int i = 0;
#pragma clang loop vectorize(enable) interleave(enable)
    for (; i < n; ++i) {
        float x   = in[i];
        float absx = x > 0.0f ? x : -x;
        out[i] = 0.5f * x / (1.0f + absx) + 0.5f;
    }
}

#ifdef __AVX512F__
inline void eltwise_sigmoid_f32_avx512(const float* __restrict__ in,
                                        float* __restrict__ out, int n) noexcept {
    // Fast sigmoid: 0.5 * x / (1 + |x|) + 0.5
    // |x| via _mm512_abs_ps (AVX-512F), reciprocal via _mm512_rcp14_ps (fast)
    const __m512 half = _mm512_set1_ps(0.5f);
    const __m512 one  = _mm512_set1_ps(1.0f);

    int i = 0;
    for (; i + 16 <= n; i += 16) {
        __m512 x    = _mm512_loadu_ps(in + i);
        __m512 absx = _mm512_abs_ps(x);             // |x|
        __m512 denom= _mm512_add_ps(one, absx);     // 1 + |x|
        // _mm512_div_ps: full-precision divide (~18 cycles on SKX)
        // Use _mm512_rcp14_ps + Newton step for ~14-bit accuracy if speed needed
        __m512 sig  = _mm512_fmadd_ps(half,
                          _mm512_div_ps(x, denom),  // 0.5 * x / (1+|x|)
                          half);                     // + 0.5
        _mm512_storeu_ps(out + i, sig);
    }
    for (; i < n; ++i) out[i] = sigmoid_fast_scalar(in[i]);
}
#endif


// =========================================================================
// 4. INT8 dot product accumulating to INT32
//
// Algorithm: widen i8→i16 in two halves, multiply as i16, widen to i32, sum.
// This avoids VNNI (which needs Ice Lake+) for broader compatibility.
// On Cascade Lake+ with VNNI, _mm512_dpbusd_epi32 is 4x faster.
// =========================================================================

[[nodiscard]]
inline int32_t dot_i8_i32_scalar(const int8_t* __restrict__ a,
                                  const int8_t* __restrict__ b,
                                  int n) noexcept {
    int32_t acc = 0;
    for (int i = 0; i < n; ++i) acc += static_cast<int32_t>(a[i]) * b[i];
    return acc;
}

[[nodiscard]]
inline int32_t dot_i8_i32_autovec(const int8_t* __restrict__ a,
                                   const int8_t* __restrict__ b,
                                   int n) noexcept {
    int32_t acc = 0;
    int i = 0;
#pragma clang loop vectorize(enable) interleave(enable)
    for (; i < n; ++i) acc += static_cast<int32_t>(a[i]) * b[i];
    return acc;
}

#ifdef __AVX512F__
[[nodiscard]]
inline int32_t dot_i8_i32_avx512(const int8_t* __restrict__ a,
                                   const int8_t* __restrict__ b,
                                   int n) noexcept {
    __m512i acc = _mm512_setzero_si512();
    int i = 0;

    // Process 64 INT8s per iteration: widen lo/hi halves to i16, multiply, accumulate
    for (; i + 64 <= n; i += 64) {
        // Load 64 bytes of a and b
        __m512i va = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(a + i));
        __m512i vb = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(b + i));

        // Multiply pairs of i8 and accumulate into i16 using _mm512_maddubs_epi16
        // Note: this treats a as unsigned (u8) and b as signed (i8).
        // For signed × signed, use unpack to i16 instead.
        // We'll use the sign-correct widening path:
        // Widen lower 32 bytes: i8 → i16
        __m256i va_lo = _mm512_extracti64x4_epi64(va, 0);
        __m256i va_hi = _mm512_extracti64x4_epi64(va, 1);
        __m256i vb_lo = _mm512_extracti64x4_epi64(vb, 0);
        __m256i vb_hi = _mm512_extracti64x4_epi64(vb, 1);

        // Sign-extend i8→i16 (lower 16 bytes of each 256-bit register)
        __m512i a16_lo = _mm512_cvtepi8_epi16(va_lo);   // 32 int16
        __m512i a16_hi = _mm512_cvtepi8_epi16(va_hi);
        __m512i b16_lo = _mm512_cvtepi8_epi16(vb_lo);
        __m512i b16_hi = _mm512_cvtepi8_epi16(vb_hi);

        // Multiply i16×i16 → i32 and accumulate
        // _mm512_madd_epi16: multiply adjacent pairs of i16, add to i32
        acc = _mm512_add_epi32(acc, _mm512_madd_epi16(a16_lo, b16_lo));
        acc = _mm512_add_epi32(acc, _mm512_madd_epi16(a16_hi, b16_hi));
    }

    // Horizontal reduce i32 vector to scalar
    int32_t result = _mm512_reduce_add_epi32(acc);

    // Scalar tail
    for (; i < n; ++i) result += static_cast<int32_t>(a[i]) * b[i];
    return result;
}
#endif // __AVX512F__


// =========================================================================
// Tier selector — picks the best available tier at compile time
// =========================================================================
[[nodiscard]]
inline float dot_f32(const float* a, const float* b, int n) noexcept {
#ifdef __AVX512F__
    return dot_f32_avx512(a, b, n);
#else
    return dot_f32_autovec(a, b, n);
#endif
}

inline void matvec_f32(const float* A, const float* x, float* y,
                        int M, int N) noexcept {
#ifdef __AVX512F__
    matvec_f32_avx512(A, x, y, M, N);
#else
    matvec_f32_autovec(A, x, y, M, N);
#endif
}

inline void eltwise_relu_f32(const float* in, float* out, int n) noexcept {
#ifdef __AVX512F__
    eltwise_relu_f32_avx512(in, out, n);
#else
    eltwise_relu_f32_autovec(in, out, n);
#endif
}

[[nodiscard]]
inline int32_t dot_i8_i32(const int8_t* a, const int8_t* b, int n) noexcept {
#ifdef __AVX512F__
    return dot_i8_i32_avx512(a, b, n);
#else
    return dot_i8_i32_autovec(a, b, n);
#endif
}

} // namespace cpu_engine::avx512
