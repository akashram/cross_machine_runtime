#include "roofline/roofline.h"
#include "avx512/kernels.h"
#include "tiling/matmul.h"
#include "inference/mlp.h"
#include "affinity/affinity.h"
#include "foundation/bench/bench.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

#if defined(__AVX2__) || defined(__AVX512F__)
#  include <immintrin.h>
#endif

using namespace cpu_engine;
using namespace cpu_engine::roofline;
using bench::tsc_now;
using bench::tsc_to_ns;

// ---------------------------------------------------------------------------
// Roofline benchmark — Phase 2 step 10
//
// Two hardware ceilings are measured directly on this machine:
//   1. Peak FLOPs: AVX2/AVX-512 FMA micro-benchmark (8 independent accumulators)
//   2. Peak bandwidth: STREAM Triad on a 192 MB working set (> LLC)
//
// Then every cpu_engine kernel is sampled with a DRAM-resident working set
// (arrays larger than LLC) so the bandwidth ceiling is actually visible.
// Each kernel is also sampled with an in-cache working set (L1/L2 resident)
// to show the same kernel can sit at a different roofline point depending on
// data locality.
//
// Arithmetic intensity (AI = FLOPs / bytes) is computed from the theoretical
// minimum bytes (one load/store per element).  Actual bytes may be higher for
// kernels with poor cache reuse (e.g. naive matmul) — step 11 will measure
// actual L3 miss counts to reveal this directly.
// ---------------------------------------------------------------------------

// ============================================================================
// 1.  Peak FLOPS measurement
// ============================================================================

static double measure_peak_flops_gflops() {
#if defined(__AVX512F__)
    // 8 independent __m512 accumulators × 16 floats × 2 ops/FMA
    __m512 a0=_mm512_set1_ps(1.0f), a1=_mm512_set1_ps(1.1f),
           a2=_mm512_set1_ps(0.9f), a3=_mm512_set1_ps(1.2f),
           a4=_mm512_set1_ps(0.8f), a5=_mm512_set1_ps(1.3f),
           a6=_mm512_set1_ps(0.7f), a7=_mm512_set1_ps(1.4f);
    __m512 m = _mm512_set1_ps(1.0001f);
    __m512 d = _mm512_set1_ps(0.5f);
    constexpr int kIter = 1 << 22;
    // Warmup
    for (int w = 0; w < 5000; ++w) {
        a0=_mm512_fmadd_ps(a0,m,d); a1=_mm512_fmadd_ps(a1,m,d);
        a2=_mm512_fmadd_ps(a2,m,d); a3=_mm512_fmadd_ps(a3,m,d);
    }
    uint64_t t0 = tsc_now();
    for (int i = 0; i < kIter; ++i) {
        a0=_mm512_fmadd_ps(a0,m,d); a1=_mm512_fmadd_ps(a1,m,d);
        a2=_mm512_fmadd_ps(a2,m,d); a3=_mm512_fmadd_ps(a3,m,d);
        a4=_mm512_fmadd_ps(a4,m,d); a5=_mm512_fmadd_ps(a5,m,d);
        a6=_mm512_fmadd_ps(a6,m,d); a7=_mm512_fmadd_ps(a7,m,d);
    }
    uint64_t t1 = tsc_now();
    volatile float sink = _mm512_cvtss_f32(
        _mm512_add_ps(_mm512_add_ps(_mm512_add_ps(a0,a1), _mm512_add_ps(a2,a3)),
                      _mm512_add_ps(_mm512_add_ps(a4,a5), _mm512_add_ps(a6,a7))));
    (void)sink;
    // 8 accum × 16 floats/vec × 2 ops/FMA × kIter
    return 8.0 * 16.0 * 2.0 * kIter / (tsc_to_ns(t1 - t0) * 1e-9) / 1e9;

#elif defined(__AVX2__)
    // 8 independent __m256 accumulators × 8 floats × 2 ops/FMA
    __m256 a0=_mm256_set1_ps(1.0f), a1=_mm256_set1_ps(1.1f),
           a2=_mm256_set1_ps(0.9f), a3=_mm256_set1_ps(1.2f),
           a4=_mm256_set1_ps(0.8f), a5=_mm256_set1_ps(1.3f),
           a6=_mm256_set1_ps(0.7f), a7=_mm256_set1_ps(1.4f);
    __m256 m = _mm256_set1_ps(1.0001f);
    __m256 d = _mm256_set1_ps(0.5f);
    constexpr int kIter = 1 << 22;
    for (int w = 0; w < 5000; ++w) {
        a0=_mm256_fmadd_ps(a0,m,d); a1=_mm256_fmadd_ps(a1,m,d);
        a2=_mm256_fmadd_ps(a2,m,d); a3=_mm256_fmadd_ps(a3,m,d);
    }
    uint64_t t0 = tsc_now();
    for (int i = 0; i < kIter; ++i) {
        a0=_mm256_fmadd_ps(a0,m,d); a1=_mm256_fmadd_ps(a1,m,d);
        a2=_mm256_fmadd_ps(a2,m,d); a3=_mm256_fmadd_ps(a3,m,d);
        a4=_mm256_fmadd_ps(a4,m,d); a5=_mm256_fmadd_ps(a5,m,d);
        a6=_mm256_fmadd_ps(a6,m,d); a7=_mm256_fmadd_ps(a7,m,d);
    }
    uint64_t t1 = tsc_now();
    volatile float sink = _mm256_cvtss_f32(
        _mm256_add_ps(_mm256_add_ps(_mm256_add_ps(a0,a1), _mm256_add_ps(a2,a3)),
                      _mm256_add_ps(_mm256_add_ps(a4,a5), _mm256_add_ps(a6,a7))));
    (void)sink;
    return 8.0 * 8.0 * 2.0 * kIter / (tsc_to_ns(t1 - t0) * 1e-9) / 1e9;

#else
    // Scalar fallback: 4 independent accumulators
    float a0=1,a1=1,a2=1,a3=1, m=1.0001f, d=0.5f;
    constexpr int kIter = 1 << 24;
    uint64_t t0 = tsc_now();
    for (int i = 0; i < kIter; ++i) {
        a0=a0*m+d; a1=a1*m+d; a2=a2*m+d; a3=a3*m+d;
    }
    uint64_t t1 = tsc_now();
    volatile float sink = a0+a1+a2+a3; (void)sink;
    return 4.0 * 2.0 * kIter / (tsc_to_ns(t1 - t0) * 1e-9) / 1e9;
#endif
}

// ============================================================================
// 2.  Peak memory bandwidth  (STREAM Triad: a[i] = b[i] + s*c[i])
// ============================================================================

static double measure_peak_bw_gbps() {
    // 16 M floats × 3 arrays = 192 MB — far exceeds any single-core LLC slice
    constexpr std::size_t N = 1u << 24;
    std::vector<float> a(N, 0.0f), b(N, 1.0f), c(N, 2.0f);
    const float scalar = 3.0f;

    double best = 0.0;
    for (int run = 0; run < 10; ++run) {
        uint64_t t0 = tsc_now();
#pragma clang loop vectorize(enable)
        for (std::size_t i = 0; i < N; ++i)
            a[i] = b[i] + scalar * c[i];
        uint64_t t1 = tsc_now();
        volatile float sink = a[N / 2]; (void)sink;
        double bytes = 3.0 * static_cast<double>(N) * sizeof(float); // 2r+1w
        best = std::max(best, bytes / (tsc_to_ns(t1 - t0) * 1e-9) / 1e9);
    }
    return best;
}

// ============================================================================
// 3.  Kernel sample helpers
// ============================================================================

template<typename Fn>
static double sample_gflops(Fn fn, double flops, int warmup=5, int passes=20) {
    for (int i = 0; i < warmup; ++i) fn();
    uint64_t t0 = tsc_now();
    for (int i = 0; i < passes; ++i) fn();
    uint64_t t1 = tsc_now();
    double ns = tsc_to_ns(t1 - t0) / passes;
    return flops / (ns * 1e-9) / 1e9;
}

// ---------------------------------------------------------------------------
// dot_f32  — AI = 0.25 FLOP/byte (2N FLOPs / 8N bytes)
// ---------------------------------------------------------------------------
static KernelPoint sample_dot(const char* tag, int N) {
    std::vector<float> a(static_cast<std::size_t>(N), 1.0f);
    std::vector<float> b(static_cast<std::size_t>(N), 2.0f);
    volatile float sink = 0;
    double gflops = sample_gflops([&]{
        sink = avx512::dot_f32(a.data(), b.data(), N);
    }, 2.0 * N);
    (void)sink;
    char name[64];
    snprintf(name, sizeof(name), "dot_f32 (%s)", tag);
    double bytes = 2.0 * N * sizeof(float);
    return {name, 2.0 * N, bytes, gflops};
}

// ---------------------------------------------------------------------------
// matvec_f32  — AI ≈ 0.5 FLOP/byte (2MN FLOPs / ~4MN bytes)
// ---------------------------------------------------------------------------
static KernelPoint sample_matvec(const char* tag, int M, int N) {
    std::vector<float> A(static_cast<std::size_t>(M * N), 0.5f);
    std::vector<float> x(static_cast<std::size_t>(N),     1.0f);
    std::vector<float> y(static_cast<std::size_t>(M),     0.0f);
    volatile float sink = 0;
    double gflops = sample_gflops([&]{
        avx512::matvec_f32(A.data(), x.data(), y.data(), M, N);
        sink = y[0];
    }, 2.0 * M * N);
    (void)sink;
    char name[64];
    snprintf(name, sizeof(name), "matvec_f32 (%s)", tag);
    // Minimum bytes: A (MN) + x (N, assumed cached for large M) + y (M)
    double bytes = static_cast<double>(M * N + N + M) * sizeof(float);
    return {name, 2.0 * M * N, bytes, gflops};
}

// ---------------------------------------------------------------------------
// eltwise_relu  — AI = 0.125 FLOP/byte (N FLOPs / 8N bytes)
// ---------------------------------------------------------------------------
static KernelPoint sample_relu(const char* tag, int N) {
    std::vector<float> in (static_cast<std::size_t>(N), -0.5f);
    std::vector<float> out(static_cast<std::size_t>(N),  0.0f);
    volatile float sink = 0;
    double gflops = sample_gflops([&]{
        avx512::eltwise_relu_f32(in.data(), out.data(), N);
        sink = out[0];
    }, 1.0 * N);
    (void)sink;
    char name[64];
    snprintf(name, sizeof(name), "eltwise_relu (%s)", tag);
    double bytes = 2.0 * N * sizeof(float);
    return {name, 1.0 * N, bytes, gflops};
}

// ---------------------------------------------------------------------------
// eltwise_add  — AI = 0.083 FLOP/byte (N FLOPs / 12N bytes)
// ---------------------------------------------------------------------------
static KernelPoint sample_add(const char* tag, int N) {
    std::vector<float> a  (static_cast<std::size_t>(N), 1.0f);
    std::vector<float> b  (static_cast<std::size_t>(N), 2.0f);
    std::vector<float> out(static_cast<std::size_t>(N), 0.0f);
    volatile float sink = 0;
    double gflops = sample_gflops([&]{
        avx512::eltwise_add_f32_autovec(a.data(), b.data(), out.data(), N);
        sink = out[0];
    }, 1.0 * N);
    (void)sink;
    char name[64];
    snprintf(name, sizeof(name), "eltwise_add (%s)", tag);
    double bytes = 3.0 * N * sizeof(float);
    return {name, 1.0 * N, bytes, gflops};
}

// ---------------------------------------------------------------------------
// matmul_naive  — AI = M/6 FLOP/byte (2M³ / 12M²) for square M×M×M
// ---------------------------------------------------------------------------
static KernelPoint sample_matmul_naive(const char* tag, int M) {
    std::vector<float> A(static_cast<std::size_t>(M * M), 0.5f);
    std::vector<float> B(static_cast<std::size_t>(M * M), 0.3f);
    std::vector<float> C(static_cast<std::size_t>(M * M), 0.0f);
    volatile float sink = 0;
    double gflops = sample_gflops([&]{
        std::memset(C.data(), 0, static_cast<std::size_t>(M * M) * sizeof(float));
        tiling::matmul_naive_f32(A.data(), B.data(), C.data(), M, M, M);
        sink = C[0];
    }, 2.0 * M * M * M, 3, 10);
    (void)sink;
    char name[64];
    snprintf(name, sizeof(name), "matmul_naive %dx%d (%s)", M, M, tag);
    // Minimum bytes = 3 matrices each read/written once
    double bytes = 3.0 * M * M * sizeof(float);
    return {name, 2.0 * M * M * M, bytes, gflops};
}

// ---------------------------------------------------------------------------
// matmul_tiled  — same FLOPs/bytes as naive; higher achieved GFLOPS (better reuse)
// ---------------------------------------------------------------------------
static KernelPoint sample_matmul_tiled(const char* tag, int M, int tile) {
    std::vector<float> A(static_cast<std::size_t>(M * M), 0.5f);
    std::vector<float> B(static_cast<std::size_t>(M * M), 0.3f);
    std::vector<float> C(static_cast<std::size_t>(M * M), 0.0f);
    volatile float sink = 0;
    double gflops = sample_gflops([&]{
        std::memset(C.data(), 0, static_cast<std::size_t>(M * M) * sizeof(float));
        tiling::matmul_tiled_f32(A.data(), B.data(), C.data(), M, M, M, tile);
        sink = C[0];
    }, 2.0 * M * M * M, 3, 10);
    (void)sink;
    char name[64];
    snprintf(name, sizeof(name), "matmul_tiled T=%d %dx%d (%s)", tile, M, M, tag);
    double bytes = 3.0 * M * M * sizeof(float);
    return {name, 2.0 * M * M * M, bytes, gflops};
}

// ---------------------------------------------------------------------------
// MLP forward pass  — AI depends on layer dims
// ---------------------------------------------------------------------------
static KernelPoint sample_mlp(const char* tag,
                               const inference::MlpConfig& cfg) {
    const int nl = cfg.n_layers();
    std::vector<std::vector<float>> weights(static_cast<std::size_t>(nl));
    std::vector<std::vector<float>> biases (static_cast<std::size_t>(nl));
    for (std::size_t l = 0; l < static_cast<std::size_t>(nl); ++l) {
        weights[l].assign(static_cast<std::size_t>(cfg.dims[l] * cfg.dims[l+1]), 0.1f);
        biases [l].assign(static_cast<std::size_t>(cfg.dims[l+1]), 0.0f);
    }
    inference::MlpInferenceEngine engine{cfg, std::move(weights), std::move(biases)};

    std::vector<float> input (static_cast<std::size_t>(engine.input_dim()),  0.5f);
    std::vector<float> output(static_cast<std::size_t>(engine.output_dim()), 0.0f);
    volatile float sink = 0;

    // Total FLOPs = 2 × Σ_l(in_dim × out_dim)  (matvec per layer)
    double flops = 0.0;
    double bytes = 0.0;
    for (std::size_t l = 0; l < static_cast<std::size_t>(nl); ++l) {
        flops += 2.0 * cfg.dims[l] * cfg.dims[l + 1];
        // Minimum bytes: weight matrix + bias (read once), output (written once)
        bytes += static_cast<double>(cfg.dims[l] * cfg.dims[l+1] + cfg.dims[l+1] + cfg.dims[l+1])
                 * sizeof(float);
    }

    double gflops = sample_gflops([&]{
        engine.forward(input.data(), output.data());
        sink = output[0];
    }, flops);
    (void)sink;

    char name[64];
    snprintf(name, sizeof(name), "mlp_fwd (%s)", tag);
    return {name, flops, bytes, gflops};
}

// ============================================================================
// main
// ============================================================================

int main() {
    ThreadPinner::pin(0);
    (void)bench::tsc_ticks_per_ns();

    printf("Roofline Model — CPU Backend\n");
    printf("============================\n");
#if defined(__AVX512F__)
    printf("ISA: AVX-512F\n");
#elif defined(__AVX2__)
    printf("ISA: AVX2\n");
#else
    printf("ISA: scalar\n");
#endif

    // --- Ceiling measurements ---
    printf("\nMeasuring hardware ceilings...\n");
    const double peak_gflops = measure_peak_flops_gflops();
    printf("  Peak FLOPS (FMA micro-bench): %.1f GFLOPS\n", peak_gflops);
    const double peak_bw = measure_peak_bw_gbps();
    printf("  Peak bandwidth (STREAM Triad, 192 MB): %.1f GB/s\n", peak_bw);
    printf("  Ridge point: %.2f FLOP/byte\n", peak_gflops / peak_bw);

    RooflineModel model{peak_gflops, peak_bw};

    // --- Kernel samples ---
    // "hot" = L1/L2 resident  (small N, fits in 256 KB L2)
    // "cold" = DRAM resident   (large N, >> LLC)
    printf("\nSampling kernels...\n");

    // dot: hot = 8K floats (32 KB/array, L1); cold = 8M floats (32 MB/array, DRAM)
    auto dot_hot  = sample_dot("L1",   8192);
    auto dot_cold = sample_dot("DRAM", 1 << 23);

    // matvec: hot = 128×128 (64 KB mat, L2); cold = 1024×1024 (4 MB mat, DRAM)
    auto mv_hot  = sample_matvec("L2",   128, 128);
    auto mv_cold = sample_matvec("DRAM", 1024, 1024);

    // relu/add: hot = 16K elems (64 KB, L2); cold = 8M elems (32 MB, DRAM)
    auto relu_hot  = sample_relu("L1",   16384);
    auto relu_cold = sample_relu("DRAM", 1 << 23);
    auto add_cold  = sample_add ("DRAM", 1 << 23);

    // matmul: 256×256 (768 KB, L3 borderline); best tile vs naive
    auto mm_naive  = sample_matmul_naive("L3",        256);
    auto mm_tiled  = sample_matmul_tiled("L3, T=64",  256, 64);

    // MLP: tiny (L1-resident weights) and small (L2-resident)
    inference::MlpConfig tiny_cfg{
        .dims={64,128,64,32},
        .acts={inference::Activation::kRelu,
               inference::Activation::kRelu,
               inference::Activation::kNone}};
    inference::MlpConfig small_cfg{
        .dims={256,512,256,128},
        .acts={inference::Activation::kRelu,
               inference::Activation::kRelu,
               inference::Activation::kNone}};
    auto mlp_tiny  = sample_mlp("tiny  64→128→64→32",   tiny_cfg);
    auto mlp_small = sample_mlp("small 256→512→256→128", small_cfg);

    std::vector<KernelPoint> kernels = {
        dot_hot, dot_cold,
        mv_hot,  mv_cold,
        relu_hot, relu_cold, add_cold,
        mm_naive, mm_tiled,
        mlp_tiny, mlp_small,
    };

    model.print_table(kernels);
    model.print_chart(kernels);

    ThreadPinner::unpin();
}
