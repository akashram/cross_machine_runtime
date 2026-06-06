#include "perf_deep_dive/perf_deep_dive.h"
#include "avx512/kernels.h"
#include "tiling/matmul.h"
#include "inference/mlp.h"
#include "affinity/affinity.h"
#include "foundation/bench/bench.h"

#include <cstdio>
#include <cstring>
#include <vector>

using namespace cpu_engine;
using namespace cpu_engine::perf_deep_dive;
using bench::tsc_now;
using bench::tsc_to_ns;

// ---------------------------------------------------------------------------
// Hardware perf counter deep dive — Phase 2 step 11
//
// For every cpu_engine kernel:
//   IPC           — instructions per cycle (how well the pipeline is utilised)
//   L1 miss %     — fraction of L1D reads that miss (→ L2 lookup)
//   L2 miss %     — fraction of L1D misses that also miss L2 (→ L3 lookup)
//   L3 miss %     — fraction of L3 accesses that go to DRAM
//   branch miss % — fraction of branches mispredicted
//
// Each kernel is sampled at two data sizes:
//   "hot"  — working set fits in L1/L2; expect low miss rates, high IPC
//   "cold" — working set exceeds LLC;   expect high L3 miss %, low IPC
//
// macOS: perf counters not available. Columns show "n/a".
// Linux: run on any x86_64 instance with perf_event_paranoid ≤ 2.
//        AWS c5.2xlarge (Skylake-SP) or c6i.2xlarge (Ice Lake) recommended.
//
// Interpretation guide:
//   Bandwidth-bound kernel:  IPC < 1.5, high L3 miss %
//   Compute-bound kernel:    IPC > 3.0, near-zero L3 miss %
//   Branch-bound kernel:     high branch miss %, erratic IPC
//   Cache-thrashing kernel:  moderate L1/L2 miss %, low L3 miss %
//                            (working set fits in L3, but evicts L1/L2 often)
// ---------------------------------------------------------------------------

static constexpr int kWarmup = 5;
static constexpr int kPasses = 100;

// --- helpers ----------------------------------------------------------------

template<typename Fn>
static void bench_kernel(const char* label, Fn fn) {
    // Timing
    for (int i = 0; i < kWarmup; ++i) fn();
    uint64_t t0 = tsc_now();
    for (int i = 0; i < kPasses; ++i) fn();
    uint64_t t1 = tsc_now();
    double ns = tsc_to_ns(t1 - t0) / kPasses;

    // Hardware counters
    auto snap = measure_deep(kWarmup, kPasses, fn);

    printf("  %-36s  %7.0f ns  ", label, ns);
    if (snap.available) {
        printf("%5.2f  %5.1f%%  %5.1f%%  %5.1f%%  %5.1f%%\n",
               snap.ipc(),
               snap.l1_miss_pct(), snap.l2_miss_pct(),
               snap.l3_miss_pct(), snap.branch_miss_pct());
    } else {
        printf("  n/a    n/a    n/a    n/a    n/a  (macOS)\n");
    }
}

static void print_header() {
    printf("  %-36s  %9s  %5s  %6s  %6s  %6s  %6s\n",
           "kernel", "ns/call", "IPC", "L1mis%", "L2mis%", "L3mis%", "Brm%");
    printf("  %s\n", std::string(90, '-').c_str());
}

// ============================================================================
// Kernel sections
// ============================================================================

static void section_dot() {
    printf("\n-- dot_f32 --\n");
    print_header();
    // hot: 8 KB / array → L1-resident
    {
        constexpr int N = 2048;
        std::vector<float> a(N, 1.0f), b(N, 2.0f);
        volatile float sink = 0;
        bench_kernel("dot_f32  8KB (L1)", [&]{
            sink = avx512::dot_f32(a.data(), b.data(), N); });
        (void)sink;
    }
    // cold: 32 MB / array → DRAM
    {
        constexpr int N = 1 << 23;
        std::vector<float> a(N, 1.0f), b(N, 2.0f);
        volatile float sink = 0;
        bench_kernel("dot_f32  64MB (DRAM)", [&]{
            sink = avx512::dot_f32(a.data(), b.data(), N); });
        (void)sink;
    }
}

static void section_matvec() {
    printf("\n-- matvec_f32 --\n");
    print_header();
    // hot: 128×128 matrix = 64 KB → L2-resident
    {
        constexpr int M = 128, N = 128;
        std::vector<float> A(M*N, 0.5f), x(N, 1.0f), y(M, 0.0f);
        volatile float sink = 0;
        bench_kernel("matvec 128×128 (L2)", [&]{
            avx512::matvec_f32(A.data(), x.data(), y.data(), M, N);
            sink = y[0]; });
        (void)sink;
    }
    // cold: 1024×1024 = 4 MB → L3 / DRAM
    {
        constexpr int M = 1024, N = 1024;
        std::vector<float> A(M*N, 0.5f), x(N, 1.0f), y(M, 0.0f);
        volatile float sink = 0;
        bench_kernel("matvec 1024×1024 (L3/DRAM)", [&]{
            avx512::matvec_f32(A.data(), x.data(), y.data(), M, N);
            sink = y[0]; });
        (void)sink;
    }
}

static void section_eltwise() {
    printf("\n-- elementwise --\n");
    print_header();
    // relu hot: 16 KB → L1
    {
        constexpr int N = 4096;
        std::vector<float> in(N, -0.5f), out(N, 0.0f);
        volatile float sink = 0;
        bench_kernel("relu  16KB (L1)", [&]{
            avx512::eltwise_relu_f32(in.data(), out.data(), N);
            sink = out[0]; });
        (void)sink;
    }
    // relu cold: 64 MB → DRAM
    {
        constexpr int N = 1 << 23;
        std::vector<float> in(N, -0.5f), out(N, 0.0f);
        volatile float sink = 0;
        bench_kernel("relu  64MB (DRAM)", [&]{
            avx512::eltwise_relu_f32(in.data(), out.data(), N);
            sink = out[0]; });
        (void)sink;
    }
    // add cold: 3 arrays × 64 MB = 192 MB → DRAM
    {
        constexpr int N = 1 << 23;
        std::vector<float> a(N, 1.0f), b(N, 2.0f), c(N, 0.0f);
        volatile float sink = 0;
        bench_kernel("add   192MB (DRAM)", [&]{
            avx512::eltwise_add_f32_autovec(a.data(), b.data(), c.data(), N);
            sink = c[0]; });
        (void)sink;
    }
}

static void section_matmul() {
    printf("\n-- matmul --\n");
    print_header();
    // 256×256: 768 KB total, L3-borderline
    {
        constexpr int M = 256;
        std::vector<float> A(M*M, 0.5f), B(M*M, 0.3f), C(M*M, 0.0f);
        volatile float sink = 0;
        bench_kernel("matmul_naive 256×256 (L3)", [&]{
            std::memset(C.data(), 0, static_cast<std::size_t>(M*M)*sizeof(float));
            tiling::matmul_naive_f32(A.data(), B.data(), C.data(), M, M, M);
            sink = C[0]; });
        bench_kernel("matmul_tiled T=64 256×256 (L3)", [&]{
            std::memset(C.data(), 0, static_cast<std::size_t>(M*M)*sizeof(float));
            tiling::matmul_tiled_f32(A.data(), B.data(), C.data(), M, M, M, 64);
            sink = C[0]; });
        bench_kernel("matmul_tiled T=32 256×256 (L3)", [&]{
            std::memset(C.data(), 0, static_cast<std::size_t>(M*M)*sizeof(float));
            tiling::matmul_tiled_f32(A.data(), B.data(), C.data(), M, M, M, 32);
            sink = C[0]; });
        (void)sink;
    }
}

static void section_mlp() {
    printf("\n-- MLP forward --\n");
    print_header();

    auto make_engine = [](const inference::MlpConfig& cfg) {
        const int nl = cfg.n_layers();
        std::vector<std::vector<float>> weights(static_cast<std::size_t>(nl));
        std::vector<std::vector<float>> biases (static_cast<std::size_t>(nl));
        for (std::size_t l = 0; l < static_cast<std::size_t>(nl); ++l) {
            weights[l].assign(static_cast<std::size_t>(cfg.dims[l]*cfg.dims[l+1]), 0.1f);
            biases [l].assign(static_cast<std::size_t>(cfg.dims[l+1]), 0.0f);
        }
        return inference::MlpInferenceEngine{cfg, std::move(weights), std::move(biases)};
    };

    // tiny: all weights fit in L1/L2 (~60 KB total)
    {
        inference::MlpConfig cfg{
            .dims={64,128,64,32},
            .acts={inference::Activation::kRelu,
                   inference::Activation::kRelu,
                   inference::Activation::kNone}};
        auto engine = make_engine(cfg);
        std::vector<float> in(64, 0.5f), out(32, 0.0f);
        volatile float sink = 0;
        bench_kernel("mlp_fwd 64→128→64→32 (L1/L2)", [&]{
            engine.forward(in.data(), out.data()); sink = out[0]; });
        (void)sink;
    }
    // small: weights ~1.25 MB, L3-resident
    {
        inference::MlpConfig cfg{
            .dims={256,512,256,128},
            .acts={inference::Activation::kRelu,
                   inference::Activation::kRelu,
                   inference::Activation::kNone}};
        auto engine = make_engine(cfg);
        std::vector<float> in(256, 0.5f), out(128, 0.0f);
        volatile float sink = 0;
        bench_kernel("mlp_fwd 256→512→256→128 (L3)", [&]{
            engine.forward(in.data(), out.data()); sink = out[0]; });
        (void)sink;
    }
    // large: weights ~18 MB, DRAM
    {
        inference::MlpConfig cfg{
            .dims={1024,2048,1024,512},
            .acts={inference::Activation::kRelu,
                   inference::Activation::kRelu,
                   inference::Activation::kNone}};
        auto engine = make_engine(cfg);
        std::vector<float> in(1024, 0.5f), out(512, 0.0f);
        volatile float sink = 0;
        bench_kernel("mlp_fwd 1024→2048→1024→512 (DRAM)", [&]{
            engine.forward(in.data(), out.data()); sink = out[0]; });
        (void)sink;
    }
}

// ============================================================================
// main
// ============================================================================

int main() {
    ThreadPinner::pin(0);
    (void)bench::tsc_ticks_per_ns();

    printf("Hardware Perf Counter Deep Dive\n");
    printf("================================\n");
    printf("Platform: ");
#if defined(__AVX512F__)
    printf("AVX-512");
#elif defined(__AVX2__)
    printf("AVX2");
#else
    printf("scalar");
#endif
    printf("\n");

    {
        perf_deep_dive::ExtendedPerfCounters probe;
        if (probe.available())
            printf("PerfCounters: available — full counter breakdown shown\n");
        else
            printf("PerfCounters: not available on macOS\n"
                   "Run on Linux (perf_event_paranoid <= 2) for real data.\n\n"
                   "Expected patterns (from roofline analysis, step 10):\n"
                   "  Bandwidth-bound (dot/matvec/relu DRAM): IPC < 1.5, L3miss > 50%%\n"
                   "  Compute-bound (matmul 256x256):          IPC > 3.0, L3miss < 5%%\n"
                   "  In-cache (hot kernels):                  IPC > 2.5, L1miss < 2%%\n");
    }

    section_dot();
    section_matvec();
    section_eltwise();
    section_matmul();
    section_mlp();

    ThreadPinner::unpin();
}
