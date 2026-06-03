#include "avx512/kernels.h"
#include "affinity/affinity.h"
#include "foundation/bench/bench.h"
#include "foundation/perf/perf.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

using namespace cpu_engine::avx512;
using cpu_engine::ThreadPinner;
using foundation::PerfCounters;

// ---------------------------------------------------------------------------
// AVX-512 kernel benchmark
//
// Methodology:
//   - Pin to CPU 0 for stable timing
//   - Allocate data in L2/L3 (compute-bound, not bandwidth-bound)
//   - Measure throughput as GFLOPS (or GOPS for INT8)
//   - For each kernel: scalar / auto-vec / AVX-512
//   - Report speedup over scalar baseline
//
// Sizes chosen to fit in L2 (256 KB) for compute-bound measurement:
//   dot_f32:    N = 32768  (128 KB per array at float32)
//   matvec_f32: M = 256, N = 256  (256 KB matrix)
//   eltwise:    N = 32768  (128 KB per array)
//   dot_i8:     N = 131072 (128 KB per array at int8)
//
// GFLOPS formula:
//   dot(N):     2N FLOPs (N FMA = 2 ops each)
//   matvec(MN): 2MN FLOPs
//   add/mul(N): N FLOPs
//   relu(N):    N FLOPs (comparison + conditional move)
// ---------------------------------------------------------------------------

static constexpr int kPasses   = 200;
static constexpr int kWarmup   = 20;
static constexpr int kN_dot    = 32768;
static constexpr int kM_matvec = 256;
static constexpr int kN_matvec = 256;
static constexpr int kN_elt    = 32768;
static constexpr int kN_i8     = 131072;

struct Result {
    double ns;
    double gflops;
    double ipc;
};

template<typename Fn>
static Result measure(Fn fn, double flops_per_call) {
    PerfCounters ctr;
    for (int i = 0; i < kWarmup; ++i) fn();
    ctr.start();
    uint64_t t0 = bench::tsc_now();
    for (int i = 0; i < kPasses; ++i) fn();
    uint64_t t1 = bench::tsc_now();
    auto snap = ctr.stop();
    double ns       = bench::tsc_to_ns(t1 - t0) / kPasses;
    double gflops   = flops_per_call / (ns * 1e-9) / 1e9;
    return { ns, gflops, snap.ipc() };
}

static void print_result(const char* label, Result r, double base_gflops = 0.0) {
    printf("  %-28s  %7.1f ns  %6.1f GFLOPS", label, r.ns, r.gflops);
    if (r.ipc > 0.0) printf("  IPC=%.2f", r.ipc);
    if (base_gflops > 0.0)
        printf("  [%.1fx speedup]", r.gflops / base_gflops);
    printf("\n");
}

static void bench_dot_f32(const float* a, const float* b) {
    printf("\n=== dot_f32 (N=%d, %.0f KB/array) ===\n",
           kN_dot, static_cast<double>(kN_dot) * 4 / 1024.0);
    printf("    FLOPs per call: %d (N FMAs = 2N ops)\n", 2 * kN_dot);

    volatile float sink = 0;
    auto rs = measure([&]{ sink = dot_f32_scalar (a, b, kN_dot); }, 2.0 * kN_dot);
    auto ra = measure([&]{ sink = dot_f32_autovec(a, b, kN_dot); }, 2.0 * kN_dot);
    (void)sink;

    print_result("scalar",    rs);
    print_result("auto-vec",  ra, rs.gflops);

#ifdef __AVX512F__
    auto r5 = measure([&]{ sink = dot_f32_avx512(a, b, kN_dot); }, 2.0 * kN_dot);
    print_result("AVX-512 FMA (4-way unroll)", r5, rs.gflops);
    printf("  Expected peak (1-core, 3 GHz): ~192 GFLOPS\n");
    printf("  Achieved: %.0f%%\n", r5.gflops / 192.0 * 100.0);
#else
    printf("  AVX-512: not available (macOS Intel has AVX2 only)\n");
    printf("  Run on Linux with --preset avx512 for AVX-512 numbers\n");
#endif
}

static void bench_matvec_f32(const float* A, const float* x, float* y) {
    printf("\n=== matvec_f32 (M=%d, N=%d, %.0f KB matrix) ===\n",
           kM_matvec, kN_matvec,
           static_cast<double>(kM_matvec) * kN_matvec * 4 / 1024.0);
    double flops = 2.0 * kM_matvec * kN_matvec;
    printf("    FLOPs per call: %.0f\n", flops);

    volatile float sink = 0;
    auto rs = measure([&]{ matvec_f32_scalar (A, x, y, kM_matvec, kN_matvec); sink=y[0]; }, flops);
    auto ra = measure([&]{ matvec_f32_autovec(A, x, y, kM_matvec, kN_matvec); sink=y[0]; }, flops);
    (void)sink;

    print_result("scalar",   rs);
    print_result("auto-vec", ra, rs.gflops);

#ifdef __AVX512F__
    auto r5 = measure([&]{ matvec_f32_avx512(A, x, y, kM_matvec, kN_matvec); sink=y[0]; }, flops);
    print_result("AVX-512 FMA", r5, rs.gflops);
#else
    printf("  AVX-512: not available on this platform\n");
#endif
}

static void bench_eltwise(const float* a, const float* /*b*/, float* out) {
    printf("\n=== element-wise FP32 (N=%d, %.0f KB/array) ===\n",
           kN_elt, static_cast<double>(kN_elt) * 4 / 1024.0);

    volatile float sink = 0;

    // relu
    auto rs_relu = measure([&]{ eltwise_relu_f32_scalar (a, out, kN_elt); sink=out[0]; }, kN_elt);
    auto ra_relu = measure([&]{ eltwise_relu_f32_autovec(a, out, kN_elt); sink=out[0]; }, kN_elt);
    printf("  relu:\n");
    print_result("  scalar",   rs_relu);
    print_result("  auto-vec", ra_relu, rs_relu.gflops);
#ifdef __AVX512F__
    auto r5_relu = measure([&]{ eltwise_relu_f32_avx512(a, out, kN_elt); sink=out[0]; }, kN_elt);
    print_result("  AVX-512 VMAXPS", r5_relu, rs_relu.gflops);
#endif

    // sigmoid
    auto rs_sig = measure([&]{ eltwise_sigmoid_f32_scalar (a, out, kN_elt); sink=out[0]; }, 3.0*kN_elt);
    auto ra_sig = measure([&]{ eltwise_sigmoid_f32_autovec(a, out, kN_elt); sink=out[0]; }, 3.0*kN_elt);
    printf("  sigmoid (fast approx: 0.5x/(1+|x|)+0.5):\n");
    print_result("  scalar",   rs_sig);
    print_result("  auto-vec", ra_sig, rs_sig.gflops);
#ifdef __AVX512F__
    auto r5_sig = measure([&]{ eltwise_sigmoid_f32_avx512(a, out, kN_elt); sink=out[0]; }, 3.0*kN_elt);
    print_result("  AVX-512 VDIVPS+FMA", r5_sig, rs_sig.gflops);
#endif

    (void)sink;
}

static void bench_dot_i8(const int8_t* a, const int8_t* b) {
    printf("\n=== dot_i8_i32 (N=%d, %.0f KB/array) ===\n",
           kN_i8, static_cast<double>(kN_i8) / 1024.0);
    printf("    Ops per call: %d (i8 multiply-accumulate to i32)\n", 2 * kN_i8);

    volatile int32_t sink = 0;
    double ops = 2.0 * kN_i8;
    auto rs = measure([&]{ sink = dot_i8_i32_scalar (a, b, kN_i8); }, ops);
    auto ra = measure([&]{ sink = dot_i8_i32_autovec(a, b, kN_i8); }, ops);
    (void)sink;

    print_result("scalar",   rs);
    print_result("auto-vec", ra, rs.gflops);

#ifdef __AVX512F__
    auto r5 = measure([&]{ sink = dot_i8_i32_avx512(a, b, kN_i8); }, ops);
    print_result("AVX-512 (i8→i16→i32, VPMADDWD)", r5, rs.gflops);
    printf("  Note: VNNI (VPDPBUSD) on Ice Lake+ gives 4x better INT8 throughput\n");
    printf("        Detect: grep avx512vnni /proc/cpuinfo\n");
#else
    printf("  AVX-512: not available on this platform\n");
#endif
}

int main() {
    ThreadPinner::pin(0);
    (void)bench::tsc_ticks_per_ns();

    printf("AVX-512 Kernel Benchmark\n");
    printf("ISA: ");
#ifdef __AVX512F__
    printf("AVX-512F + BW + VL — full intrinsic paths active\n");
#elif defined(__AVX2__)
    printf("AVX2 (no AVX-512 on macOS Intel) — scalar + auto-vec only\n");
    printf("Scalar and auto-vec paths always run; build with --preset avx512\n");
    printf("on a Linux cloud instance to get AVX-512 numbers.\n");
    printf("\nSetup for AWS c5.2xlarge (Xeon Platinum 8275CL):\n");
    printf("  cmake --preset avx512 && cmake --build --preset avx512 --target avx512_bench\n");
    printf("  # check AVX-512: grep -o 'avx512[a-z]*' /proc/cpuinfo | sort -u\n");
#endif
    printf("\n");

    if (PerfCounters{}.available())
        printf("PerfCounters: available — IPC shown\n");
    else
        printf("PerfCounters: not available on macOS\n");

    std::mt19937 rng{42};
    std::uniform_real_distribution<float> fd{-1.0f, 1.0f};
    std::uniform_int_distribution<int>    id{-127, 127};

    std::vector<float> fa(std::max(kN_dot, kM_matvec * kN_matvec));
    std::vector<float> fb(std::max(kN_dot, kN_matvec));
    std::vector<float> fy(std::max(kM_matvec, kN_elt));
    for (auto& v : fa) v = fd(rng);
    for (auto& v : fb) v = fd(rng);

    std::vector<int8_t> ia(static_cast<std::size_t>(kN_i8));
    std::vector<int8_t> ib(static_cast<std::size_t>(kN_i8));
    for (auto& v : ia) v = static_cast<int8_t>(id(rng));
    for (auto& v : ib) v = static_cast<int8_t>(id(rng));

    bench_dot_f32  (fa.data(), fb.data());
    bench_matvec_f32(fa.data(), fb.data(), fy.data());
    bench_eltwise  (fa.data(), fb.data(), fy.data());
    bench_dot_i8   (ia.data(), ib.data());

    ThreadPinner::unpin();
}
