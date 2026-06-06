#include "tiling/matmul.h"
#include "affinity/affinity.h"
#include "foundation/bench/bench.h"
#include "foundation/perf/perf.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <vector>

using namespace cpu_engine::tiling;
using cpu_engine::ThreadPinner;
using foundation::PerfCounters;

// ---------------------------------------------------------------------------
// Cache-aware tiling benchmark
//
// Goal: show that choosing a tile size that fits in L1/L2 cache dramatically
// reduces memory traffic and improves GFLOPS vs. the untiled baseline.
//
// Methodology:
//   - Pin to CPU 0 (stable clocks, no migration)
//   - Square matrices M=N=K so working set = 3 * M^2 * 4 bytes
//   - Sweep tile size T ∈ {16, 32, 48, 64, 96, 128, 256, 512}
//   - For each tile: measure GFLOPS and IPC via PerfCounters
//   - Also show the "naive" (untiled) baseline
//   - Print the working-set size alongside so cache tier is obvious
//
// GFLOPS formula: 2*M*K*N FLOPs per call (M*K*N multiply-adds).
//
// Hardware note:
//   macOS: PerfCounters.available() == false; only GFLOPS and TSC are shown.
//   Linux: IPC, LLC miss rate also printed.
// ---------------------------------------------------------------------------

static constexpr int kWarmup = 5;
static constexpr int kPasses = 20;

struct Result {
    double ns;
    double gflops;
    double ipc;
    double llc_miss_rate;
};

template <typename Fn>
static Result measure(Fn fn, double flops_per_call) {
    PerfCounters ctr;
    for (int i = 0; i < kWarmup; ++i) fn();
    ctr.start();
    uint64_t t0 = bench::tsc_now();
    for (int i = 0; i < kPasses; ++i) fn();
    uint64_t t1 = bench::tsc_now();
    auto snap = ctr.stop();
    double ns     = bench::tsc_to_ns(t1 - t0) / kPasses;
    double gflops = flops_per_call / (ns * 1e-9) / 1e9;
    return {ns, gflops, snap.ipc(), snap.cache_miss_rate()};
}

static void print_header() {
    printf("  %-12s  %-12s  %8s  %8s  %5s  %8s\n",
           "variant", "working-set", "ns/call", "GFLOPS", "IPC", "LLCmiss%");
    printf("  %s\n", std::string(72, '-').c_str());
}

static void print_row(const char* label, const char* ws_str,
                      Result r, double base_gflops = 0.0) {
    printf("  %-12s  %-12s  %8.0f  %8.2f", label, ws_str, r.ns, r.gflops);
    if (r.ipc > 0.0)
        printf("  %5.2f  %7.1f%%", r.ipc, r.llc_miss_rate * 100.0);
    else
        printf("  %5s  %8s", "n/a", "n/a");
    if (base_gflops > 0.0)
        printf("  [%.1fx]", r.gflops / base_gflops);
    printf("\n");
}

static void run_sweep(int M, float* A, float* B, float* C) {
    const double flops = 2.0 * M * M * M;

    printf("\n=== %dx%d matmul  (total data = %.1f MB) ===\n",
           M, M, 3.0 * M * M * sizeof(float) / (1024.0 * 1024.0));
    printf("  FLOPs per call: %.3g\n", flops);
    printf("  L1=32KB, L2=256KB, L3=varies  —  tile fits L1 when T≤52, L2 when T≤146\n");
    print_header();

    // Baseline: naive (tile = full matrix)
    auto run_naive = [&] {
        std::memset(C, 0, static_cast<std::size_t>(M * M) * sizeof(float));
        matmul_naive_f32(A, B, C, M, M, M);
    };
    Result r_naive = measure(run_naive, flops);
    // working-set string for naive: the full 3 matrices
    char ws_naive[32];
    snprintf(ws_naive, sizeof(ws_naive), "%.0f KB",
             3.0 * M * M * sizeof(float) / 1024.0);
    print_row("naive", ws_naive, r_naive);

    // Tile sweep
    static constexpr int kTiles[] = {16, 32, 48, 64, 96, 128, 256, 512};
    for (int T : kTiles) {
        if (T > M) continue; // no point if tile >= matrix

        auto run_tiled = [&] {
            std::memset(C, 0, static_cast<std::size_t>(M * M) * sizeof(float));
            matmul_tiled_f32(A, B, C, M, M, M, T);
        };
        Result r = measure(run_tiled, flops);

        char label[16];
        snprintf(label, sizeof(label), "tile=%d", T);
        char ws_str[32];
        std::size_t ws = tile_working_set_bytes(T);
        if (ws < 1024)
            snprintf(ws_str, sizeof(ws_str), "%zu B", ws);
        else if (ws < 1024 * 1024)
            snprintf(ws_str, sizeof(ws_str), "%.0f KB", ws / 1024.0);
        else
            snprintf(ws_str, sizeof(ws_str), "%.1f MB", ws / (1024.0 * 1024.0));

        print_row(label, ws_str, r, r_naive.gflops);
    }
}

int main() {
    ThreadPinner::pin(0);
    (void)bench::tsc_ticks_per_ns(); // calibrate TSC once

    printf("Cache-aware Tiling Benchmark\n");
    printf("ISA: ");
#if defined(__AVX2__)
    printf("AVX2");
#else
    printf("SSE2");
#endif
#if defined(__AVX512F__)
    printf("+AVX512");
#endif
    printf("\n");

    if (PerfCounters{}.available())
        printf("PerfCounters: available — IPC + LLC miss rate shown\n");
    else
        printf("PerfCounters: not available on macOS — timing only\n");

    // Allocate max size (1024×1024) once; reuse subarrays for smaller sizes.
    static constexpr int kMaxN = 1024;
    std::vector<float> A(kMaxN * kMaxN, 0.5f);
    std::vector<float> B(kMaxN * kMaxN, 0.3f);
    std::vector<float> C(kMaxN * kMaxN, 0.0f);

    // Fill with non-trivial values so the compiler can't optimize them away.
    for (std::size_t i = 0; i < static_cast<std::size_t>(kMaxN * kMaxN); ++i) {
        A[i] = static_cast<float>((static_cast<int>(i) % 17) - 8) * 0.1f;
        B[i] = static_cast<float>((static_cast<int>(i) % 13) - 6) * 0.1f;
    }

    // 256×256: entire working set (768 KB) barely fits in a shared L3 slice.
    // Tiling into T=32 (12 KB) should land squarely in L1.
    run_sweep(256, A.data(), B.data(), C.data());

    // 512×512: 3 MB — spills L3 on many CPUs.  Tiling wins clearly here.
    run_sweep(512, A.data(), B.data(), C.data());

    // 1024×1024: 12 MB — definitely beyond L3 on most mobile/server chips.
    // Only run smaller tile sizes to keep wall time reasonable.
    run_sweep(1024, A.data(), B.data(), C.data());

    ThreadPinner::unpin();
}
