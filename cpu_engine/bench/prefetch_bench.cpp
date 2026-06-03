#include "prefetch/prefetch.h"
#include "affinity/affinity.h"
#include "foundation/bench/bench.h"
#include "foundation/perf/perf.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <numeric>
#include <vector>

using cpu_engine::PrefetchHint;
using cpu_engine::prefetch_r;
using cpu_engine::prefetch_ahead;
using cpu_engine::make_pointer_chase_list;
using cpu_engine::ThreadPinner;
using foundation::PerfCounters;

// ---------------------------------------------------------------------------
// Prefetch distance benchmark
//
// Two experiments that show when software prefetch matters:
//
// 1. Pointer-chase scan (hardware prefetcher completely blind):
//    Walk a 32 MB random linked list. Each access depends on the previous
//    result, so the hardware prefetcher cannot detect the pattern.
//    We sweep software prefetch distances 0..128 and measure ns/access.
//    Optimal distance ≈ DRAM_latency_ns / ns_per_chain_step_without_miss.
//
// 2. Large-stride sequential scan (hardware prefetcher degrades):
//    Access every N-th element of a 32 MB array with stride > 2 KB.
//    Below ~2 KB stride the hardware prefetcher can keep up; above that it
//    falls off. We compare no-prefetch vs T0/T1/T2/NTA at the optimal
//    distance found in experiment 1.
//
// Machine-specific context (L3=4 MB, DRAM latency ≈ 80 ns):
//   Each pointer-chase step without prefetch: ~80-100 ns (DRAM miss).
//   With perfect prefetch at distance D: DRAM latency is hidden behind
//   D * (compute_ns) of computation. Goal: D ≈ 80 / compute_ns.
// ---------------------------------------------------------------------------

static constexpr std::size_t kNodes     = (32u << 20) / sizeof(std::size_t); // 32 MiB
static constexpr int         kChaseIter = 1 << 22;   // chain walk iterations per trial
static constexpr int         kWarmup    = 1 << 19;

// ---------------------------------------------------------------------------
// Experiment 1: pointer-chase distance sweep
// ---------------------------------------------------------------------------

// Walk the chain without software prefetch (baseline)
static double chain_no_prefetch(const std::size_t* chain) {
    std::size_t idx = 0;
    for (int i = 0; i < kWarmup; ++i) idx = chain[idx];

    uint64_t t0 = bench::tsc_now();
    for (int i = 0; i < kChaseIter; ++i) idx = chain[idx];
    uint64_t t1 = bench::tsc_now();

    if (idx == 42) printf(" ");  // prevent dead-code elimination
    return bench::tsc_to_ns(t1 - t0) / kChaseIter;
}

// Walk the chain with T0 prefetch at given distance
static double chain_with_prefetch(const std::size_t* chain, int dist,
                                  PrefetchHint hint) {
    std::size_t idx = 0;
    // Warmup — prime the prefetch pipeline
    for (int i = 0; i < kWarmup; ++i) {
        prefetch_r<PrefetchHint::T0>(&chain[chain[idx]]);
        idx = chain[idx];
    }

    // Unrolled prefetch-ahead loop: maintain a lookahead pointer at `dist` steps
    // We walk two pointers: `cur` (current) and `pre` (dist steps ahead for prefetch)
    std::size_t cur = 0;
    std::size_t pre = 0;
    for (int i = 0; i < dist; ++i) pre = chain[pre];  // advance `pre` by dist

    uint64_t t0 = bench::tsc_now();
    for (int i = 0; i < kChaseIter; ++i) {
        // Prefetch the node that is `dist` steps ahead
        switch (hint) {
            case PrefetchHint::T0:  prefetch_r<PrefetchHint::T0> (&chain[pre]); break;
            case PrefetchHint::T1:  prefetch_r<PrefetchHint::T1> (&chain[pre]); break;
            case PrefetchHint::T2:  prefetch_r<PrefetchHint::T2> (&chain[pre]); break;
            case PrefetchHint::NTA: prefetch_r<PrefetchHint::NTA>(&chain[pre]); break;
        }
        cur = chain[cur];
        pre = chain[pre];
    }
    uint64_t t1 = bench::tsc_now();

    if (cur == 42) printf(" ");
    return bench::tsc_to_ns(t1 - t0) / kChaseIter;
}

// ---------------------------------------------------------------------------
// Experiment 2: strided access distance sweep at the optimal distance found
// ---------------------------------------------------------------------------

static constexpr std::size_t kArrElems = (32u << 20) / sizeof(uint64_t); // 32 MiB

// Stride in elements that defeats the hardware prefetcher (> 2 KB = 256 uint64)
static constexpr std::size_t kStride = 256;  // 2 KB stride

static double strided_no_prefetch(const uint64_t* arr) {
    uint64_t sink = 0;
    for (std::size_t i = 0; i < kWarmup / kStride; ++i) sink += arr[(i * kStride) % kArrElems];

    std::size_t accesses = kArrElems / kStride;
    uint64_t t0 = bench::tsc_now();
    for (std::size_t i = 0; i < accesses; ++i)
        sink += arr[i * kStride];
    uint64_t t1 = bench::tsc_now();

    if (sink == 0xDEAD) printf(" ");  // prevent dead-code elimination
    return bench::tsc_to_ns(t1 - t0) / static_cast<double>(accesses);
}

static double strided_with_prefetch(const uint64_t* arr, std::size_t dist,
                                    PrefetchHint hint) {
    uint64_t sink = 0;
    std::size_t accesses = kArrElems / kStride;

    // Warmup
    for (std::size_t i = 0; i < kWarmup / kStride; ++i) {
        if (i + dist < kArrElems / kStride)
            prefetch_ahead(arr, i * kStride, dist * kStride);
        sink += arr[(i * kStride) % kArrElems];
    }

    uint64_t t0 = bench::tsc_now();
    for (std::size_t i = 0; i < accesses; ++i) {
        if (i + dist < accesses) {
            switch (hint) {
                case PrefetchHint::T0:  prefetch_ahead<uint64_t, PrefetchHint::T0> (arr, i * kStride, dist * kStride); break;
                case PrefetchHint::T1:  prefetch_ahead<uint64_t, PrefetchHint::T1> (arr, i * kStride, dist * kStride); break;
                case PrefetchHint::T2:  prefetch_ahead<uint64_t, PrefetchHint::T2> (arr, i * kStride, dist * kStride); break;
                case PrefetchHint::NTA: prefetch_ahead<uint64_t, PrefetchHint::NTA>(arr, i * kStride, dist * kStride); break;
            }
        }
        sink += arr[i * kStride];
    }
    uint64_t t1 = bench::tsc_now();

    if (sink == 0xDEAD) printf(" ");
    return bench::tsc_to_ns(t1 - t0) / static_cast<double>(accesses);
}

int main() {
    ThreadPinner::pin(0);
    (void)bench::tsc_ticks_per_ns();

    printf("Prefetch distance benchmark\n");
    printf("Cache: L1=%u KB  L2=%u KB  L3=%u MB\n", 32u, 256u, 4u);
    printf("Working set: %zu MiB (%zu MiB pointer array + %zu MiB uint64 array)\n",
           std::size_t{64},
           kNodes * sizeof(std::size_t) >> 20,
           kArrElems * sizeof(uint64_t) >> 20);
    printf("Both >> L3 (4 MB) — all accesses go to DRAM.\n\n");

    // Allocate and initialise the pointer-chase list
    std::vector<std::size_t> chain(kNodes);
    printf("Building random pointer-chase list (%zu nodes)... ", kNodes);
    fflush(stdout);
    make_pointer_chase_list(chain.data(), kNodes);
    printf("done.\n\n");

    // Allocate strided-access array
    std::vector<uint64_t> arr(kArrElems);
    std::iota(arr.begin(), arr.end(), uint64_t{1});

    // -----------------------------------------------------------------------
    // Experiment 1: pointer-chase distance sweep
    // -----------------------------------------------------------------------
    printf("=== Experiment 1: Pointer-chase scan (random access) ===\n");
    printf("Hardware prefetcher: BLIND (each address is data-dependent)\n");
    printf("Expected: large speedup from software prefetch at optimal distance\n\n");

    double baseline_ns = chain_no_prefetch(chain.data());
    printf("  dist=  0 (no prefetch):  %6.1f ns/access  (baseline)\n", baseline_ns);

    double best_ns   = baseline_ns;
    int    best_dist = 0;

    for (int dist : {1, 2, 4, 8, 16, 24, 32, 48, 64, 96, 128}) {
        double t0 = chain_with_prefetch(chain.data(), dist, PrefetchHint::T0);
        double t1 = chain_with_prefetch(chain.data(), dist, PrefetchHint::T1);
        double tnta = chain_with_prefetch(chain.data(), dist, PrefetchHint::NTA);
        printf("  dist=%3d  T0=%5.1f ns  T1=%5.1f ns  NTA=%5.1f ns",
               dist, t0, t1, tnta);
        if (t0 < best_ns) { best_ns = t0; best_dist = dist; }
        double speedup = baseline_ns / t0;
        if (speedup > 1.05) printf("  [%.1fx speedup]", speedup);
        printf("\n");
    }

    printf("\nBest: dist=%d  %.1f ns/access  (%.1fx vs no prefetch)\n",
           best_dist, best_ns, baseline_ns / best_ns);

    // -----------------------------------------------------------------------
    // Experiment 2: strided access
    // -----------------------------------------------------------------------
    printf("\n=== Experiment 2: Strided scan (stride=%zu elements = %zu KB) ===\n",
           kStride, kStride * sizeof(uint64_t) / 1024);
    printf("Hardware prefetcher: DEGRADES at this stride\n");
    printf("Optimal prefetch distance ≈ %d (from experiment 1)\n\n", best_dist);

    double base_str = strided_no_prefetch(arr.data());
    printf("  No prefetch:  %6.1f ns/access\n", base_str);

    for (int dist : {1, 4, 8, 16, 32}) {
        double t0  = strided_with_prefetch(arr.data(), static_cast<std::size_t>(dist), PrefetchHint::T0);
        double nta = strided_with_prefetch(arr.data(), static_cast<std::size_t>(dist), PrefetchHint::NTA);
        printf("  dist=%3d  T0=%5.1f ns  NTA=%5.1f ns", dist, t0, nta);
        double speedup = base_str / t0;
        if (speedup > 1.05) printf("  [%.1fx speedup]", speedup);
        printf("\n");
    }

    printf("\nNote: hardware prefetcher already handles strides < ~512 bytes.\n");
    printf("Software prefetch matters most for:\n");
    printf("  - Pointer chasing (shown above)\n");
    printf("  - Large irregular strides (shown above)\n");
    printf("  - Gather/scatter from sparse index arrays\n");
    printf("\nOn Linux, confirm cache miss rates with:\n");
    printf("  perf stat -e cache-misses,cache-references,LLC-load-misses ./prefetch_bench\n");

    ThreadPinner::unpin();
}
