#include "branchless/branchless.h"
#include "affinity/affinity.h"
#include "foundation/bench/bench.h"
#include "foundation/perf/perf.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

using namespace cpu_engine;
using foundation::PerfCounters;
using foundation::PerfSnapshot;

// ---------------------------------------------------------------------------
// Branchless primitives benchmark
//
// We benchmark three operations on 4M random integers:
//
//  A. Count-if  (v > threshold):
//     Branchy:     if (v > T) count++;
//     Branchless:  count += (v > T);     → cmov-based
//
//  B. Clamp to [lo, hi]:
//     Branchy:     if (v < lo) v = lo; else if (v > hi) v = hi;
//     Branchless:  branchless_clamp(v, lo, hi)
//
//  C. ReLU: max(0, v):
//     Branchy:     if (v < 0) v = 0;
//     Branchless:  branchless_relu(v)
//
// For each operation, we report:
//   - ns/element
//   - Branch miss rate (from PerfCounters, Linux only)
//   - Speedup over branchy version
//
// With random uniformly-distributed data, expected branch miss rate:
//   Branchy count-if (threshold = median):  ~50%  (~20 cycles wasted/element)
//   Branchless count-if:                     ~0%  (no branch instruction)
//
// Expected timing relationship:
//   On random data: branchless wins (50% mispredictions are expensive)
//   On sorted data: branchy wins (predictor learns the pattern, cmov adds
//                   data-dependency latency the predictor avoids)
// ---------------------------------------------------------------------------

static constexpr int kN      = 4 << 20;   // 4M elements (~16 MB)
static constexpr int kPasses = 5;

struct Result {
    double ns_per_elem;
    double branch_miss_pct;
    double ipc;
};

static void print_result(const char* label, Result r, double baseline_ns = 0.0) {
    printf("  %-28s %5.2f ns/elem", label, r.ns_per_elem);
    if (r.ipc > 0.0)
        printf("  IPC=%.2f  Brmiss=%.1f%%", r.ipc, r.branch_miss_pct);
    if (baseline_ns > 0.0 && r.ns_per_elem < baseline_ns)
        printf("  [%.1fx faster]", baseline_ns / r.ns_per_elem);
    else if (baseline_ns > 0.0 && r.ns_per_elem > baseline_ns)
        printf("  [%.1fx slower]", r.ns_per_elem / baseline_ns);
    printf("\n");
}

// ---------------------------------------------------------------------------
// Benchmark harness: measure a loop over kN elements for kPasses passes
// ---------------------------------------------------------------------------
template<typename Fn>
static Result measure(Fn fn) {
    PerfCounters ctr;
    // Warmup
    fn();

    ctr.start();
    uint64_t t0 = bench::tsc_now();
    for (int p = 1; p < kPasses; ++p) fn();
    uint64_t t1 = bench::tsc_now();
    auto snap = ctr.stop();

    double ns      = bench::tsc_to_ns(t1 - t0);
    double per_elem= ns / (static_cast<double>(kN) * (kPasses - 1));

    return {
        per_elem,
        snap.branch_miss_rate() * 100.0,
        snap.ipc()
    };
}

// ---------------------------------------------------------------------------
// A. Count-if: v > threshold
// ---------------------------------------------------------------------------
static void bench_count_if(const std::vector<int32_t>& arr) {
    const int32_t threshold = 0;  // median of uniform distribution → 50% branch

    printf("\nA. count-if (v > %d) on %d M random int32  [threshold=median → 50%% branch]\n",
           threshold, kN >> 20);

    volatile int64_t sink = 0;

    // Branchy
    auto r_branch = measure([&] {
        int64_t count = 0;
        for (std::size_t i = 0; i < static_cast<std::size_t>(kN); ++i)
            if (arr[i] > threshold) ++count;
        sink = count;
    });
    print_result("branchy (if/else)", r_branch);

    // Branchless
    auto r_bl = measure([&] {
        int64_t count = 0;
        for (std::size_t i = 0; i < static_cast<std::size_t>(kN); ++i)
            count += static_cast<int64_t>(arr[i] > threshold);
        sink = count;
    });
    print_result("branchless (cmov)", r_bl, r_branch.ns_per_elem);

    (void)sink;
}

// ---------------------------------------------------------------------------
// B. Clamp to [-100, 100]
// ---------------------------------------------------------------------------
static void bench_clamp(const std::vector<int32_t>& arr) {
    const int32_t lo = -100, hi = 100;

    printf("\nB. clamp(v, %d, %d) on %d M random int32\n", lo, hi, kN >> 20);

    std::vector<int32_t> out(kN);
    volatile int32_t sink = 0;

    // Branchy
    auto r_branch = measure([&] {
        for (std::size_t i = 0; i < static_cast<std::size_t>(kN); ++i) {
            int32_t v = arr[i];
            if      (v < lo) v = lo;
            else if (v > hi) v = hi;
            out[i] = v;
        }
        sink = out[0];
    });
    print_result("branchy (if/else if)", r_branch);

    // Branchless
    auto r_bl = measure([&] {
        for (std::size_t i = 0; i < static_cast<std::size_t>(kN); ++i)
            out[i] = branchless_clamp(arr[i], lo, hi);
        sink = out[0];
    });
    print_result("branchless (2× cmov)", r_bl, r_branch.ns_per_elem);

    (void)sink;
}

// ---------------------------------------------------------------------------
// C. ReLU: max(0, v)  — neural network activation
// ---------------------------------------------------------------------------
static void bench_relu(const std::vector<int32_t>& arr) {
    printf("\nC. relu(v) = max(0, v)  on %d M random int32  [NN activation pattern]\n",
           kN >> 20);

    std::vector<int32_t> out(kN);
    volatile int32_t sink = 0;

    // Branchy
    auto r_branch = measure([&] {
        for (std::size_t i = 0; i < static_cast<std::size_t>(kN); ++i)
            out[i] = arr[i] < 0 ? 0 : arr[i];
        sink = out[0];
    });
    print_result("branchy (ternary)", r_branch);

    // Branchless
    auto r_bl = measure([&] {
        for (std::size_t i = 0; i < static_cast<std::size_t>(kN); ++i)
            out[i] = branchless_relu(arr[i]);
        sink = out[0];
    });
    print_result("branchless (cmov)", r_bl, r_branch.ns_per_elem);

    (void)sink;
}

// ---------------------------------------------------------------------------
// D. Sorted data — show that branchless loses when predictor wins
// ---------------------------------------------------------------------------
static void bench_sorted(std::vector<int32_t> sorted_arr) {
    const int32_t threshold = 0;

    printf("\nD. count-if on SORTED data  [predictor wins → branchless expected slower]\n");

    volatile int64_t sink = 0;

    auto r_branch = measure([&] {
        int64_t count = 0;
        for (std::size_t i = 0; i < static_cast<std::size_t>(kN); ++i)
            if (sorted_arr[i] > threshold) ++count;
        sink = count;
    });
    print_result("branchy (sorted)", r_branch);

    auto r_bl = measure([&] {
        int64_t count = 0;
        for (std::size_t i = 0; i < static_cast<std::size_t>(kN); ++i)
            count += static_cast<int64_t>(sorted_arr[i] > threshold);
        sink = count;
    });
    print_result("branchless (sorted)", r_bl, r_branch.ns_per_elem);

    (void)sink;
}

int main() {
    ThreadPinner::pin(0);
    (void)bench::tsc_ticks_per_ns();

    printf("Branchless primitives benchmark\n");
    printf("N=%d M elements  Passes=%d  Data: uniform random int32\n\n",
           kN >> 20, kPasses);

    if (PerfCounters{}.available()) {
        printf("PerfCounters: AVAILABLE — branch miss rate shown\n");
    } else {
        printf("PerfCounters: not available on macOS — timing only\n");
        printf("On Linux: perf stat -e branch-misses,branches ./branchless_bench\n");
    }

    // Generate random data once
    std::mt19937 rng{42};
    std::uniform_int_distribution<int32_t> dist(INT32_MIN, INT32_MAX);

    std::vector<int32_t> arr(kN);
    for (auto& v : arr) v = dist(rng);

    std::vector<int32_t> sorted_arr = arr;
    std::sort(sorted_arr.begin(), sorted_arr.end());

    bench_count_if(arr);
    bench_clamp(arr);
    bench_relu(arr);
    bench_sorted(sorted_arr);

    printf("\nKey takeaways:\n");
    printf("  Random data:  branchless wins (50%% miss rate → 15-20 wasted cycles/miss)\n");
    printf("  Sorted data:  branchy may win (predictor learns, cmov adds data dependency)\n");
    printf("  Rule of thumb: use branchless for unpredictable data-dependent conditions\n");
    printf("  Use branchy when: one branch is expensive (fn call, cache miss), or data sorted\n");

    ThreadPinner::unpin();
}
