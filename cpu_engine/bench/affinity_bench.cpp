#include "affinity/affinity.h"
#include "foundation/bench/bench.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <numeric>
#include <thread>
#include <vector>

using cpu_engine::ThreadPinner;

// ---------------------------------------------------------------------------
// Benchmark: TSC jitter on a pinned vs unpinned thread
//
// Method: take N back-to-back TSC samples. Between each sample, touch a small
// array (128 ints, 512 bytes, fits in L1) to give the CPU real work that
// would cost extra cycles after a cache-invalidating migration.
//
// Metric: stddev of inter-sample deltas (converted to ns).
//
// Expected results (Linux, hard pinning):
//   Pinned:   stddev ≈  10–50 ns  — only thermal/frequency noise
//   Unpinned: stddev ≈ 200–2000 ns — occasional migration restores I$, D$, BP
//
// macOS note: pinning is advisory. The OS may honour the affinity tag when
// the system is lightly loaded, but there is no guarantee. The numbers here
// will likely be similar for pinned vs unpinned; the benchmark still validates
// the API and shows the correct measuring methodology.
// ---------------------------------------------------------------------------

static constexpr int    kSamples     = 100'000;
static constexpr int    kWarmup      =  10'000;
static constexpr int    kWorkElems   =      128;  // 512 bytes — fits in L1

// Simple L1-resident workload: volatile sum to prevent elision.
static volatile int g_sink = 0;

static void do_work(int* arr) {
    int s = 0;
    for (int i = 0; i < kWorkElems; ++i) s += arr[i];
    g_sink = s;  // prevent dead-code elimination
}

struct Stats {
    double mean_ns;
    double stddev_ns;
    double p50_ns;
    double p99_ns;
    double p999_ns;
};

static Stats measure(bool pinned) {
    if (pinned) ThreadPinner::pin(0);
    else        ThreadPinner::unpin();

    // Warm-up: allow branch predictor and cache to settle.
    int arr[kWorkElems];
    for (int i = 0; i < kWorkElems; ++i) arr[i] = i;

    for (int i = 0; i < kWarmup; ++i) {
        do_work(arr);
        bench::tsc_now();
    }

    std::vector<double> deltas;
    deltas.reserve(kSamples);

    uint64_t prev = bench::tsc_now();
    for (int i = 0; i < kSamples; ++i) {
        do_work(arr);
        uint64_t now = bench::tsc_now();
        deltas.push_back(bench::tsc_to_ns(now - prev));
        prev = now;
    }

    std::sort(deltas.begin(), deltas.end());

    double mean = std::accumulate(deltas.begin(), deltas.end(), 0.0) / kSamples;
    double var  = 0.0;
    for (double d : deltas) var += (d - mean) * (d - mean);
    var /= kSamples;

    auto pct = [&](double p) {
        std::size_t idx = static_cast<std::size_t>(p / 100.0 * kSamples);
        if (idx >= deltas.size()) idx = deltas.size() - 1;
        return deltas[idx];
    };

    if (pinned) ThreadPinner::unpin();

    return { mean, std::sqrt(var), pct(50), pct(99), pct(99.9) };
}

int main() {
    printf("CPU affinity jitter benchmark\n");
    printf("Platform: ");
#if defined(__linux__)
    printf("Linux (hard pinning via pthread_setaffinity_np)\n");
#elif defined(__APPLE__)
    printf("macOS (advisory hint via thread_policy_set)\n");
#else
    printf("unknown\n");
#endif
    printf("Logical CPUs: %d\n", cpu_engine::cpu_count());
    printf("Samples: %d  Warmup: %d  Work: %d int reads per sample\n\n",
           kSamples, kWarmup, kWorkElems);

    // Force calibration before spawning threads so tsc_ticks_per_ns() is ready.
    (void)bench::tsc_ticks_per_ns();

    Stats pinned_stats{}, unpinned_stats{};

    std::thread t_pinned([&] {
        pinned_stats = measure(true);
    });
    t_pinned.join();

    std::thread t_unpinned([&] {
        unpinned_stats = measure(false);
    });
    t_unpinned.join();

    auto print = [](const char* label, const Stats& s) {
        printf("  %-10s  mean=%7.1f ns  stddev=%7.1f ns  "
               "p50=%7.1f ns  p99=%8.1f ns  p99.9=%9.1f ns\n",
               label, s.mean_ns, s.stddev_ns,
               s.p50_ns, s.p99_ns, s.p999_ns);
    };

    printf("Results (inter-sample delta, lower stddev = less jitter):\n");
    print("pinned",   pinned_stats);
    print("unpinned", unpinned_stats);

    double ratio = unpinned_stats.stddev_ns / (pinned_stats.stddev_ns + 1e-9);
    printf("\nUnpinned/pinned stddev ratio: %.1fx\n", ratio);

#if defined(__linux__)
    if (ratio < 2.0)
        printf("NOTE: ratio < 2x is unexpected on Linux with a loaded system.\n"
               "      Try 'taskset -c 1 ./affinity_bench' to reserve CPU 1.\n");
#else
    printf("NOTE: macOS affinity is advisory; ratio close to 1.0 is expected.\n"
           "      Run on Linux for hard-pinning results.\n");
#endif
}
