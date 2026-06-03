#include "affinity/affinity.h"
#include "foundation/bench/bench.h"

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <time.h>
#include <vector>

using cpu_engine::ThreadPinner;

// ---------------------------------------------------------------------------
// Scheduling jitter measurement — the same metric cyclictest measures
//
// Method: ask the OS to sleep for exactly `target_us` microseconds via
// clock_nanosleep(CLOCK_MONOTONIC, TIMER_ABSTIME, ...).  Measure the
// actual elapsed time with RDTSCP.  The excess over target_us is the
// OS scheduling latency ("jitter").
//
// Sources of jitter (each addressed by one tuning script):
//
//   Kernel timer tick (HZ=250 default):
//     Linux wakes sleeping threads on the next timer tick, not immediately
//     when the deadline expires.  With CONFIG_HZ=250, ticks fire every 4 ms.
//     A 200 us sleep can wake up 0–4 ms late depending on tick phase.
//     fix: nohz_full=<cpus> — disables the tick on isolated CPUs so the
//          next wakeup fires on the exact timer expiry (tickless kernel).
//
//   Interrupts (IRQs):
//     Any IRQ handled on the measurement CPU adds latency.  Network, disk,
//     timer, IPI, and CPU frequency change interrupts all fire unexpectedly.
//     fix: irq affinity — move all IRQs to housekeeping CPUs, leaving the
//          isolated CPU interrupt-free except for local timer and IPI.
//
//   CPU C-states (power management):
//     When idle, the CPU enters C2/C3/C6 sleep states to save power.
//     Exiting a C-state adds 10–200 us of latency before the CPU can
//     retire instructions again.
//     fix: disable C-states deeper than C1 on isolated CPUs.
//
//   CPU frequency scaling (P-states):
//     The CPU may run at a lower frequency when waking from idle, then
//     ramp up over ~10–50 us.  This shows up as variable per-iteration time.
//     fix: set the CPU governor to "performance" (fixed maximum frequency).
//
//   Scheduler migration:
//     An unpinned thread can be moved to a different CPU while sleeping.
//     The wakeup must then migrate it back, adding latency.
//     fix: isolcpus=<cpus> — kernel scheduler never places tasks there
//          unless explicitly asked (via CPU affinity), and ThreadPinner::pin().
//
//
// Expected results:
//   Baseline (no tuning):          p99 jitter  500–5000 us
//   governor=performance + C1:     p99 jitter   50–200  us
//   + IRQ affinity:                p99 jitter   10–50   us
//   + isolcpus + nohz_full:        p99 jitter    2–10   us
//
// macOS: advisory pinning only, no C-state or IRQ control; jitter numbers
// will reflect macOS scheduler behaviour, not the tuning effect.
// ---------------------------------------------------------------------------

static constexpr long   kTargetUs    = 200;       // sleep target in microseconds
static constexpr int    kIterations  = 2000;
static constexpr int    kWarmup      = 100;

static long timespec_diff_us(const struct timespec& a, const struct timespec& b) {
    return (a.tv_sec - b.tv_sec) * 1'000'000L +
           (a.tv_nsec - b.tv_nsec) / 1'000L;
}

int main(int argc, char** argv) {
    int cpu = (argc > 1) ? std::atoi(argv[1]) : 0;

    printf("Scheduling jitter benchmark\n");
    printf("Platform: ");
#ifdef __linux__
    printf("Linux\n");
#else
    printf("macOS (advisory pinning; no C-state/IRQ control)\n");
#endif
    printf("Target sleep: %ld us  Iterations: %d  CPU: %d\n\n",
           kTargetUs, kIterations, cpu);

    bool pinned = ThreadPinner::pin(cpu);
    if (!pinned)
        printf("WARNING: could not pin to CPU %d — results will include migration jitter\n\n", cpu);

    (void)bench::tsc_ticks_per_ns();  // calibrate before measuring

    std::vector<long> jitter_us;
    jitter_us.reserve(kIterations);

    struct timespec now{};
    clock_gettime(CLOCK_MONOTONIC, &now);

    // Warmup: let the CPU frequency stabilise and branch predictor settle
    for (int i = 0; i < kWarmup; ++i) {
        now.tv_nsec += kTargetUs * 1000L;
        if (now.tv_nsec >= 1'000'000'000L) {
            now.tv_nsec -= 1'000'000'000L;
            now.tv_sec  += 1;
        }
#ifdef __linux__
        clock_nanosleep(CLOCK_MONOTONIC, TIMER_ABSTIME, &now, nullptr);
#else
        struct timespec rem{};
        nanosleep(&now, &rem);  // macOS: relative sleep (less accurate)
#endif
    }

    // Measurement loop
    clock_gettime(CLOCK_MONOTONIC, &now);
    for (int i = 0; i < kIterations; ++i) {
        // Advance deadline by exactly target_us
        now.tv_nsec += kTargetUs * 1000L;
        if (now.tv_nsec >= 1'000'000'000L) {
            now.tv_nsec -= 1'000'000'000L;
            now.tv_sec  += 1;
        }

        struct timespec before{};
        clock_gettime(CLOCK_MONOTONIC, &before);

#ifdef __linux__
        clock_nanosleep(CLOCK_MONOTONIC, TIMER_ABSTIME, &now, nullptr);
#else
        // macOS lacks clock_nanosleep; use nanosleep with the target duration.
        // This is relative sleep — accuracy is lower than TIMER_ABSTIME.
        struct timespec req{ 0, kTargetUs * 1000L };
        struct timespec rem{};
        nanosleep(&req, &rem);
#endif

        struct timespec after{};
        clock_gettime(CLOCK_MONOTONIC, &after);

        long actual_us = timespec_diff_us(after, before);
        long overshoot = actual_us - kTargetUs;
        // Clamp negative jitter (clock resolution) to 0 to avoid skewing stats
        jitter_us.push_back(overshoot > 0 ? overshoot : 0);
    }

    // Statistics
    std::sort(jitter_us.begin(), jitter_us.end());
    double mean = static_cast<double>(
        std::accumulate(jitter_us.begin(), jitter_us.end(), 0LL)) / kIterations;

    double var = 0.0;
    for (long v : jitter_us) {
        double d = static_cast<double>(v) - mean;
        var += d * d;
    }
    double stddev = std::sqrt(var / kIterations);

    auto pct = [&](double p) {
        std::size_t idx = static_cast<std::size_t>(p / 100.0 * kIterations);
        if (idx >= jitter_us.size()) idx = jitter_us.size() - 1;
        return jitter_us[idx];
    };

    printf("Jitter (overshoot past target %ld us):\n", kTargetUs);
    printf("  min=%4ld us  mean=%6.1f us  stddev=%6.1f us  "
           "p50=%4ld us  p99=%5ld us  p99.9=%6ld us  max=%5ld us\n",
           jitter_us.front(), mean, stddev,
           pct(50), pct(99), pct(99.9),
           jitter_us.back());

    printf("\nInterpretation:\n");
    long p99 = pct(99);
    if (p99 < 10)
        printf("  EXCELLENT — p99 < 10 us. OS tuning is fully applied.\n");
    else if (p99 < 50)
        printf("  GOOD      — p99 < 50 us. Governor + C-states tuned; consider isolcpus.\n");
    else if (p99 < 200)
        printf("  OK        — p99 < 200 us. IRQ affinity may help.\n");
    else
        printf("  BASELINE  — p99 >= 200 us. Apply os_tuning scripts to improve.\n");

    if (pinned) ThreadPinner::unpin();
}
