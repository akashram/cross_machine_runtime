#pragma once

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <concepts>
#include <numeric>
#include <string>
#include <thread>
#include <vector>

// ---------------------------------------------------------------------------
// TSC timer
//
// RDTSCP is the serializing form of RDTSC: it waits for all prior instructions
// to retire before reading the counter, and it prevents the counter read from
// being reordered before prior instructions. This is critical for tight loops —
// without serialization, the CPU's out-of-order engine can move the timer reads
// outside the code you're trying to measure.
//
// On x86, TSC increments at the processor's nominal base frequency regardless
// of actual clock speed (the "invariant TSC" guarantee, CPUID 0x80000007 EDX
// bit 8). This is what lets us use it as a stable nanosecond source.
// ---------------------------------------------------------------------------

namespace bench {

inline uint64_t tsc_now() {
    uint32_t lo, hi;
    // RDTSCP also writes the TSC_AUX register (processor ID) into ECX, which
    // we discard. The __volatile__ and the memory clobber prevent the compiler
    // from reordering this asm past surrounding instructions.
    __asm__ __volatile__(
        "rdtscp"
        : "=a"(lo), "=d"(hi)
        :
        : "rcx", "memory"
    );
    return (static_cast<uint64_t>(hi) << 32) | lo;
}

// ---------------------------------------------------------------------------
// TSC frequency calibration
//
// We measure how many TSC ticks occur during a known wall-clock interval.
// std::chrono::steady_clock is not TSC-based on all platforms, but it is
// accurate enough over a 10ms window to give us a stable conversion factor.
// We call this once and cache the result — calibration itself takes ~10ms.
// ---------------------------------------------------------------------------

inline double tsc_ticks_per_ns() {
    static const double freq = []() -> double {
        using namespace std::chrono;
        const int samples = 5;
        double total = 0.0;
        for (int i = 0; i < samples; ++i) {
            auto wall_start = steady_clock::now();
            uint64_t tsc_start = tsc_now();
            std::this_thread::sleep_for(milliseconds(10));
            uint64_t tsc_end = tsc_now();
            auto wall_end = steady_clock::now();
            double ns = static_cast<double>(
                duration_cast<nanoseconds>(wall_end - wall_start).count());
            total += static_cast<double>(tsc_end - tsc_start) / ns;
        }
        return total / samples;
    }();
    return freq;
}

inline double tsc_to_ns(uint64_t ticks) {
    return static_cast<double>(ticks) / tsc_ticks_per_ns();
}

// ---------------------------------------------------------------------------
// Statistics
//
// We collect raw TSC-delta samples, convert to nanoseconds, then:
//   1. Sort the samples.
//   2. Remove outliers using the IQR (interquartile range) method:
//      any sample outside [Q1 - 1.5*IQR, Q3 + 1.5*IQR] is dropped.
//      This is Tukey's fences — the standard method in exploratory statistics.
//   3. Compute mean, stddev, and percentiles on the cleaned set.
//   4. Compute the 95% confidence interval on the mean using the
//      standard error: CI = mean ± 1.96 * (stddev / sqrt(n)).
//      1.96 is the z-score for 95% confidence under a normal distribution.
//      This is valid when n is large (n > 30 is the usual rule of thumb).
// ---------------------------------------------------------------------------

struct BenchResult {
    std::string name;
    double mean_ns;
    double stddev_ns;
    double p50_ns;
    double p95_ns;
    double p99_ns;
    double p999_ns;
    double ci95_low_ns;   // 95% confidence interval lower bound on mean
    double ci95_high_ns;
    size_t samples_used;  // after outlier removal
    size_t samples_total;
};

inline BenchResult compute_stats(std::string name, std::vector<double>& ns_samples) {
    std::sort(ns_samples.begin(), ns_samples.end());

    // IQR outlier removal
    size_t n = ns_samples.size();
    double q1 = ns_samples[n / 4];
    double q3 = ns_samples[3 * n / 4];
    double iqr = q3 - q1;
    double lo_fence = q1 - 1.5 * iqr;
    double hi_fence = q3 + 1.5 * iqr;

    std::vector<double> clean;
    clean.reserve(n);
    for (double v : ns_samples) {
        if (v >= lo_fence && v <= hi_fence) clean.push_back(v);
    }

    size_t m = clean.size();
    double mean = std::accumulate(clean.begin(), clean.end(), 0.0) / m;
    double variance = 0.0;
    for (double v : clean) variance += (v - mean) * (v - mean);
    variance /= m;
    double stddev = std::sqrt(variance);

    auto pct = [&](double p) -> double {
        size_t idx = static_cast<size_t>(p * m);
        if (idx >= m) idx = m - 1;
        return clean[idx];
    };

    double ci_half = 1.96 * (stddev / std::sqrt(static_cast<double>(m)));

    return BenchResult{
        .name         = std::move(name),
        .mean_ns      = mean,
        .stddev_ns    = stddev,
        .p50_ns       = pct(0.50),
        .p95_ns       = pct(0.95),
        .p99_ns       = pct(0.99),
        .p999_ns      = pct(0.999),
        .ci95_low_ns  = mean - ci_half,
        .ci95_high_ns = mean + ci_half,
        .samples_used  = m,
        .samples_total = n,
    };
}

// ---------------------------------------------------------------------------
// Main entry point
//
// run_bench(name, iterations, warmup_iters, fn)
//
// fn is called once per sample. Each sample measures a single call's TSC delta.
// Warmup runs fn `warmup_iters` times without recording — this lets the branch
// predictor and instruction cache reach steady state before we start timing.
// ---------------------------------------------------------------------------

template<std::invocable Fn>
inline BenchResult run_bench(
    std::string name,
    size_t iterations,
    size_t warmup_iters,
    Fn fn
) {
    // Warmup
    for (size_t i = 0; i < warmup_iters; ++i) fn();

    // Collect samples
    std::vector<double> ns_samples;
    ns_samples.reserve(iterations);
    for (size_t i = 0; i < iterations; ++i) {
        uint64_t t0 = tsc_now();
        fn();
        uint64_t t1 = tsc_now();
        ns_samples.push_back(tsc_to_ns(t1 - t0));
    }

    return compute_stats(std::move(name), ns_samples);
}

// ---------------------------------------------------------------------------
// Output helpers
// ---------------------------------------------------------------------------

inline void print_result(const BenchResult& r) {
    printf("%-30s  mean=%7.1f ns  p50=%7.1f  p99=%7.1f  p999=%7.1f  "
           "ci95=[%.1f, %.1f]  n=%zu/%zu\n",
           r.name.c_str(),
           r.mean_ns, r.p50_ns, r.p99_ns, r.p999_ns,
           r.ci95_low_ns, r.ci95_high_ns,
           r.samples_used, r.samples_total);
}

} // namespace bench
