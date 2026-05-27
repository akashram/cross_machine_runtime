#include "bench.h"
#include <cstdio>
#include <thread>
#include <chrono>

// Prevents the compiler from optimizing away a value entirely.
// Without this, the optimizer sees that the result of a computation is never
// used and deletes the computation — giving you a benchmark of "nothing".
// Writing to a volatile variable forces the value to be materialized.
// This is the standard low-tech alternative to compiler-specific escape
// intrinsics like __asm__ __volatile__("" : "+r"(x)).
template<typename T>
inline void do_not_optimize(T const& val) {
    volatile T sink = val;
    (void)sink;
}

int main() {
    // Print the calibrated frequency so we can sanity-check it.
    // On a modern Intel/Apple Silicon Mac at ~3GHz you should see ~3.0 ticks/ns.
    // If this prints something wildly off (e.g. 0.1 or 300), the calibration
    // or the TSC read is broken.
    printf("TSC frequency: %.3f ticks/ns\n\n", bench::tsc_ticks_per_ns());

    // --- Benchmark 1: harness overhead floor ---
    //
    // The lambda does nothing. The measured time is the cost of two RDTSCP
    // instructions plus the call overhead. This is the noise floor — any
    // real benchmark result should be larger than this. On a 3GHz machine
    // expect roughly 5–15 ns.
    auto result_nop = bench::run_bench("nop (harness floor)", 100'000, 1'000,
        []() {
            // nothing
        });
    bench::print_result(result_nop);

    // --- Benchmark 2: single integer addition ---
    //
    // One ALU operation. The actual add takes <1ns, so what you're measuring
    // here is still dominated by harness overhead. The interesting thing is
    // comparing it to the nop above — it should be indistinguishable, because
    // the add gets folded into the surrounding loop by the CPU's out-of-order
    // engine. do_not_optimize forces the result to be written to memory,
    // which adds a store but prevents the computation from disappearing.
    int x = 1;
    auto result_add = bench::run_bench("integer add + store", 100'000, 1'000,
        [&x]() {
            do_not_optimize(x + 1);
        });
    bench::print_result(result_add);

    // --- Benchmark 3: sleep_for(1ns) ---
    //
    // sleep_for is NOT a nanosecond-precision sleep on any real OS. On macOS,
    // the scheduler granularity is ~1ms (1,000,000 ns). This benchmark is here
    // to verify that the harness can faithfully measure *long* durations — if
    // the mean comes back near 1,000,000 ns, the timer is working correctly at
    // scale. The p99 will be much higher due to scheduler jitter.
    auto result_sleep = bench::run_bench("sleep_for(1ns) [expect ~10-1000us on macOS]", 100, 5,
        []() {
            std::this_thread::sleep_for(std::chrono::nanoseconds(1));
        });
    bench::print_result(result_sleep);

    return 0;
}
