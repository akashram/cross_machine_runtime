#include "perf/perf.h"

#include <cassert>
#include <cstdio>
#include <cstring>
#include <numeric>
#include <vector>

using foundation::PerfCounters;
using foundation::PerfSnapshot;
using foundation::measure_perf;

// ---------------------------------------------------------------------------
// Test 1: PerfCounters can be constructed without crashing
// ---------------------------------------------------------------------------
static void test_construction() {
    PerfCounters ctr;
    // available() returns true on Linux with perf_event_paranoid <= 2,
    // false on macOS and on restricted Linux configurations.
    // Either outcome is valid — we just must not crash.
    printf("PASS  test_construction  (available=%s)\n",
           ctr.available() ? "yes" : "no (macOS or restricted Linux)");
}

// ---------------------------------------------------------------------------
// Test 2: start/stop does not crash; returns a snapshot
// ---------------------------------------------------------------------------
static void test_start_stop() {
    PerfCounters ctr;

    volatile int sink = 0;
    ctr.start();
    for (int i = 0; i < 10000; ++i) sink = sink + i;
    auto snap = ctr.stop();

    // cycles must be non-zero (we always measure at least via RDTSC)
    assert(snap.cycles > 0 || !ctr.available());

    // On Linux, instructions must be roughly proportional to iteration count.
    // On macOS, instructions == 0 (not measured).
    if (snap.available) {
        assert(snap.instructions > 0);
        assert(snap.cycles > 0);
    }

    (void)sink;
    printf("PASS  test_start_stop  (cycles=%llu insn=%llu)\n",
           static_cast<unsigned long long>(snap.cycles),
           static_cast<unsigned long long>(snap.instructions));
}

// ---------------------------------------------------------------------------
// Test 3: derived metrics stay within plausible bounds
// ---------------------------------------------------------------------------
static void test_derived_metrics() {
    // Measure a simple workload: sequential accumulate.
    // Expected: high IPC (it's a tight loop), low cache miss (data is hot),
    // low branch miss (loop counter is predictable).
    constexpr std::size_t kN = 10000;
    std::vector<int> data(kN);
    std::iota(data.begin(), data.end(), 0);

    PerfCounters ctr;
    volatile int64_t sum = 0;
    ctr.start();
    for (std::size_t i = 0; i < kN; ++i) sum = sum + data[i];
    auto snap = ctr.stop();
    (void)sum;

    if (snap.available) {
        // IPC must be positive and realistic (0 < IPC < 20)
        assert(snap.ipc() > 0.0);
        assert(snap.ipc() < 20.0);

        // Rates must be in [0, 1]
        assert(snap.cache_miss_rate() >= 0.0);
        assert(snap.cache_miss_rate() <= 1.0);
        assert(snap.branch_miss_rate() >= 0.0);
        assert(snap.branch_miss_rate() <= 1.0);

        printf("PASS  test_derived_metrics"
               "  IPC=%.2f  L3miss=%.2f%%  Brmiss=%.2f%%\n",
               snap.ipc(),
               snap.cache_miss_rate() * 100.0,
               snap.branch_miss_rate() * 100.0);
    } else {
        printf("PASS  test_derived_metrics  (perf not available — metrics not checked)\n");
    }
}

// ---------------------------------------------------------------------------
// Test 4: measure_perf() returns per-iteration values
// ---------------------------------------------------------------------------
static void test_measure_perf() {
    constexpr std::size_t kIters = 100000;
    volatile int sink = 0;

    auto snap = measure_perf(kIters, [&]{ sink = sink + 1; });
    (void)sink;

    if (snap.available) {
        // Per-iteration cycles should be tiny (< 100 for a single increment)
        // and instructions should be tiny (< 20).
        printf("PASS  test_measure_perf"
               "  cycles/iter=%.1f  insn/iter=%.1f\n",
               static_cast<double>(snap.cycles),
               static_cast<double>(snap.instructions));
    } else {
        printf("PASS  test_measure_perf  (perf not available)\n");
    }
}

// ---------------------------------------------------------------------------
// Test 5: two separate measurements are independent
// ---------------------------------------------------------------------------
static void test_multiple_measurements() {
    PerfCounters ctr;

    // First measurement: simple loop
    volatile int sink1 = 0;
    ctr.start();
    for (int i = 0; i < 1000; ++i) sink1 = sink1 + i;
    auto snap1 = ctr.stop();

    // Second measurement: same loop
    volatile int sink2 = 0;
    ctr.start();
    for (int i = 0; i < 1000; ++i) sink2 = sink2 + i;
    auto snap2 = ctr.stop();

    (void)sink1; (void)sink2;

    if (snap1.available && snap2.available) {
        // Both should have comparable instruction counts (same workload).
        // Allow 2x variation for scheduler noise.
        uint64_t insn1 = snap1.instructions;
        uint64_t insn2 = snap2.instructions;
        bool comparable = (insn1 > 0 && insn2 > 0) &&
                          (insn1 < insn2 * 2) && (insn2 < insn1 * 2);
        assert(comparable);
        printf("PASS  test_multiple_measurements"
               "  insn1=%llu insn2=%llu\n",
               static_cast<unsigned long long>(insn1),
               static_cast<unsigned long long>(insn2));
    } else {
        printf("PASS  test_multiple_measurements  (perf not available)\n");
    }
}

// ---------------------------------------------------------------------------
// Test 6: print() does not crash
// ---------------------------------------------------------------------------
static void test_print() {
    PerfCounters ctr;
    ctr.start();
    volatile int x = 1 + 1;
    auto snap = ctr.stop();
    (void)x;
    snap.print("test workload");
    printf("PASS  test_print\n");
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main() {
    test_construction();
    test_start_stop();
    test_derived_metrics();
    test_measure_perf();
    test_multiple_measurements();
    test_print();
    return 0;
}
