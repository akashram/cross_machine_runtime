#include "affinity/affinity.h"

#include <cassert>
#include <cstdio>
#include <latch>
#include <thread>

using cpu_engine::cpu_count;
using cpu_engine::current_cpu;
using cpu_engine::ThreadPinner;

// ---------------------------------------------------------------------------
// Test 1: cpu_count() is sane
// ---------------------------------------------------------------------------
static void test_cpu_count() {
    int n = cpu_count();
    assert(n >= 1);
    printf("PASS  test_cpu_count  (logical_cpus=%d)\n", n);
}

// ---------------------------------------------------------------------------
// Test 2: pin to each valid CPU succeeds
// ---------------------------------------------------------------------------
static void test_pin_each_cpu() {
    int n = cpu_count();
    for (int i = 0; i < n; ++i) {
        bool ok = ThreadPinner::pin(i);
        assert(ok);
    }
    ThreadPinner::unpin();
    printf("PASS  test_pin_each_cpu  (tested %d CPUs)\n", n);
}

// ---------------------------------------------------------------------------
// Test 3: pin to negative or out-of-range CPU returns false
// ---------------------------------------------------------------------------
static void test_pin_invalid() {
    assert(!ThreadPinner::pin(-1));
    assert(!ThreadPinner::pin(cpu_count()));   // one past the end
    assert(!ThreadPinner::pin(1 << 20));       // absurdly large
    printf("PASS  test_pin_invalid\n");
}

// ---------------------------------------------------------------------------
// Test 4: unpin succeeds
// ---------------------------------------------------------------------------
static void test_unpin() {
    bool ok_pin   = ThreadPinner::pin(0);
    bool ok_unpin = ThreadPinner::unpin();
    assert(ok_pin);
    assert(ok_unpin);
    printf("PASS  test_unpin\n");
}

// ---------------------------------------------------------------------------
// Test 5: RAII Guard restores scheduling on scope exit
// ---------------------------------------------------------------------------
static void test_guard() {
    {
        ThreadPinner::Guard g(0);
        assert(g.ok());
    }
    // After guard destructor, unpin has been called.
    // We can't easily assert scheduling freedom, but if it didn't crash
    // and the next pin still works, the Guard cleaned up correctly.
    assert(ThreadPinner::pin(0));
    ThreadPinner::unpin();
    printf("PASS  test_guard\n");
}

// ---------------------------------------------------------------------------
// Test 6: current_cpu() returns a valid index (or -1 on unsupported platform)
// ---------------------------------------------------------------------------
static void test_current_cpu() {
    int cpu = current_cpu();
    int n   = cpu_count();
    // -1 is acceptable on platforms without a current-cpu API.
    assert(cpu == -1 || (cpu >= 0 && cpu < n));
    printf("PASS  test_current_cpu  (cpu=%d)\n", cpu);
}

// ---------------------------------------------------------------------------
// Test 7: pin/unpin work correctly from a spawned thread
// ---------------------------------------------------------------------------
static void test_pin_from_thread() {
    std::latch done{1};
    bool thread_ok = false;

    std::thread t([&] {
        bool p = ThreadPinner::pin(0);
        bool u = ThreadPinner::unpin();
        thread_ok = p && u;
        done.count_down();
    });
    done.wait();
    t.join();

    assert(thread_ok);
    printf("PASS  test_pin_from_thread\n");
}

// ---------------------------------------------------------------------------
// Test 8: Linux only — verify sched_getcpu() reflects the pinned CPU
//
// macOS is advisory-only so we cannot guarantee current_cpu() == pinned_cpu
// after thread_policy_set.  We skip that assertion on Apple.
// ---------------------------------------------------------------------------
static void test_pin_verifies_cpu() {
#if defined(__linux__)
    int n = cpu_count();
    for (int target = 0; target < n; ++target) {
        ThreadPinner::pin(target);
        int got = current_cpu();
        assert(got == target);
    }
    ThreadPinner::unpin();
    printf("PASS  test_pin_verifies_cpu  (%d CPUs verified)\n", n);
#else
    printf("SKIP  test_pin_verifies_cpu  (advisory-only on macOS)\n");
#endif
}

int main() {
    test_cpu_count();
    test_pin_each_cpu();
    test_pin_invalid();
    test_unpin();
    test_guard();
    test_current_cpu();
    test_pin_from_thread();
    test_pin_verifies_cpu();
    printf("\nAll affinity tests passed.\n");
}
