#pragma once

// CPU Affinity + Thread Pinning
// =========================================================================
//
// Scheduling jitter is the enemy of latency-sensitive code. An unpinned
// thread can be migrated by the OS to a different physical core at any point.
// Migration costs:
//   - L1/L2 cache cold: ~4–10 ns per line that must be re-fetched from L3
//   - Branch predictor reset: ~15–20 cycles per misprediction during warm-up
//   - TLB shootdown: if page table entries differ across cores
//   - TSC skew: on pre-invariant-TSC hardware (not a concern on modern x86,
//     but worth understanding)
//
// On Linux, pthread_setaffinity_np() is a hard binding — the kernel will
// never schedule the thread on a CPU outside the affinity set. This is the
// right tool for latency-critical threads (network receive, timer interrupt
// handler, benchmark threads).
//
// On macOS, thread_policy_set(THREAD_AFFINITY_POLICY) is advisory only.
// The scheduler hints that threads with the same affinity_tag should share
// an L2/L3 cache (i.e., be scheduled on sibling cores). The OS ignores the
// hint under memory pressure or scheduling imbalance. Hard pinning is not
// supported on macOS — this is a deliberate OS design choice for consumer
// hardware where there are no latency-critical dedicated cores.
//
//
// current_cpu() on macOS (x86)
// ----------------------------
// There is no sched_getcpu() on macOS. We use RDTSCP instead.
// The RDTSCP instruction writes the kernel-managed TSC_AUX MSR into ECX.
// On x86 Linux and macOS, the kernel sets TSC_AUX per-CPU to the logical
// CPU number. This is a standard OS convention, not just Linux — macOS XNU
// also sets it (see osfmk/i386/mp.c). ECX[15:0] gives the CPU index.
//
//
// Expected jitter (TSC stddev of fixed-cost loop iterations):
//   Pinned, Linux:   10–50 ns     (thermal noise, memory latency variance)
//   Unpinned, Linux: 200–2000 ns  (cache cold-start after migration)
//   macOS:           advisory only; may not show clear separation
//
// =========================================================================

#ifdef __linux__
#  include <pthread.h>
#  include <sched.h>
#  include <unistd.h>
#endif
#ifdef __APPLE__
#  include <mach/mach.h>
#  include <mach/thread_policy.h>
#  include <sys/sysctl.h>
#  include <pthread.h>
#endif

#include <cstddef>
#include <cstdint>

namespace cpu_engine {

// Returns the number of logical CPUs on this machine.
inline int cpu_count() noexcept {
#if defined(__linux__)
    return static_cast<int>(::sysconf(_SC_NPROCESSORS_CONF));
#elif defined(__APPLE__)
    int n = 1;
    std::size_t len = sizeof(n);
    ::sysctlbyname("hw.logicalcpu", &n, &len, nullptr, 0);
    return n;
#else
    return 1;
#endif
}

// Returns the logical CPU the calling thread is currently scheduled on.
// Linux: sched_getcpu() — typically a vsyscall, near-zero overhead.
// x86 macOS: reads ECX from RDTSCP — kernel writes CPU number to TSC_AUX.
// Returns -1 if the platform provides no way to determine this.
inline int current_cpu() noexcept {
#if defined(__linux__)
    return ::sched_getcpu();
#elif defined(__x86_64__)
    uint32_t cpu = 0;
    // RDTSCP writes TSC_AUX into ECX. We only need ECX (cpu), not the
    // timestamp (EAX:EDX), so we declare them as clobbers.
    __asm__ __volatile__(
        "rdtscp"
        : "=c"(cpu)
        :
        : "rax", "rdx"
    );
    return static_cast<int>(cpu & 0xFFFFu);
#else
    return -1;
#endif
}

// ThreadPinner — pin/unpin the calling thread to a specific logical CPU.
//
// Usage:
//   ThreadPinner::pin(2);   // hard-pin to CPU 2 (Linux) / hint (macOS)
//   // ... latency-sensitive work ...
//   ThreadPinner::unpin();  // restore scheduler freedom
//
// Thread safety: each method operates only on the calling thread's own
// affinity settings. No shared state — safe to call from any thread.
class ThreadPinner {
public:
    ThreadPinner() = delete;

    // Pin the calling thread to cpu_id.
    // Returns true on success, false on invalid cpu_id or syscall failure.
    static bool pin(int cpu_id) noexcept {
        if (cpu_id < 0 || cpu_id >= cpu_count()) return false;

#if defined(__linux__)
        cpu_set_t cs;
        CPU_ZERO(&cs);
        CPU_SET(static_cast<std::size_t>(cpu_id), &cs);
        return ::pthread_setaffinity_np(::pthread_self(), sizeof(cs), &cs) == 0;

#elif defined(__APPLE__)
        // affinity_tag must be non-zero to activate the hint.
        // We use cpu_id+1 so each CPU maps to a distinct tag (tag 0 = no preference).
        thread_affinity_policy_data_t policy{ cpu_id + 1 };
        return ::thread_policy_set(
            ::pthread_mach_thread_np(::pthread_self()),
            THREAD_AFFINITY_POLICY,
            reinterpret_cast<thread_policy_t>(&policy),
            THREAD_AFFINITY_POLICY_COUNT) == KERN_SUCCESS;
#else
        (void)cpu_id;
        return false;
#endif
    }

    // Remove affinity binding — allow the scheduler to use any CPU.
    static bool unpin() noexcept {
#if defined(__linux__)
        int n = static_cast<int>(::sysconf(_SC_NPROCESSORS_CONF));
        cpu_set_t cs;
        CPU_ZERO(&cs);
        for (int i = 0; i < n; ++i)
            CPU_SET(static_cast<std::size_t>(i), &cs);
        return ::pthread_setaffinity_np(::pthread_self(), sizeof(cs), &cs) == 0;

#elif defined(__APPLE__)
        thread_affinity_policy_data_t policy{ THREAD_AFFINITY_TAG_NULL };
        return ::thread_policy_set(
            ::pthread_mach_thread_np(::pthread_self()),
            THREAD_AFFINITY_POLICY,
            reinterpret_cast<thread_policy_t>(&policy),
            THREAD_AFFINITY_POLICY_COUNT) == KERN_SUCCESS;
#else
        return false;
#endif
    }

    // RAII wrapper: pin on construction, unpin on destruction.
    struct Guard {
        explicit Guard(int cpu_id) : ok_(pin(cpu_id)) {}
        ~Guard() { if (ok_) unpin(); }
        bool ok() const noexcept { return ok_; }
        Guard(const Guard&)            = delete;
        Guard& operator=(const Guard&) = delete;
    private:
        bool ok_;
    };
};

} // namespace cpu_engine
