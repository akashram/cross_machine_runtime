#pragma once

// x86 Hardware Performance Counter Infrastructure
// =========================================================================
//
// Hardware performance counters (PMCs) are special CPU registers that count
// architectural events: clock cycles, retired instructions, cache misses,
// branch mispredictions.  They reveal WHY a piece of code is slow in a way
// that wall-clock timing cannot:
//
//   "10 ns per alloc" tells you it's slow.
//   "IPC=0.3, cache_miss_rate=42%" tells you it's memory-bound.
//   "IPC=3.8, branch_miss_rate=18%" tells you it's branch-bound.
//
//
// Key metrics
// -----------
// IPC (instructions per cycle):
//   A modern out-of-order x86 core can retire ~4–6 instructions per cycle.
//   IPC < 1   → severe stall (memory-bound or long-latency dependency chain)
//   IPC 1–2   → some stalls; cache pressure or data dependencies
//   IPC 2–4   → good; CPU is mostly utilised
//   IPC > 4   → excellent; wide-issue execution, SIMD, or unrolled loops
//
// LLC miss rate (last-level cache misses / LLC references):
//   < 1%      → working set fits in L2/L3; no DRAM traffic
//   1–10%     → partial spill to DRAM; investigate working set size
//   > 10%     → memory-bound; cache-blocking, prefetching, or layout changes needed
//
// Branch miss rate (mispredictions / branches):
//   < 1%      → branch predictor handles the pattern well
//   1–5%      → some overhead; consider branchless alternatives for hot paths
//   > 5%      → costly; ~15–20 stall cycles per misprediction on modern x86
//
//
// Linux: perf_event_open(2)
// -------------------------
// The kernel exposes hardware counters via the perf_event_open() syscall.
// We open 6 events as a GROUP (leader fd = -1, follower fds = leader fd).
// Reading the leader with PERF_FORMAT_GROUP returns all event values
// atomically in one read(2) call.
//
// The kernel may multiplex events if more events are requested than the
// CPU has physical PMU registers (typically 4–8 on Intel).  It reports
// time_enabled / time_running so we can scale values proportionally.
// With only 6 events, multiplexing is unlikely on modern Intel/AMD.
//
// Privileges: on Linux, perf_event_paranoid ≤ 2 (the common default)
// allows unprivileged processes to measure their own threads
// (pid=0, cpu=-1 = current thread, any CPU).
//
// Events measured:
//   PERF_COUNT_HW_CPU_CYCLES          → retired CPU cycles
//   PERF_COUNT_HW_INSTRUCTIONS        → retired instructions
//   PERF_COUNT_HW_CACHE_REFERENCES    → LLC references (accesses to L3)
//   PERF_COUNT_HW_CACHE_MISSES        → LLC misses (went to DRAM)
//   PERF_COUNT_HW_BRANCH_INSTRUCTIONS → branch instructions
//   PERF_COUNT_HW_BRANCH_MISSES       → mispredicted branches
//
//
// macOS reality
// -------------
// macOS has no perf_event_open equivalent accessible from user space without
// Apple private entitlements.  The kernel performance counter API (kpc/kperf)
// is private and requires a macOS developer entitlement not available to
// third-party code.
//
// On macOS we fall back to RDTSC for wall-clock cycles only.  IPC, cache
// miss rate, and branch miss rate are not available.  PerfCounters::available()
// returns false; all snapshot fields except cycles are zero.
//
// Run the full benchmark suite on a Linux cloud instance (e.g. AWS c5.2xlarge)
// to get meaningful hardware counter data for Phase 2+ optimisation.
//
// =========================================================================

#include <cstdint>
#include <cstdio>
#include <cstring>

#ifdef __linux__
#  include <linux/perf_event.h>
#  include <sys/ioctl.h>
#  include <sys/syscall.h>
#  include <unistd.h>
#endif

namespace foundation {

// =========================================================================
// PerfSnapshot — a single measurement of all hardware counters
// =========================================================================
struct PerfSnapshot {
    uint64_t cycles{0};
    uint64_t instructions{0};
    uint64_t cache_refs{0};    // LLC references
    uint64_t cache_misses{0};  // LLC misses (went to DRAM)
    uint64_t branches{0};
    uint64_t branch_misses{0};
    bool     available{false}; // false on macOS / unsupported kernel config

    // Derived metrics -------------------------------------------------------

    // Instructions per cycle.  0 if not available or cycles == 0.
    double ipc() const noexcept {
        return (available && cycles > 0)
            ? static_cast<double>(instructions) / static_cast<double>(cycles)
            : 0.0;
    }

    // LLC miss rate as a fraction [0, 1].  0 if not available.
    double cache_miss_rate() const noexcept {
        return (available && cache_refs > 0)
            ? static_cast<double>(cache_misses) / static_cast<double>(cache_refs)
            : 0.0;
    }

    // Branch misprediction rate as a fraction [0, 1].  0 if not available.
    double branch_miss_rate() const noexcept {
        return (available && branches > 0)
            ? static_cast<double>(branch_misses) / static_cast<double>(branches)
            : 0.0;
    }

    void print(const char* label = nullptr) const noexcept {
        if (label) std::printf("  [%s]\n", label);
        if (!available) {
            std::printf("  perf counters: not available"
                        " (macOS: no user-space PMC API)\n");
            return;
        }
        std::printf("  IPC=%.2f  L3miss=%.2f%%  Brmiss=%.2f%%"
                    "  cycles=%llu  insn=%llu\n",
                    ipc(),
                    cache_miss_rate() * 100.0,
                    branch_miss_rate() * 100.0,
                    static_cast<unsigned long long>(cycles),
                    static_cast<unsigned long long>(instructions));
    }
};

// =========================================================================
// PerfCounters — open/start/stop hardware counters
// =========================================================================
class PerfCounters {
public:
    PerfCounters() noexcept {
#ifdef __linux__
        init_linux();
#endif
    }

    ~PerfCounters() noexcept {
#ifdef __linux__
        for (int i = 0; i < kNumEvents; ++i)
            if (fds_[i] >= 0) ::close(fds_[i]);
#endif
    }

    PerfCounters(const PerfCounters&)            = delete;
    PerfCounters& operator=(const PerfCounters&) = delete;

    bool available() const noexcept { return available_; }

    // Reset and enable all counters.
    void start() noexcept {
#ifdef __linux__
        if (!available_) return;
        ::ioctl(fds_[0], PERF_EVENT_IOC_RESET,  PERF_IOC_FLAG_GROUP);
        ::ioctl(fds_[0], PERF_EVENT_IOC_ENABLE, PERF_IOC_FLAG_GROUP);
#endif
        tsc_start_ = rdtsc();
    }

    // Disable counters and return a snapshot.
    PerfSnapshot stop() noexcept {
        uint64_t tsc_end = rdtsc();
        PerfSnapshot snap;

#ifdef __linux__
        if (!available_) {
            snap.cycles = tsc_end - tsc_start_;
            return snap;
        }
        ::ioctl(fds_[0], PERF_EVENT_IOC_DISABLE, PERF_IOC_FLAG_GROUP);
        snap = read_group();
#else
        snap.cycles = tsc_end - tsc_start_;
#endif
        return snap;
    }

    // Read current values without stopping.
    PerfSnapshot read() noexcept {
#ifdef __linux__
        if (available_) return read_group();
#endif
        PerfSnapshot snap;
        snap.cycles = rdtsc() - tsc_start_;
        return snap;
    }

private:
    static constexpr int kNumEvents = 6;

    bool     available_{false};
    uint64_t tsc_start_{0};

#ifdef __linux__
    int fds_[kNumEvents]{-1, -1, -1, -1, -1, -1};

    static long perf_event_open_syscall(struct perf_event_attr* attr,
                                        pid_t pid, int cpu,
                                        int group_fd, unsigned long flags) noexcept {
        return ::syscall(SYS_perf_event_open, attr, pid, cpu, group_fd, flags);
    }

    void init_linux() noexcept {
        static const uint32_t kConfigs[kNumEvents] = {
            PERF_COUNT_HW_CPU_CYCLES,
            PERF_COUNT_HW_INSTRUCTIONS,
            PERF_COUNT_HW_CACHE_REFERENCES,
            PERF_COUNT_HW_CACHE_MISSES,
            PERF_COUNT_HW_BRANCH_INSTRUCTIONS,
            PERF_COUNT_HW_BRANCH_MISSES,
        };

        struct perf_event_attr attr;
        std::memset(&attr, 0, sizeof(attr));
        attr.type           = PERF_TYPE_HARDWARE;
        attr.size           = sizeof(attr);
        attr.disabled       = 1;
        attr.exclude_kernel = 1;
        attr.exclude_hv     = 1;
        // PERF_FORMAT_GROUP lets us read all events with one read() call.
        // TOTAL_TIME_* lets us detect and correct for multiplexing.
        attr.read_format = PERF_FORMAT_GROUP |
                           PERF_FORMAT_TOTAL_TIME_ENABLED |
                           PERF_FORMAT_TOTAL_TIME_RUNNING;

        // Open group leader (event 0) with group_fd = -1.
        attr.config = kConfigs[0];
        fds_[0] = static_cast<int>(perf_event_open_syscall(&attr, 0, -1, -1, 0));
        if (fds_[0] < 0) return;  // insufficient privilege or unsupported event

        // Open remaining events as followers in the same group.
        attr.disabled    = 0;  // followers start/stop with leader
        attr.read_format = 0;  // followers don't need group format
        for (int i = 1; i < kNumEvents; ++i) {
            attr.config = kConfigs[i];
            fds_[i] = static_cast<int>(
                perf_event_open_syscall(&attr, 0, -1, fds_[0], 0));
            if (fds_[i] < 0) {
                // Couldn't open this event — close what we have and bail.
                for (int j = 0; j < i; ++j) { ::close(fds_[j]); fds_[j] = -1; }
                return;
            }
        }
        available_ = true;
    }

    // Read all 6 events from the group leader in one syscall.
    // Scales for multiplexing if time_running < time_enabled.
    PerfSnapshot read_group() noexcept {
        // Layout when PERF_FORMAT_GROUP | PERF_FORMAT_TOTAL_TIME_{ENABLED,RUNNING}:
        //   uint64_t nr            (number of events)
        //   uint64_t time_enabled
        //   uint64_t time_running
        //   uint64_t value[nr]     (one per event, in open order)
        struct {
            uint64_t nr;
            uint64_t time_enabled;
            uint64_t time_running;
            uint64_t values[kNumEvents];
        } data{};

        if (::read(fds_[0], &data, sizeof(data)) <= 0) return {};

        double scale = (data.time_running > 0)
            ? static_cast<double>(data.time_enabled) /
              static_cast<double>(data.time_running)
            : 1.0;

        auto scaled = [&](int i) -> uint64_t {
            return static_cast<uint64_t>(
                static_cast<double>(data.values[i]) * scale);
        };

        PerfSnapshot s;
        s.available      = true;
        s.cycles         = scaled(0);
        s.instructions   = scaled(1);
        s.cache_refs     = scaled(2);
        s.cache_misses   = scaled(3);
        s.branches       = scaled(4);
        s.branch_misses  = scaled(5);
        return s;
    }
#endif // __linux__

    static uint64_t rdtsc() noexcept {
#if defined(__x86_64__)
        uint32_t lo, hi;
        __asm__ __volatile__("rdtscp" : "=a"(lo), "=d"(hi) : : "rcx", "memory");
        return (static_cast<uint64_t>(hi) << 32) | lo;
#elif defined(__aarch64__)
        uint64_t v;
        __asm__ __volatile__("mrs %0, cntvct_el0" : "=r"(v));
        return v;
#else
        return 0;
#endif
    }
};

// =========================================================================
// Convenience: measure a callable N times and return a PerfSnapshot
// with per-iteration values.
// =========================================================================
//
// Usage:
//   auto snap = measure_perf(100'000, [&]{ /* work */ });
//   if (snap.available)
//       printf("IPC=%.2f  L3miss=%.2f%%\n", snap.ipc(), snap.cache_miss_rate()*100);

template<typename Fn>
PerfSnapshot measure_perf(std::size_t iterations, Fn fn) noexcept {
    PerfCounters ctr;
    for (std::size_t i = 0; i < iterations / 10; ++i) fn();  // warmup
    ctr.start();
    for (std::size_t i = 0; i < iterations; ++i) fn();
    auto snap = ctr.stop();

    // Normalise to per-iteration counts.
    if (snap.available && iterations > 0) {
        auto n = iterations;
        snap.cycles        /= n;
        snap.instructions  /= n;
        snap.cache_refs    /= n;
        snap.cache_misses  /= n;
        snap.branches      /= n;
        snap.branch_misses /= n;
    }
    return snap;
}

} // namespace foundation
