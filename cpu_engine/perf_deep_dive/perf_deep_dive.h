#pragma once

// Extended hardware performance counter set for CPU backend deep-dive
// ====================================================================
//
// The foundation PerfCounters (foundation/perf/perf.h) measures six
// PERF_TYPE_HARDWARE events: cycles, instructions, LLC refs/misses, branches,
// branch misses.  That gives IPC and L3 miss rate but not the full cache
// hierarchy.
//
// This header adds L1D and L2 miss rates using PERF_TYPE_HW_CACHE events,
// giving the complete picture:
//
//   L1D read accesses  — total load traffic from the L1 data cache
//   L1D read misses    — loads that missed L1 (→ L2 lookup)
//   LLC read accesses  — loads that reached the L3 (= L2 misses)
//   LLC read misses    — loads that went to DRAM (= L3 misses)
//
// Derived miss rates:
//   L1 miss %  = L1D_misses  / L1D_accesses
//   L2 miss %  = LLC_accesses / L1D_misses    (fraction of L2 lookups that go to L3)
//   L3 miss %  = LLC_misses  / LLC_accesses
//
// Implementation notes
// --------------------
// PERF_TYPE_HW_CACHE events cannot be grouped with PERF_TYPE_HARDWARE events
// in a single perf_event_open() group on all kernel versions.  We therefore
// open all 8 events as *independent* file descriptors and ioctl them together
// (PERF_EVENT_IOC_ENABLE / DISABLE with no PERF_IOC_FLAG_GROUP).  Reads are
// slightly non-atomic but accurate enough for multi-pass benchmarks.
//
// Each fd uses PERF_FORMAT_TOTAL_TIME_ENABLED | PERF_FORMAT_TOTAL_TIME_RUNNING
// so we can scale for multiplexing (the PMU has ~4 HW counters; with 8 events
// each counter runs ~50% of the time and the kernel scales accordingly).
//
// On macOS: perf_event_open() is unavailable.  available() returns false;
// all snapshot fields are 0.  Run on a Linux cloud instance for real data.

#include <cstdint>
#include <cstdio>
#include <cstring>

#ifdef __linux__
#  include <linux/perf_event.h>
#  include <sys/ioctl.h>
#  include <sys/syscall.h>
#  include <unistd.h>
#endif

namespace cpu_engine::perf_deep_dive {

struct DeepDiveSnapshot {
    uint64_t instructions{0};
    uint64_t cycles{0};
    uint64_t l1d_accesses{0};   // L1D read accesses
    uint64_t l1d_misses{0};     // L1D read misses  (= L2 accesses)
    uint64_t llc_accesses{0};   // LLC read accesses (= L2 misses / L3 accesses)
    uint64_t llc_misses{0};     // LLC read misses   (= DRAM accesses)
    uint64_t branches{0};
    uint64_t branch_misses{0};
    bool     available{false};

    double ipc()            const noexcept {
        return (available && cycles > 0)
            ? static_cast<double>(instructions) / static_cast<double>(cycles)
            : 0.0;
    }
    double l1_miss_pct()    const noexcept {
        return (available && l1d_accesses > 0)
            ? 100.0 * static_cast<double>(l1d_misses) / static_cast<double>(l1d_accesses)
            : 0.0;
    }
    // Fraction of L1D misses that also miss L2 (reach L3)
    double l2_miss_pct()    const noexcept {
        return (available && l1d_misses > 0)
            ? 100.0 * static_cast<double>(llc_accesses) / static_cast<double>(l1d_misses)
            : 0.0;
    }
    double l3_miss_pct()    const noexcept {
        return (available && llc_accesses > 0)
            ? 100.0 * static_cast<double>(llc_misses) / static_cast<double>(llc_accesses)
            : 0.0;
    }
    double branch_miss_pct() const noexcept {
        return (available && branches > 0)
            ? 100.0 * static_cast<double>(branch_misses) / static_cast<double>(branches)
            : 0.0;
    }

    void print_header() const noexcept {
        printf("  %-32s  %5s  %6s  %6s  %6s  %6s  %6s\n",
               "kernel", "IPC", "L1mis%", "L2mis%", "L3mis%", "Brm%", "avail");
    }

    void print_row(const char* label) const noexcept {
        if (!available) {
            printf("  %-32s  %5s  %6s  %6s  %6s  %6s  macOS\n",
                   label, "n/a", "n/a", "n/a", "n/a", "n/a");
        } else {
            printf("  %-32s  %5.2f  %5.1f%%  %5.1f%%  %5.1f%%  %5.1f%%  ok\n",
                   label, ipc(),
                   l1_miss_pct(), l2_miss_pct(), l3_miss_pct(),
                   branch_miss_pct());
        }
    }
};

class ExtendedPerfCounters {
public:
    ExtendedPerfCounters() noexcept {
#ifdef __linux__
        open_events();
#endif
    }

    ~ExtendedPerfCounters() noexcept {
#ifdef __linux__
        for (int i = 0; i < kN; ++i)
            if (fds_[i] >= 0) ::close(fds_[i]);
#endif
    }

    ExtendedPerfCounters(const ExtendedPerfCounters&)            = delete;
    ExtendedPerfCounters& operator=(const ExtendedPerfCounters&) = delete;

    bool available() const noexcept { return available_; }

    void start() noexcept {
#ifdef __linux__
        if (!available_) return;
        for (int i = 0; i < kN; ++i)
            ::ioctl(fds_[i], PERF_EVENT_IOC_RESET,  0);
        for (int i = 0; i < kN; ++i)
            ::ioctl(fds_[i], PERF_EVENT_IOC_ENABLE, 0);
#endif
    }

    DeepDiveSnapshot stop() noexcept {
#ifdef __linux__
        if (!available_) return {};
        for (int i = 0; i < kN; ++i)
            ::ioctl(fds_[i], PERF_EVENT_IOC_DISABLE, 0);
        return read_all();
#else
        return {};
#endif
    }

private:
    static constexpr int kN = 8;
    bool available_{false};

#ifdef __linux__
    int fds_[kN]{-1,-1,-1,-1,-1,-1,-1,-1};

    static long perf_open(struct perf_event_attr* attr) noexcept {
        return ::syscall(SYS_perf_event_open, attr,
                         /*pid=*/0, /*cpu=*/-1, /*group_fd=*/-1, /*flags=*/0);
    }

    static uint64_t hw_cache_config(uint32_t cache, uint32_t op,
                                    uint32_t result) noexcept {
        return cache | (op << 8) | (result << 16);
    }

    void open_events() noexcept {
        struct perf_event_attr attr{};
        attr.size           = sizeof(attr);
        attr.disabled       = 1;
        attr.exclude_kernel = 1;
        attr.exclude_hv     = 1;
        // Individual read format — scaling for multiplexing
        attr.read_format = PERF_FORMAT_TOTAL_TIME_ENABLED |
                           PERF_FORMAT_TOTAL_TIME_RUNNING;

        // --- PERF_TYPE_HARDWARE events ---
        struct { uint32_t type; uint64_t config; } hw_evts[4] = {
            {PERF_TYPE_HARDWARE, PERF_COUNT_HW_INSTRUCTIONS},
            {PERF_TYPE_HARDWARE, PERF_COUNT_HW_CPU_CYCLES},
            {PERF_TYPE_HARDWARE, PERF_COUNT_HW_BRANCH_INSTRUCTIONS},
            {PERF_TYPE_HARDWARE, PERF_COUNT_HW_BRANCH_MISSES},
        };
        for (int i = 0; i < 4; ++i) {
            attr.type   = hw_evts[i].type;
            attr.config = hw_evts[i].config;
            fds_[i] = static_cast<int>(perf_open(&attr));
            if (fds_[i] < 0) return; // not enough privilege
        }

        // --- PERF_TYPE_HW_CACHE events ---
        struct { uint32_t cache; uint32_t op; uint32_t result; } cache_evts[4] = {
            {PERF_COUNT_HW_CACHE_L1D, PERF_COUNT_HW_CACHE_OP_READ, PERF_COUNT_HW_CACHE_RESULT_ACCESS},
            {PERF_COUNT_HW_CACHE_L1D, PERF_COUNT_HW_CACHE_OP_READ, PERF_COUNT_HW_CACHE_RESULT_MISS},
            {PERF_COUNT_HW_CACHE_LL,  PERF_COUNT_HW_CACHE_OP_READ, PERF_COUNT_HW_CACHE_RESULT_ACCESS},
            {PERF_COUNT_HW_CACHE_LL,  PERF_COUNT_HW_CACHE_OP_READ, PERF_COUNT_HW_CACHE_RESULT_MISS},
        };
        for (int i = 0; i < 4; ++i) {
            attr.type   = PERF_TYPE_HW_CACHE;
            attr.config = hw_cache_config(cache_evts[i].cache,
                                          cache_evts[i].op,
                                          cache_evts[i].result);
            fds_[4 + i] = static_cast<int>(perf_open(&attr));
            if (fds_[4 + i] < 0) return;
        }
        available_ = true;
    }

    // Read one fd: {value, time_enabled, time_running}
    static uint64_t read_scaled(int fd) noexcept {
        struct { uint64_t value; uint64_t time_enabled; uint64_t time_running; } d{};
        if (::read(fd, &d, sizeof(d)) <= 0) return 0;
        if (d.time_running == 0) return 0;
        return static_cast<uint64_t>(
            static_cast<double>(d.value) *
            static_cast<double>(d.time_enabled) /
            static_cast<double>(d.time_running));
    }

    DeepDiveSnapshot read_all() noexcept {
        DeepDiveSnapshot s;
        s.instructions  = read_scaled(fds_[0]);
        s.cycles        = read_scaled(fds_[1]);
        s.branches      = read_scaled(fds_[2]);
        s.branch_misses = read_scaled(fds_[3]);
        s.l1d_accesses  = read_scaled(fds_[4]);
        s.l1d_misses    = read_scaled(fds_[5]);
        s.llc_accesses  = read_scaled(fds_[6]);
        s.llc_misses    = read_scaled(fds_[7]);
        s.available     = true;
        return s;
    }
#endif // __linux__
};

// Convenience: run fn for `passes` iterations with ExtendedPerfCounters,
// return a per-iteration snapshot.
template<typename Fn>
DeepDiveSnapshot measure_deep(int warmup, int passes, Fn fn) noexcept {
    ExtendedPerfCounters ctr;
    for (int i = 0; i < warmup; ++i) fn();
    ctr.start();
    for (int i = 0; i < passes; ++i) fn();
    auto snap = ctr.stop();

    if (snap.available && passes > 0) {
        auto n = static_cast<uint64_t>(passes);
        snap.instructions  /= n;
        snap.cycles        /= n;
        snap.branches      /= n;
        snap.branch_misses /= n;
        snap.l1d_accesses  /= n;
        snap.l1d_misses    /= n;
        snap.llc_accesses  /= n;
        snap.llc_misses    /= n;
    }
    return snap;
}

} // namespace cpu_engine::perf_deep_dive
