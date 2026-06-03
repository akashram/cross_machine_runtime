#include "hugepage/hugepage.h"
#include "affinity/affinity.h"
#include "foundation/bench/bench.h"
#include "foundation/perf/perf.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <numeric>
#include <vector>

using cpu_engine::HugeRegion;
using cpu_engine::ThreadPinner;
using foundation::PerfCounters;
using foundation::PerfSnapshot;

// ---------------------------------------------------------------------------
// Benchmark: sequential read scan — 4 KB pages vs 2 MB huge pages
//
// Workload: read every 64-byte cache line in a large buffer, accumulate sum.
// Metric: ns per cache-line access, memory bandwidth (GB/s), LLC miss rate.
//
// TLB pressure analysis (256 MB buffer):
//   4 KB pages : 65,536 unique pages → exceeds any dTLB (typically 64–4096)
//   2 MB pages :     128 unique pages → fits in large-page TLB (32–64 entries)
//
// On Linux, get exact dTLB miss counts with:
//   perf stat -e dTLB-load-misses,dTLB-store-misses ./hugepage_bench
//
// Expected dTLB miss rates on a 256 MB scan:
//   4 KB pages : ~65,000 misses per pass (~2.9% of cache-line accesses)
//   2 MB pages :    ~128 misses per pass (~0.006% of cache-line accesses)
//
// Wall-clock speedup typically 10–40% for bandwidth-bound workloads.
// ---------------------------------------------------------------------------

static constexpr std::size_t kRegionSize  = 256u << 20;  // 256 MiB
static constexpr std::size_t kCacheLine   = 64;
static constexpr int         kPasses      = 5;
static constexpr int         kWarmupPasses= 2;

struct BenchResult {
    double  ns_per_access;   // ns per 64-byte cache line read
    double  bandwidth_gbs;   // GB/s
    PerfSnapshot perf;
    bool    is_huge;
    std::size_t tlb_entries;
};

static BenchResult run_scan(HugeRegion& region) {
    auto* base   = static_cast<const volatile uint64_t*>(region.data());
    std::size_t n_words  = region.size() / sizeof(uint64_t);
    std::size_t n_lines  = region.size() / kCacheLine;
    std::size_t step     = kCacheLine / sizeof(uint64_t);  // 8 words per cache line

    // Warmup passes (not timed)
    uint64_t sink = 0;
    for (int p = 0; p < kWarmupPasses; ++p)
        for (std::size_t i = 0; i < n_words; i += step)
            sink += base[i];

    PerfCounters ctr;
    ctr.start();
    uint64_t t0 = bench::tsc_now();

    for (int p = 0; p < kPasses; ++p)
        for (std::size_t i = 0; i < n_words; i += step)
            sink += base[i];

    uint64_t t1 = bench::tsc_now();
    auto perf   = ctr.stop();

    // Prevent dead-code elimination
    if (sink == 0xDEAD) std::printf(" ");

    double total_ns      = bench::tsc_to_ns(t1 - t0);
    double total_accesses= static_cast<double>(n_lines) * kPasses;
    double ns_per_access = total_ns / total_accesses;
    double bytes_read    = static_cast<double>(region.size()) * kPasses;
    double bandwidth_gbs = bytes_read / (total_ns * 1e-9) / 1e9;

    return { ns_per_access, bandwidth_gbs, perf, region.is_huge(),
             region.expected_tlb_entries() };
}

static void print_result(const char* label, const BenchResult& r) {
    printf("  %-12s  huge=%-3s  tlb_entries=%6zu  "
           "%5.2f ns/line  %5.1f GB/s",
           label,
           r.is_huge ? "yes" : "no",
           r.tlb_entries,
           r.ns_per_access,
           r.bandwidth_gbs);

    if (r.perf.available) {
        printf("  IPC=%.2f  L3miss=%.1f%%",
               r.perf.ipc(),
               r.perf.cache_miss_rate() * 100.0);
    }
    printf("\n");
}

int main() {
    // Pin to CPU 0 for stable timing — eliminates migration jitter.
    ThreadPinner::pin(0);

    printf("Hugepage TLB benchmark\n");
    printf("Platform: ");
#ifdef __linux__
    printf("Linux (MAP_HUGETLB supported)\n");
    printf("NOTE: for dTLB counters run:\n");
    printf("  perf stat -e dTLB-load-misses,dTLB-store-misses ./hugepage_bench\n");
    printf("\nHugepage pool status: /proc/sys/vm/nr_hugepages\n");
    {
        FILE* f = std::fopen("/proc/sys/vm/nr_hugepages", "r");
        if (f) {
            int n = 0;
            (void)std::fscanf(f, "%d", &n);
            std::fclose(f);
            printf("  nr_hugepages = %d ", n);
            if (n == 0)
                printf("(no huge pages reserved — run: "
                       "echo 128 | sudo tee /proc/sys/vm/nr_hugepages)\n");
            else
                printf("(%.0f MiB available)\n",
                       static_cast<double>(n) * 2.0);
        }
    }
#else
    printf("macOS (huge pages not available; 4 KB only)\n");
#endif

    printf("\nRegion size: %zu MiB  Passes: %d  Cache-line stride: %zu bytes\n\n",
           kRegionSize >> 20, kPasses, kCacheLine);

    // Force TSC calibration before allocation (avoids skewing alloc timing).
    (void)bench::tsc_ticks_per_ns();

    // Allocate both regions up front so both are ready before benchmarking.
    auto small = HugeRegion::alloc(kRegionSize, /*try_huge=*/false);
    auto huge  = HugeRegion::alloc(kRegionSize, /*try_huge=*/true);

    if (!small.ok()) { printf("ERROR: 4KB alloc failed\n"); return 1; }
    if (!huge.ok())  { printf("ERROR: huge alloc failed\n");  return 1; }

    // Initialize with non-zero data (avoids OS zero-page optimizations).
    {
        auto* b = static_cast<uint64_t*>(small.data());
        auto* h = static_cast<uint64_t*>(huge.data());
        std::size_t n = kRegionSize / sizeof(uint64_t);
        for (std::size_t i = 0; i < n; ++i) b[i] = h[i] = i + 1;
    }

    // Prefault both regions — trigger all page faults before benchmarking.
    small.prefault();
    huge.prefault();

    printf("Results (lower ns/line = better; lower tlb_entries = less TLB pressure):\n");
    auto r_small = run_scan(small);
    print_result("4KB pages",  r_small);

    auto r_huge  = run_scan(huge);
    print_result("2MB pages",  r_huge);

    printf("\nTLB entry reduction: %.0fx  ",
           static_cast<double>(r_small.tlb_entries) /
           static_cast<double>(r_huge.tlb_entries > 0 ? r_huge.tlb_entries : 1));
    if (r_huge.is_huge) {
        double speedup = r_small.ns_per_access / r_huge.ns_per_access;
        printf("Wall-clock speedup: %.2fx\n", speedup);
    } else {
        printf("(huge pages not available — no speedup comparison)\n");
    }

#ifndef __linux__
    printf("\nmacOS: run on Linux with nr_hugepages > 0 for real numbers.\n");
#endif
}
