#include "nt_store/nt_store.h"
#include "affinity/affinity.h"
#include "hugepage/hugepage.h"
#include "foundation/bench/bench.h"
#include "foundation/perf/perf.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <numeric>
#include <vector>

using cpu_engine::ThreadPinner;
using cpu_engine::HugeRegion;
using cpu_engine::nt_memset;
using cpu_engine::nt_memcpy;
using foundation::PerfCounters;

// ---------------------------------------------------------------------------
// Non-temporal store benchmark: regular vs NT writes on large buffers
//
// We measure three scenarios that illustrate when NT stores help and hurt:
//
//  Scenario A — Cold write (buffer not in cache):
//    Both regular and NT stores must go to DRAM. Regular stores do an RFO
//    (read the cache line first for ownership), then write. NT stores skip
//    the RFO. NT should be ~1.5-2× faster.
//
//  Scenario B — Warm write (buffer in L3 from a prior read):
//    Regular stores hit L3 (fast). NT stores bypass L3 (slow, go to DRAM).
//    Regular stores win — NT stores are wrong tool here.
//
//  Scenario C — NT memcpy (write-only dst staging buffer):
//    src read normally (hardware prefetcher helps), dst written with NT stores.
//    Avoids polluting cache with output data that won't be read back.
//
// The buffer is sized to guarantee it exceeds L3 on virtually any machine.
// On this Mac: L3 ≈ 8 MB. Buffer = 512 MB >> L3.
// ---------------------------------------------------------------------------

static constexpr std::size_t kBufSize  = 512u << 20;  // 512 MiB
static constexpr int         kPasses   = 3;
static constexpr int         kWarmup   = 1;

struct BenchResult {
    double gbs;         // GB/s
    double ipc;
    double llc_miss_pct;
};

static BenchResult measure_write(
    void* buf, std::size_t size,
    void (*write_fn)(void*, uint8_t, std::size_t),
    uint8_t fill, int passes)
{
    PerfCounters ctr;

    // Warmup
    for (int i = 0; i < kWarmup; ++i)
        write_fn(buf, fill, size);

    ctr.start();
    uint64_t t0 = bench::tsc_now();
    for (int i = 0; i < passes; ++i)
        write_fn(buf, fill, size);
    uint64_t t1 = bench::tsc_now();
    auto perf = ctr.stop();

    double ns  = bench::tsc_to_ns(t1 - t0);
    double gbs = (static_cast<double>(size) * passes) / (ns * 1e-9) / 1e9;

    return { gbs,
             perf.ipc(),
             perf.cache_miss_rate() * 100.0 };
}

static BenchResult measure_memcpy(
    void* dst, const void* src, std::size_t size,
    void (*copy_fn)(void*, const void*, std::size_t),
    int passes)
{
    PerfCounters ctr;

    for (int i = 0; i < kWarmup; ++i)
        copy_fn(dst, src, size);

    ctr.start();
    uint64_t t0 = bench::tsc_now();
    for (int i = 0; i < passes; ++i)
        copy_fn(dst, src, size);
    uint64_t t1 = bench::tsc_now();
    auto perf = ctr.stop();

    double ns  = bench::tsc_to_ns(t1 - t0);
    // Count bytes written only (dst side) for apples-to-apples with memset
    double gbs = (static_cast<double>(size) * passes) / (ns * 1e-9) / 1e9;

    return { gbs, perf.ipc(), perf.cache_miss_rate() * 100.0 };
}

static void print_result(const char* label, const BenchResult& r) {
    printf("  %-28s  %5.1f GB/s", label, r.gbs);
    if (r.ipc > 0.0)
        printf("  IPC=%.2f  L3miss=%.1f%%", r.ipc, r.llc_miss_pct);
    printf("\n");
}

// Wrappers matching the function pointer signatures
static void regular_memset(void* p, uint8_t v, std::size_t n) {
    std::memset(p, static_cast<int>(v), n);
}
static void nontemp_memset(void* p, uint8_t v, std::size_t n) {
    nt_memset(p, v, n);
}
static void regular_memcpy(void* d, const void* s, std::size_t n) {
    std::memcpy(d, s, n);
}
static void nontemp_memcpy(void* d, const void* s, std::size_t n) {
    nt_memcpy(d, s, n);
}

int main() {
    ThreadPinner::pin(0);
    (void)bench::tsc_ticks_per_ns();

    printf("Non-temporal store benchmark\n");
    printf("ISA: ");
#if defined(__AVX2__)
    printf("AVX2 (256-bit NT stores, 4×32B per loop)\n");
#elif defined(__SSE2__)
    printf("SSE2 (128-bit NT stores, 4×16B per loop)\n");
#else
    printf("scalar fallback (no NT stores — numbers will be identical)\n");
#endif
    printf("Buffer: %zu MiB  Passes: %d\n\n",
           kBufSize >> 20, kPasses);

    // Allocate with 4 KB pages (hugepages not needed; we're measuring DRAM bandwidth)
    auto buf_a = HugeRegion::alloc(kBufSize, /*try_huge=*/false);
    auto buf_b = HugeRegion::alloc(kBufSize, /*try_huge=*/false);
    if (!buf_a.ok() || !buf_b.ok()) {
        printf("ERROR: allocation failed (need %zu MiB free RAM)\n",
               2 * (kBufSize >> 20));
        return 1;
    }
    buf_a.prefault();
    buf_b.prefault();

    void* a = buf_a.data();
    void* b = buf_b.data();

    // -----------------------------------------------------------------------
    // Scenario A: cold write — both buffers were just prefaulted (cold cache)
    // -----------------------------------------------------------------------
    printf("Scenario A: cold write (buffer >> L3, both must hit DRAM)\n");
    printf("  Expectation: NT ~1.5-2x faster (skips RFO)\n");
    print_result("regular memset (cold)",   measure_write(a, kBufSize, regular_memset, 0x00, kPasses));
    print_result("NT memset (cold)",        measure_write(a, kBufSize, nontemp_memset, 0x00, kPasses));

    // -----------------------------------------------------------------------
    // Scenario B: warm write — warm the buffer in L3 with a read first,
    // then measure the write. On a 512 MB buffer, the buffer cannot actually
    // stay in L3 (L3 is typically 8-32 MB), so both go to DRAM. This shows
    // that for huge buffers NT stores are never worse than regular stores.
    // -----------------------------------------------------------------------
    printf("\nScenario B: warm-ish write (buffer read first to pull into L3 if possible)\n");
    printf("  Note: 512 MB >> L3, so both still go to DRAM on this machine\n");
    {
        // Touch the buffer to warm whatever fits in L3
        volatile uint8_t* p = static_cast<volatile uint8_t*>(a);
        uint64_t sink = 0;
        for (std::size_t i = 0; i < kBufSize; i += 64) sink += p[i];
        (void)sink;
    }
    print_result("regular memset (warm)",   measure_write(a, kBufSize, regular_memset, 0xFF, kPasses));
    print_result("NT memset (warm)",        measure_write(a, kBufSize, nontemp_memset, 0xFF, kPasses));

    // -----------------------------------------------------------------------
    // Scenario C: memcpy — NT write side, regular read side
    // -----------------------------------------------------------------------
    printf("\nScenario C: memcpy 512 MiB src→dst\n");
    printf("  NT variant: dst bypasses cache (staging buffer pattern)\n");
    print_result("regular memcpy",  measure_memcpy(b, a, kBufSize, regular_memcpy, kPasses));
    print_result("NT memcpy",       measure_memcpy(b, a, kBufSize, nontemp_memcpy, kPasses));

    printf("\nNote on ERMSB (Enhanced REP MOVSB/STOSB):\n");
    printf("  Intel Broadwell+ implements 'rep stosb' (used by libc memset) in\n");
    printf("  hardware-optimised microcode that often outperforms manual NT store loops.\n");
    printf("  If regular memset > NT memset above: ERMSB is winning. This is expected.\n");
    printf("\n");
    printf("Decision guide:\n");
    printf("  Use NT stores when:\n");
    printf("    - You need a specific non-zero fill AND want to avoid cache pollution\n");
    printf("    - Zeroing on a system/CPU without ERMSB (older x86, some ARM)\n");
    printf("    - Building a DMA staging buffer that another thread or device will consume\n");
    printf("    - sfence-based ordering is required (e.g. lock-free producer signalling)\n");
    printf("  Prefer memset/memcpy when:\n");
    printf("    - Pure zero-fill on modern Intel (ERMSB wins)\n");
    printf("    - Data read back soon (NT bypass causes cache misses on read)\n");
    printf("\n");
    printf("On Linux, measure cache behaviour with:\n");
    printf("  perf stat -e cache-misses,cache-references,LLC-load-misses,\\\n");
    printf("             dTLB-load-misses,iTLB-load-misses ./nt_store_bench\n");

    ThreadPinner::unpin();
}
