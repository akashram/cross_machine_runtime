#pragma once

// Explicit Hugepage Memory Region
// =========================================================================
//
// WHY HUGEPAGES MATTER FOR LATENCY
// ---------------------------------
// Every memory access that misses all caches must be translated from virtual
// to physical address by the CPU's TLB (Translation Lookaside Buffer).
// A TLB miss requires a hardware page-table walk: typically 4 memory
// accesses (PGD → PUD → PMD → PTE on x86-64), adding ~10–100 ns of latency
// per unique page touched.
//
// With 4 KB pages, a 256 MB working set spans 65,536 unique pages.
// A typical L1 dTLB has 64 entries. Accessing even a fraction of that
// working set in random order causes constant TLB evictions ("thrashing").
//
// With 2 MB huge pages, the same 256 MB working set spans only 128 pages.
// A typical STLB (L2 TLB) for large pages has 32–64 entries — the entire
// working set fits. TLB misses drop by ~512x.
//
//
// LINUX MECHANICS
// ---------------
// MAP_HUGETLB (Linux 2.6.16):
//   Requests pages from the kernel's pre-allocated hugepage pool.
//   Pool size: /proc/sys/vm/nr_hugepages (set to 0 at boot by default).
//   If the pool is empty, mmap(MAP_HUGETLB) returns ENOMEM.
//   We fall back to 4 KB pages silently; call is_huge() to check.
//
// Reservation requirement:
//   The kernel pool must be pre-allocated before HugeRegion::alloc() is
//   called. To reserve N huge pages (N × 2 MB):
//     echo N | sudo tee /proc/sys/vm/nr_hugepages
//   For 256 MB: echo 128 | sudo tee /proc/sys/vm/nr_hugepages
//
// First-touch policy:
//   mmap only reserves virtual address space. Physical pages are assigned
//   the first time each page is written (the "page fault"). This means
//   benchmarks can see fault overhead on the first pass.
//   Call prefault() to trigger all faults before benchmarking.
//
// mbind(2) + NUMA binding:
//   After mmap, we can bind the VA range to a specific NUMA node with
//   mbind(). Page faults on that range are then served from that node's
//   DRAM. Combined with ThreadPinner, this eliminates remote-NUMA overhead
//   on the hot path.
//
//
// macOS REALITY
// -------------
// macOS has no user-space API for explicit huge pages. The VM system may
// promote 4 KB pages to 2 MB pages transparently ("transparent huge pages"),
// but this cannot be controlled or guaranteed from user code.
// try_huge=true is silently ignored on macOS; is_huge() always returns false.
//
//
// MEASURING TLB MISS REDUCTION
// ----------------------------
// On Linux, after running the hugepage benchmark:
//
//   perf stat -e dTLB-load-misses,dTLB-store-misses \
//             ./hugepage_bench
//
// Expected with 4 KB pages on a 256 MB scan:
//   dTLB-load-misses: ~65,000 per pass (one per page)
//
// Expected with 2 MB pages on a 256 MB scan:
//   dTLB-load-misses: ~128 per pass (one per huge page)
//
// The wall-clock speedup is typically 10–40% for bandwidth-bound workloads.
//
// =========================================================================

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <sys/mman.h>

#ifdef __linux__
#  include <sys/syscall.h>
#  include <unistd.h>
#endif

namespace cpu_engine {

// =========================================================================
// HugeRegion — RAII wrapper for a huge-page-backed memory region
// =========================================================================
class HugeRegion {
public:
    static constexpr std::size_t kHugePageSize  = 2u << 20;   // 2 MiB
    static constexpr std::size_t kSmallPageSize = 4u << 10;   // 4 KiB

    // Allocate a region of at least `size` bytes.
    //
    // try_huge=true (Linux only):
    //   Rounds size up to a 2 MB boundary and calls mmap(MAP_HUGETLB).
    //   Falls back silently to 4 KB if hugepage pool is empty.
    //
    // numa_node >= 0 (Linux only):
    //   Binds the physical pages to that NUMA node via mbind().
    //   Call before prefault() so page faults go to the correct node.
    static HugeRegion alloc(std::size_t size,
                            bool        try_huge  = true,
                            int         numa_node = -1) noexcept
    {
        if (size == 0) return {};

        HugeRegion r;

#ifdef __linux__
        if (try_huge) {
            // Round up to 2 MB multiple — required by MAP_HUGETLB.
            std::size_t aligned = (size + kHugePageSize - 1) & ~(kHugePageSize - 1);
            void* p = ::mmap(nullptr, aligned,
                             PROT_READ | PROT_WRITE,
                             MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB,
                             -1, 0);
            if (p != MAP_FAILED) {
                r.ptr_  = p;
                r.size_ = aligned;
                r.huge_ = true;
                numa_bind(p, aligned, numa_node);
                return r;
            }
            // Hugepage pool empty or CONFIG_HUGETLB_PAGE not set — fallthrough.
        }
#endif

        void* p = ::mmap(nullptr, size,
                         PROT_READ | PROT_WRITE,
                         MAP_PRIVATE | MAP_ANONYMOUS,
                         -1, 0);
        if (p == MAP_FAILED) return {};

        r.ptr_  = p;
        r.size_ = size;
        r.huge_ = false;

#ifdef __linux__
        numa_bind(p, size, numa_node);
#else
        (void)try_huge;
        (void)numa_node;
#endif
        return r;
    }

    ~HugeRegion() noexcept {
        if (ptr_) ::munmap(ptr_, size_);
    }

    HugeRegion(HugeRegion&& o) noexcept
        : ptr_(o.ptr_), size_(o.size_), huge_(o.huge_)
    {
        o.ptr_  = nullptr;
        o.size_ = 0;
        o.huge_ = false;
    }

    HugeRegion& operator=(HugeRegion&& o) noexcept {
        if (this != &o) {
            if (ptr_) ::munmap(ptr_, size_);
            ptr_  = o.ptr_;
            size_ = o.size_;
            huge_ = o.huge_;
            o.ptr_  = nullptr;
            o.size_ = 0;
            o.huge_ = false;
        }
        return *this;
    }

    HugeRegion(const HugeRegion&)            = delete;
    HugeRegion& operator=(const HugeRegion&) = delete;

    void*       data()    const noexcept { return ptr_; }
    std::size_t size()    const noexcept { return size_; }
    bool        is_huge() const noexcept { return huge_; }
    bool        ok()      const noexcept { return ptr_ != nullptr; }

    // Number of TLB entries required to cover this region (no eviction assumed).
    // This is the theoretical minimum; actual TLB misses depend on access pattern
    // and TLB capacity. The key ratio: 4 KB gives 512x more entries than 2 MB.
    std::size_t expected_tlb_entries() const noexcept {
        std::size_t pgsz = huge_ ? kHugePageSize : kSmallPageSize;
        return (size_ + pgsz - 1) / pgsz;
    }

    // Write a zero byte to every page to trigger physical page faults now,
    // before benchmarking. Without this, the first benchmark pass includes
    // fault handling overhead which skews timing.
    void prefault() noexcept {
        if (!ptr_) return;
        std::size_t pgsz = huge_ ? kHugePageSize : kSmallPageSize;
        auto* b = static_cast<char*>(ptr_);
        for (std::size_t off = 0; off < size_; off += pgsz)
            b[off] = 0;
    }

private:
    HugeRegion() noexcept = default;

#ifdef __linux__
    // Bind the VA range to a NUMA node via mbind(MPOL_BIND).
    // Called after mmap, before any page fault, so MPOL_MF_MOVE is unnecessary.
    static void numa_bind(void* ptr, std::size_t size, int node) noexcept {
        if (node < 0 || node >= 64) return;
        static constexpr int kMpolBind = 2;
        unsigned long mask = 1UL << static_cast<unsigned>(node);
        ::syscall(SYS_mbind, ptr, size, kMpolBind, &mask, 64UL, 0UL);
        // Ignore errors: region still works, just without NUMA binding.
    }
#endif

    void*       ptr_  = nullptr;
    std::size_t size_ = 0;
    bool        huge_ = false;
};

} // namespace cpu_engine
