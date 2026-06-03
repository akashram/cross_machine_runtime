#pragma once

// Arena Allocator
// =========================================================================
//
// Three allocator layers, each building on the one below:
//
//   Arena              — bump allocator over a contiguous mmap'd slab.
//                        Alloc is O(1) (pointer increment + alignment).
//                        Free is O(1) only in aggregate: reset() rewinds
//                        the cursor to the start.  No per-object free.
//
//   SizeClassedArena   — recycling allocator on top of Arena.
//                        8 power-of-2 size classes (8..1024 bytes), each
//                        backed by an intrusive singly-linked freelist.
//                        alloc: pop from freelist or bump from arena.
//                        free:  push back onto the matching freelist.
//                        Large (>1024) objects bump directly, no recycling.
//
//   ThreadLocalArena   — static API backed by a thread_local
//                        SizeClassedArena.  The common path has zero
//                        atomic operations and zero lock contention.
//
//
// Why bump allocator?
// -------------------
// For workloads with a clear lifetime boundary (per-request scratch, per-
// frame temporaries, parser output), a bump allocator eliminates per-object
// metadata entirely.  Throughput is limited only by cache miss rate on the
// slab — typically 1–3 ns/alloc vs. 20–200 ns for malloc in contended cases.
//
// Why size classes?
// -----------------
// Pure bump allocators don't recycle individual objects.  Adding an
// intrusive freelist per size class gives us O(1) alloc AND O(1) free for
// hot-path objects (task nodes, coroutine frames, network buffers) while
// keeping the bump fast for everything else.  Power-of-2 classes mean:
//   - Size class lookup: one bit_ceil + ctz instruction.
//   - Alignment: an N-byte size class is N-byte aligned (power-of-2 base).
//   - Overhead: at most 2x wasted space (e.g. a 33-byte request gets 64).
//
// Why per-thread?
// ---------------
// Sharing an allocator across threads requires either a mutex or a lock-free
// freelist with ABA protection (see foundation/freelist/).  Both add latency.
// Per-thread arenas eliminate contention entirely: the fast path is a
// thread-local load + a pointer bump.  Cross-thread frees are intentionally
// unsupported — if you need them, use a lock-free freelist or malloc.
//
// Why mmap instead of new[]?
// --------------------------
//   (a) Hugepage support: MAP_HUGETLB (Linux) gives 2 MB pages, reducing
//       TLB pressure by 512x vs. 4 KB pages for the same working set.
//       One 2 MB page = 512 TLB entries worth of coverage; a 4 MB slab
//       needs only 2 TLB entries instead of 1024.
//   (b) MADV_DONTNEED / MADV_FREE_REUSABLE: after reset(), we can tell the
//       OS to reclaim the physical backing immediately.  The VA range stays
//       mapped; the physical pages are returned to the free pool and will be
//       zero-filled on next access.  This matters when a 4 MB arena is reset
//       after processing a single request — without madvise the OS holds the
//       pages until process exit.
//   (c) Heap isolation: the slab lives in its own VA range, not interleaved
//       with malloc's heap, so arena pointers don't fragment the system heap.
//
// Hugepage mechanics
// ------------------
// On Linux, MAP_HUGETLB allocates pages from the kernel's hugepage pool
// (configured via /proc/sys/vm/nr_hugepages).  If the pool is empty or
// the size isn't a 2 MB multiple, mmap returns ENOMEM.  Arena falls back
// to regular 4 KB pages silently; call is_hugepage() to check.
//
// On macOS, there is no MAP_HUGETLB equivalent.  The OS may promote pages
// to 2 MB internally (transparent huge pages), but there is no user-space
// API to request it.  try_hugepage=true is ignored on macOS.
//
// When to choose what
// -------------------
//   malloc              General purpose.  Unknown sizes, long or mixed
//                       lifetimes, needs cross-thread free.
//
//   Arena               Known aggregate lifetime, no per-object free
//                       needed.  Per-request scratch buffers, AST nodes,
//                       temporary deserialised records.
//
//   SizeClassedArena    Hot path, repeated alloc+free of small objects of
//                       predictable sizes.  Task nodes, I/O buffers, pool
//                       of fixed-size messages.
//
//   FreeList<T>         Single known type, pool-style, bounded latency
//                       required.  Typed, no size-class rounding waste.
//
// =========================================================================

#include <bit>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <algorithm>
#include <sys/mman.h>

// macOS exposes MADV_FREE_REUSABLE in <sys/mman.h>; define it here as a
// fallback for older SDKs or non-Apple platforms that don't have it.
#if defined(__APPLE__) && !defined(MADV_FREE_REUSABLE)
#define MADV_FREE_REUSABLE 7
#endif

namespace foundation {

// ---------------------------------------------------------------------------
// Arena — raw bump allocator
// ---------------------------------------------------------------------------
class Arena {
public:
    // Allocate a slab of `size` bytes via mmap.
    // If try_hugepage && Linux: attempt MAP_HUGETLB, fall back on failure.
    explicit Arena(std::size_t size, bool try_hugepage = false) noexcept
        : capacity_(size)
    {
        base_ = static_cast<char*>(map_slab(size, try_hugepage));
        cursor_ = base_;
        end_    = base_ ? base_ + capacity_ : nullptr;
    }

    ~Arena() noexcept {
        if (base_) ::munmap(base_, capacity_);
    }

    Arena(const Arena&)            = delete;
    Arena& operator=(const Arena&) = delete;

    // Allocate `size` bytes with `align`-byte alignment (must be power of 2).
    // Returns nullptr if the slab is full.
    void* alloc(std::size_t size, std::size_t align = alignof(std::max_align_t)) noexcept {
        assert(size > 0);
        assert((align & (align - 1)) == 0);  // power of 2

        auto p = reinterpret_cast<uintptr_t>(cursor_);
        p = (p + align - 1) & ~(align - 1);  // align up
        if (p + size > reinterpret_cast<uintptr_t>(end_)) return nullptr;
        cursor_ = reinterpret_cast<char*>(p + size);
        return reinterpret_cast<void*>(p);
    }

    // Reset cursor to start.  Optionally advise the OS to reclaim physical
    // pages (they are re-faulted as zero-filled on next access).
    void reset(bool release_pages = false) noexcept {
        cursor_ = base_;
        if (release_pages && base_) {
            std::size_t sz = static_cast<std::size_t>(end_ - base_);
#if defined(__linux__)
            ::madvise(base_, sz, MADV_DONTNEED);
#elif defined(__APPLE__)
            ::madvise(base_, sz, MADV_FREE_REUSABLE);
#endif
        }
    }

    bool        is_hugepage() const noexcept { return hugepage_; }
    char*       base()        const noexcept { return base_; }
    bool        owns(const void* p) const noexcept {
        auto a = reinterpret_cast<uintptr_t>(p);
        return a >= reinterpret_cast<uintptr_t>(base_) &&
               a <  reinterpret_cast<uintptr_t>(end_);
    }
    std::size_t used()     const noexcept { return static_cast<std::size_t>(cursor_ - base_); }
    std::size_t capacity() const noexcept { return capacity_; }
    bool        ok()       const noexcept { return base_ != nullptr; }

private:
    void* map_slab(std::size_t size, bool try_hugepage) noexcept {
#ifdef __linux__
        if (try_hugepage) {
            constexpr std::size_t kHuge = 2u << 20;  // 2 MiB
            std::size_t aligned = (size + kHuge - 1) & ~(kHuge - 1);
            void* p = ::mmap(nullptr, aligned,
                             PROT_READ | PROT_WRITE,
                             MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB,
                             -1, 0);
            if (p != MAP_FAILED) {
                hugepage_ = true;
                capacity_ = aligned;
                return p;
            }
            // fallthrough: no hugepages available
        }
#else
        (void)try_hugepage;
#endif
        void* p = ::mmap(nullptr, size,
                         PROT_READ | PROT_WRITE,
                         MAP_PRIVATE | MAP_ANONYMOUS,
                         -1, 0);
        return p != MAP_FAILED ? p : nullptr;
    }

    char*       base_{nullptr};
    char*       cursor_{nullptr};
    char*       end_{nullptr};
    std::size_t capacity_{0};
    bool        hugepage_{false};
};

// ---------------------------------------------------------------------------
// SizeClassedArena — per-size-class freelist recycling on top of Arena
// ---------------------------------------------------------------------------
//
// Size classes and their indices:
//   index  0   1   2   3    4    5    6     7
//   bytes  8  16  32  64  128  256  512  1024
//
// Freelist nodes are stored in-place in freed blocks.  A freed N-byte block
// stores one FreeNode* (8 bytes on 64-bit) at its start; the rest is unused.
// The minimum size class is 8 bytes = sizeof(FreeNode*), so this always fits.
//
// Objects larger than kMaxClass bypass the freelist and are bump-allocated.
// They are reclaimed only by reset().

class SizeClassedArena {
public:
    static constexpr std::size_t kNumClasses  = 8;
    static constexpr std::size_t kMaxClass    = 1024;
    static constexpr std::size_t kSizes[kNumClasses] = {8,16,32,64,128,256,512,1024};

    explicit SizeClassedArena(std::size_t slab_size,
                              bool try_hugepage = false) noexcept
        : arena_(slab_size, try_hugepage) {}

    void* alloc(std::size_t size) noexcept {
        if (size == 0) return nullptr;
        int idx = class_index(size);
        if (idx < 0) {
            // Large: bump only, no recycling
            return arena_.alloc(size, alignof(std::max_align_t));
        }
        std::size_t cls = kSizes[idx];
        if (free_[idx]) {
            auto* node = free_[idx];
            free_[idx] = node->next;
            return node;
        }
        // Aligned to size class — a power-of-2 class N is N-byte aligned.
        return arena_.alloc(cls, cls);
    }

    void free(void* ptr, std::size_t size) noexcept {
        if (!ptr || size == 0) return;
        int idx = class_index(size);
        if (idx < 0) return;  // large: reclaimed on reset()
        auto* node  = static_cast<FreeNode*>(ptr);
        node->next  = free_[idx];
        free_[idx]  = node;
    }

    void reset(bool release_pages = false) noexcept {
        for (auto& f : free_) f = nullptr;
        arena_.reset(release_pages);
    }

    Arena&       arena()       noexcept { return arena_; }
    const Arena& arena() const noexcept { return arena_; }

    // Returns the size class index for `n` in [0, kNumClasses), or -1 if large.
    static int class_index(std::size_t n) noexcept {
        if (n > kMaxClass) return -1;
        // Round up to the next power-of-2, floor at 8.
        std::size_t c = std::bit_ceil(std::max(n, std::size_t{8}));
        // 8 = 2^3 → index 0;  16 = 2^4 → index 1;  ...  1024 = 2^10 → index 7
        return static_cast<int>(__builtin_ctzl(c)) - 3;
    }

    // Returns the actual allocation size for a given request size.
    static std::size_t class_size(std::size_t n) noexcept {
        int idx = class_index(n);
        return idx >= 0 ? kSizes[idx] : n;
    }

private:
    struct FreeNode { FreeNode* next; };

    Arena    arena_;
    FreeNode* free_[kNumClasses]{};
};

// ---------------------------------------------------------------------------
// ThreadLocalArena — process-wide API, per-thread SizeClassedArena backing
// ---------------------------------------------------------------------------
//
// The default 4 MiB slab is sized to hold ~130k typical 32-byte objects
// before the arena needs to grow (which it doesn't — it's fixed size; the
// freelist handles recycling, and reset() is called at epoch boundaries).
//
// Cross-thread frees are NOT supported: free() must be called from the same
// thread that called alloc().  Violating this silently corrupts the freelist.

struct ThreadLocalArena {
    static constexpr std::size_t kSlabSize = 4u << 20;  // 4 MiB

    static void* alloc(std::size_t size) noexcept {
        return local().alloc(size);
    }

    static void free(void* ptr, std::size_t size) noexcept {
        local().free(ptr, size);
    }

    static void reset(bool release_pages = false) noexcept {
        local().reset(release_pages);
    }

    static SizeClassedArena& local() noexcept {
        thread_local SizeClassedArena arena{kSlabSize};
        return arena;
    }
};

} // namespace foundation
