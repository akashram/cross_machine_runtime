#pragma once

// Software Prefetch Primitives
// =========================================================================
//
// WHEN THE HARDWARE PREFETCHER IS NOT ENOUGH
// ------------------------------------------
// Modern CPUs have an automatic hardware prefetcher that detects sequential
// and simple stride access patterns and issues prefetches speculatively.
// It works well for:
//   - Sequential array scans (stride = cache line = 64 bytes)
//   - Regular strides up to ~512 bytes (on Skylake)
//
// It fails for:
//   - Pointer chasing (each address depends on the previous load)
//   - Random access (no detectable stride)
//   - Large irregular strides (> 512 bytes)
//   - Gathers / sparse tensor reads
//
// For these patterns, software prefetch issues an explicit memory request
// N iterations before the data is needed, hiding DRAM latency (80-100 ns)
// behind useful computation.
//
//
// PREFETCH HINT LEVELS (x86 PREFETCHT0/T1/T2/NTA)
// -------------------------------------------------
//   T0  (locality=3) → PREFETCHT0 → all levels (L1 + L2 + L3)
//     Use when: data will be accessed multiple times in the near future.
//     Trade-off: occupies L1 lines; can evict other hot data.
//
//   T1  (locality=2) → PREFETCHT1 → L2 + L3 (not L1)
//     Use when: data accessed soon but not immediately; too large for L1.
//
//   T2  (locality=1) → PREFETCHT2 → L3 only
//     Use when: prefetching far ahead; L1/L2 fill too early would cause
//     eviction before use.
//
//   NTA (locality=0) → PREFETCHNTA → one way of L1, no L2/L3 fill
//     Non-temporal read: doesn't pollute L2/L3. Use when data is read
//     exactly once — analogous to NT stores on the read side.
//
//
// CHOOSING PREFETCH DISTANCE
// --------------------------
//   optimal_distance ≈ memory_latency_ns / time_per_iteration_ns
//
// For this machine (L3=4MB, DRAM latency ≈ 80 ns):
//   Tight loop, 2 ns/iter   → distance ≈ 40
//   Moderate work, 10 ns/iter → distance ≈ 8
//   Heavy work, 50 ns/iter  → distance ≈ 2
//
// Pitfalls:
//   Too small → prefetch arrives late → still a cache miss.
//   Too large → prefetched line evicted before use.
//   Sequential data → hardware prefetcher already covers this; software
//     prefetch adds front-end pressure with no benefit.
//
//
// PORTABILITY
// -----------
// __builtin_prefetch: GCC/Clang on all architectures.
//   x86-64: PREFETCHT0/T1/T2/NTA instructions.
//   ARM64:  PRFM PLDL1KEEP (T0), PLDL2KEEP (T1), PLDL3KEEP (T2),
//           PLDL1STRM (NTA).
//   Others: may compile to a no-op; correctness unaffected.
// =========================================================================

#include <cstddef>
#include <cstdint>
#include <vector>

namespace cpu_engine {

// =========================================================================
// PrefetchHint — maps to __builtin_prefetch locality parameter
// =========================================================================
enum class PrefetchHint : int {
    T0  = 3,  // all cache levels (L1+L2+L3) — highest locality
    T1  = 2,  // L2 + L3
    T2  = 1,  // L3 only
    NTA = 0,  // non-temporal: one way of L1, no L2/L3 fill
};

// =========================================================================
// prefetch_r<H>(addr) — issue a read prefetch with hint H
// prefetch_w<H>(addr) — issue a write prefetch with hint H
//
// Prefetching is a pure hint: correctness is unaffected; only performance
// changes. These inline to a single PREFETCH* instruction.
// =========================================================================
template<PrefetchHint H = PrefetchHint::T0>
inline void prefetch_r(const void* addr) noexcept {
    __builtin_prefetch(addr, /*rw=*/0, static_cast<int>(H));
}

template<PrefetchHint H = PrefetchHint::T0>
inline void prefetch_w(void* addr) noexcept {
    __builtin_prefetch(addr, /*rw=*/1, static_cast<int>(H));
}

// =========================================================================
// prefetch_ahead<T, H>(base, i, dist)
//
// Prefetch base[i + dist] for a read with hint H.
// The canonical loop pattern:
//
//   for (std::size_t i = 0; i < N; ++i) {
//       prefetch_ahead(arr, i, DIST);   // request arr[i+DIST] from memory
//       result += process(arr[i]);      // use arr[i] (expected in cache)
//   }
//
// H=T0 (default): suitable when data may be accessed multiple times.
// H=NTA: use for one-pass streaming reads to avoid polluting L2/L3.
// =========================================================================
template<typename T, PrefetchHint H = PrefetchHint::T0>
inline void prefetch_ahead(const T* base, std::size_t i, std::size_t dist) noexcept {
    __builtin_prefetch(base + i + dist, /*rw=*/0, static_cast<int>(H));
}

template<typename T, PrefetchHint H = PrefetchHint::T0>
inline void prefetch_ahead_w(T* base, std::size_t i, std::size_t dist) noexcept {
    __builtin_prefetch(base + i + dist, /*rw=*/1, static_cast<int>(H));
}

// =========================================================================
// make_pointer_chase_list(buf, n, seed)
//
// Fills buf[0..n-1] with a random cyclic successor permutation:
//   buf[i] = "the next index to visit after visiting index i"
// Visiting every element: idx = 0; for (n iters) idx = buf[idx];
//
// This is the gold-standard benchmark for measuring the benefit of software
// prefetch when the hardware prefetcher is powerless (fully random access).
//
// The walk visits every element exactly once before returning to 0.
// =========================================================================
inline void make_pointer_chase_list(std::size_t* buf, std::size_t n,
                                    uint64_t seed = 0xDEADBEEFCAFEBABEull) noexcept {
    // Build a random permutation of {0..n-1} via Fisher-Yates.
    for (std::size_t i = 0; i < n; ++i) buf[i] = i;

    auto rng = [&]() -> uint64_t {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        return seed;
    };

    for (std::size_t i = n - 1; i > 0; --i) {
        std::size_t j = rng() % (i + 1);
        std::size_t tmp = buf[i];
        buf[i] = buf[j];
        buf[j] = tmp;
    }

    // buf is now a permutation P. Convert to a cyclic successor list:
    // next[P(i)] = P((i+1) % n)
    // so a walk starting at any element visits all n elements.
    std::vector<std::size_t> succ(n);
    for (std::size_t i = 0; i < n; ++i)
        succ[buf[i]] = buf[(i + 1) % n];
    for (std::size_t i = 0; i < n; ++i)
        buf[i] = succ[i];
}

} // namespace cpu_engine
