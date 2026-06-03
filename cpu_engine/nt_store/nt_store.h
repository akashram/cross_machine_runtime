#pragma once

// Non-Temporal Store Primitives
// =========================================================================
//
// PROBLEM: THE READ-FOR-OWNERSHIP PENALTY
// ----------------------------------------
// Every regular (temporal) store on x86 goes through the cache hierarchy.
// Before a CPU can write a cache line it doesn't own, it must first load
// that line into its cache (a "Read For Ownership" or RFO). This happens
// even when you know you will never read the data back — for example, when
// zeroing a tensor buffer before use, or staging output into a DMA region.
//
// For a 512 MB buffer that exceeds L3:
//   Regular store:  RFO (read 64 B from DRAM) + modify + evict = ~2× DRAM traffic
//   NT store:       write 64 B directly to DRAM via write-combining buffer = 1× traffic
//
// NT stores can therefore achieve ~2× write bandwidth for large write-only buffers.
//
//
// HOW NON-TEMPORAL STORES WORK
// -----------------------------
// _mm_stream_* instructions write to the CPU's Write-Combining (WC) buffers,
// bypassing L1/L2/L3 entirely:
//
//   WC buffers: x86 has 12 (Skylake) to 16 (Ice Lake) 64-byte WC line fill
//   buffers. When a WC buffer fills (after writing all 64 bytes of a cache
//   line), it is flushed to DRAM as a full-line burst — maximally efficient
//   for the memory controller. If you write a partial line and then move on,
//   the WC buffer eventually drains as a partial write (less efficient).
//   For maximum bandwidth, use stores that complete full 64-byte lines.
//
//
// SFENCE IS MANDATORY
// --------------------
// NT stores are weakly ordered — they may be globally visible after stores
// that appear later in program order, and other threads may see them in a
// different order than they were issued. SFENCE (store fence) ensures all
// preceding NT stores are globally visible before any subsequent store.
//
// Call sfence() after any sequence of NT stores before:
//   - Signalling another thread that the written data is ready
//   - Reading from the same addresses yourself (though this will be slow —
//     you'll take a cache miss since NT stores bypassed the cache)
//   - Using the buffer with DMA hardware
//
//
// ALIGNMENT REQUIREMENTS
// -----------------------
//   _mm_stream_si128  (16-byte NT store): dst must be 16-byte aligned
//   _mm256_stream_si256 (32-byte NT store): dst must be 32-byte aligned
//
// nt_memset / nt_memcpy handle unaligned heads with scalar stores so the
// caller does not need to worry about alignment. For raw nt_store<T>() calls
// in tight loops, alignment is the caller's responsibility.
//
//
// WHEN TO USE NT STORES — DECISION GUIDE
// ----------------------------------------
// USE when ALL of the following are true:
//   1. Write-only: you will not read this memory back within the next
//      ~10 ms (long enough for the cache to have naturally evicted it).
//   2. Large: the write set is significantly larger than L3 cache.
//      Rule of thumb: > 2× L3 size. Below that, regular stores may be
//      faster because the RFO hits L3 instead of DRAM.
//   3. Sequential: you're writing a contiguous region at near-line-size
//      stride. Scattered NT stores don't fill WC buffers efficiently.
//
// DO NOT USE when:
//   - Data will be read back soon (tensor forward pass reads its own output)
//   - Buffer is small (fits in L2/L3) — RFO is cheap, cache warmth is valuable
//   - Access pattern is random — WC buffers evict before filling
//
// PRACTICAL EXAMPLES
//   ✓ Zero-initialise a 1 GB tensor buffer before a batch run
//   ✓ Copy activations to a DMA staging buffer for GPU transfer
//   ✓ Fill a ring-buffer's storage region at allocation time
//   ✗ Write output of a matmul that will be consumed immediately by the next op
//   ✗ Update a 64 KB weight buffer that is hot in L2
//
//
// macOS / non-x86
// ---------------
// Intrinsics are x86-only. On non-x86 (e.g., ARM) or when the required ISA
// extension is absent, all functions fall back to regular memset/memcpy.
// The fallback is correct; it is only slower for the large-write-only case.
//
// =========================================================================

#include <cstddef>
#include <cstdint>
#include <cstring>

#if defined(__SSE2__)
#  include <emmintrin.h>   // SSE2: _mm_stream_si128, _mm_sfence
#endif
#if defined(__AVX__)
#  include <immintrin.h>   // AVX/AVX2: _mm256_stream_si256
#endif

namespace cpu_engine {

// =========================================================================
// sfence — store fence; call after any sequence of NT stores
// =========================================================================
inline void sfence() noexcept {
#if defined(__SSE2__)
    _mm_sfence();
#else
    // On non-x86 with weaker memory models, an appropriate barrier would be
    // needed here. ARM: __dmb(ish). For now the fallback is a compiler fence.
    __asm__ __volatile__("" ::: "memory");
#endif
}

// =========================================================================
// nt_store<T> — raw non-temporal store, no implicit sfence
//
// Use in tight loops; call sfence() once after the loop completes.
// T must be one of: int32_t, int64_t, __m128i (SSE2), __m256i (AVX2).
// =========================================================================
template<typename T>
inline void nt_store(T* dst, T val) noexcept = delete;

#if defined(__SSE2__)
template<>
inline void nt_store<int32_t>(int32_t* dst, int32_t val) noexcept {
    _mm_stream_si32(dst, val);
}

template<>
inline void nt_store<int64_t>(int64_t* dst, int64_t val) noexcept {
    _mm_stream_si64(reinterpret_cast<long long*>(dst),
                    static_cast<long long>(val));
}

template<>
inline void nt_store<__m128i>(__m128i* dst, __m128i val) noexcept {
    _mm_stream_si128(dst, val);  // dst must be 16-byte aligned
}

#endif // __SSE2__

#if defined(__AVX__)
template<>
inline void nt_store<__m256i>(__m256i* dst, __m256i val) noexcept {
    _mm256_stream_si256(dst, val);  // dst must be 32-byte aligned
}

template<>
inline void nt_store<__m256>(__m256* dst, __m256 val) noexcept {
    _mm256_stream_ps(reinterpret_cast<float*>(dst), val);
}

template<>
inline void nt_store<__m256d>(__m256d* dst, __m256d val) noexcept {
    _mm256_stream_pd(reinterpret_cast<double*>(dst), val);
}
#endif // __AVX__

// =========================================================================
// nt_memset — fill `size` bytes at `dst` with `fill_byte` using NT stores
//
// Handles arbitrary alignment and size. Includes sfence() at the end.
// Faster than ::memset for large (> 2× L3) write-only buffers.
// =========================================================================
inline void nt_memset(void* dst, uint8_t fill_byte, std::size_t size) noexcept {
    if (size == 0) return;

#if defined(__AVX__)
    // Broadcast fill_byte to a 256-bit register.
    const __m256i ymm_fill = _mm256_set1_epi8(static_cast<char>(fill_byte));

    auto* p   = static_cast<uint8_t*>(dst);
    auto* end = p + size;

    // Scalar head: advance p to 32-byte alignment.
    while (p < end && (reinterpret_cast<uintptr_t>(p) & 31u) != 0)
        *p++ = fill_byte;

    // AVX2 main loop: 128 bytes (4 × 32) per iteration fills WC buffers
    // efficiently — 2 WC buffers per iteration on Skylake (64 bytes each).
    auto* p256 = reinterpret_cast<__m256i*>(p);
    auto* end256 = reinterpret_cast<__m256i*>(end - ((end - p) & 127));
    while (p256 < end256) {
        _mm256_stream_si256(p256,     ymm_fill);
        _mm256_stream_si256(p256 + 1, ymm_fill);
        _mm256_stream_si256(p256 + 2, ymm_fill);
        _mm256_stream_si256(p256 + 3, ymm_fill);
        p256 += 4;
    }
    p = reinterpret_cast<uint8_t*>(p256);

    // Remaining 32-byte blocks
    auto* end32 = reinterpret_cast<__m256i*>(end - ((end - p) & 31));
    auto* p32   = reinterpret_cast<__m256i*>(p);
    while (p32 < end32) {
        _mm256_stream_si256(p32, ymm_fill);
        ++p32;
    }
    p = reinterpret_cast<uint8_t*>(p32);

    // Scalar tail
    while (p < end) *p++ = fill_byte;

#elif defined(__SSE2__)
    // Broadcast fill_byte to a 128-bit register.
    const __m128i xmm_fill = _mm_set1_epi8(static_cast<char>(fill_byte));

    auto* p   = static_cast<uint8_t*>(dst);
    auto* end = p + size;

    // Scalar head to 16-byte alignment
    while (p < end && (reinterpret_cast<uintptr_t>(p) & 15u) != 0)
        *p++ = fill_byte;

    // SSE2 main loop: 64 bytes per iteration
    auto* p128  = reinterpret_cast<__m128i*>(p);
    auto* end128= reinterpret_cast<__m128i*>(end - ((end - p) & 63));
    while (p128 < end128) {
        _mm_stream_si128(p128,     xmm_fill);
        _mm_stream_si128(p128 + 1, xmm_fill);
        _mm_stream_si128(p128 + 2, xmm_fill);
        _mm_stream_si128(p128 + 3, xmm_fill);
        p128 += 4;
    }
    p = reinterpret_cast<uint8_t*>(p128);

    // Remaining 16-byte blocks
    auto* end16 = reinterpret_cast<__m128i*>(end - ((end - p) & 15));
    auto* p16   = reinterpret_cast<__m128i*>(p);
    while (p16 < end16) {
        _mm_stream_si128(p16, xmm_fill);
        ++p16;
    }
    p = reinterpret_cast<uint8_t*>(p16);

    // Scalar tail
    while (p < end) *p++ = fill_byte;

#else
    // Non-x86 fallback: regular memset (correct, not faster)
    std::memset(dst, static_cast<int>(fill_byte), size);
    return;  // no sfence needed for regular stores
#endif

    sfence();
}

// =========================================================================
// nt_memcpy — copy `size` bytes from `src` to `dst` using NT stores
//
// Read side: regular loads (hardware prefetcher handles sequential reads).
// Write side: NT stores (dst bypasses cache, no RFO penalty).
// Use when dst is a write-only staging buffer and src is hot in cache.
// Includes sfence() at end.
// =========================================================================
inline void nt_memcpy(void* dst, const void* src, std::size_t size) noexcept {
    if (size == 0) return;

#if defined(__AVX__)
    auto* d   = static_cast<uint8_t*>(dst);
    auto* s   = static_cast<const uint8_t*>(src);
    auto* end = d + size;

    // Scalar head: align dst to 32 bytes (src alignment may differ; that's OK,
    // unaligned loads are fast on modern x86).
    while (d < end && (reinterpret_cast<uintptr_t>(d) & 31u) != 0)
        *d++ = *s++;

    // AVX2 loop: 128 bytes per iteration
    auto* d256   = reinterpret_cast<__m256i*>(d);
    auto* s256   = reinterpret_cast<const __m256i*>(s);
    auto* end256 = reinterpret_cast<__m256i*>(end - ((end - d) & 127));
    while (d256 < end256) {
        // loadu: unaligned load (fine on modern x86, same speed as aligned for AVX)
        _mm256_stream_si256(d256,     _mm256_loadu_si256(s256));
        _mm256_stream_si256(d256 + 1, _mm256_loadu_si256(s256 + 1));
        _mm256_stream_si256(d256 + 2, _mm256_loadu_si256(s256 + 2));
        _mm256_stream_si256(d256 + 3, _mm256_loadu_si256(s256 + 3));
        d256 += 4;
        s256 += 4;
    }
    d = reinterpret_cast<uint8_t*>(d256);
    s = reinterpret_cast<const uint8_t*>(s256);

    // Remaining 32-byte blocks
    auto* d32 = reinterpret_cast<__m256i*>(d);
    auto* s32 = reinterpret_cast<const __m256i*>(s);
    auto* end32 = reinterpret_cast<__m256i*>(end - ((end - d) & 31));
    while (d32 < end32) {
        _mm256_stream_si256(d32, _mm256_loadu_si256(s32));
        ++d32; ++s32;
    }
    d = reinterpret_cast<uint8_t*>(d32);
    s = reinterpret_cast<const uint8_t*>(s32);

    // Scalar tail
    while (d < end) *d++ = *s++;

#elif defined(__SSE2__)
    auto* d   = static_cast<uint8_t*>(dst);
    auto* s   = static_cast<const uint8_t*>(src);
    auto* end = d + size;

    while (d < end && (reinterpret_cast<uintptr_t>(d) & 15u) != 0)
        *d++ = *s++;

    auto* d128   = reinterpret_cast<__m128i*>(d);
    auto* s128   = reinterpret_cast<const __m128i*>(s);
    auto* end128 = reinterpret_cast<__m128i*>(end - ((end - d) & 63));
    while (d128 < end128) {
        _mm_stream_si128(d128,     _mm_loadu_si128(s128));
        _mm_stream_si128(d128 + 1, _mm_loadu_si128(s128 + 1));
        _mm_stream_si128(d128 + 2, _mm_loadu_si128(s128 + 2));
        _mm_stream_si128(d128 + 3, _mm_loadu_si128(s128 + 3));
        d128 += 4; s128 += 4;
    }
    d = reinterpret_cast<uint8_t*>(d128);
    s = reinterpret_cast<const uint8_t*>(s128);

    auto* d16  = reinterpret_cast<__m128i*>(d);
    auto* s16  = reinterpret_cast<const __m128i*>(s);
    auto* end16= reinterpret_cast<__m128i*>(end - ((end - d) & 15));
    while (d16 < end16) {
        _mm_stream_si128(d16, _mm_loadu_si128(s16));
        ++d16; ++s16;
    }
    d = reinterpret_cast<uint8_t*>(d16);
    s = reinterpret_cast<const uint8_t*>(s16);

    while (d < end) *d++ = *s++;

#else
    std::memcpy(dst, src, size);
    return;
#endif

    sfence();
}

} // namespace cpu_engine
