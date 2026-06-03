#include "nt_store/nt_store.h"

#include <cassert>
#include <cstdio>
#include <cstring>
#include <vector>

using cpu_engine::sfence;
using cpu_engine::nt_memset;
using cpu_engine::nt_memcpy;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

// Allocate a buffer aligned to `align` bytes.
// We need aligned storage because AVX2 stream stores require 32-byte alignment.
static std::vector<uint8_t> make_aligned(std::size_t size, std::size_t align) {
    // Over-allocate so we can find an aligned pointer within the allocation.
    std::vector<uint8_t> buf(size + align);
    return buf;
}

static uint8_t* align_ptr(std::vector<uint8_t>& buf, std::size_t align) {
    auto addr = reinterpret_cast<uintptr_t>(buf.data());
    uintptr_t off = (align - (addr & (align - 1))) & (align - 1);
    return buf.data() + off;
}

// ---------------------------------------------------------------------------
// Test 1: sfence() does not crash
// ---------------------------------------------------------------------------
static void test_sfence() {
    sfence();
    sfence();
    printf("PASS  test_sfence\n");
}

// ---------------------------------------------------------------------------
// Test 2: nt_memset fills with zero
// ---------------------------------------------------------------------------
static void test_nt_memset_zero() {
    constexpr std::size_t kSize = 256;
    auto raw = make_aligned(kSize, 32);
    uint8_t* p = align_ptr(raw, 32);

    // Pre-fill with non-zero
    std::memset(p, 0xFF, kSize);

    nt_memset(p, 0x00, kSize);

    for (std::size_t i = 0; i < kSize; ++i)
        assert(p[i] == 0x00);

    printf("PASS  test_nt_memset_zero  (%zu bytes)\n", kSize);
}

// ---------------------------------------------------------------------------
// Test 3: nt_memset with arbitrary fill byte
// ---------------------------------------------------------------------------
static void test_nt_memset_pattern() {
    constexpr std::size_t kSize  = 1024;
    constexpr uint8_t     kFill  = 0xAB;

    auto raw = make_aligned(kSize, 32);
    uint8_t* p = align_ptr(raw, 32);

    nt_memset(p, kFill, kSize);

    for (std::size_t i = 0; i < kSize; ++i)
        assert(p[i] == kFill);

    printf("PASS  test_nt_memset_pattern  (fill=0x%02X  size=%zu)\n", kFill, kSize);
}

// ---------------------------------------------------------------------------
// Test 4: nt_memset on unaligned pointer — scalar head must handle this
// ---------------------------------------------------------------------------
static void test_nt_memset_unaligned() {
    constexpr std::size_t kSize = 200;
    constexpr uint8_t     kFill = 0x55;

    std::vector<uint8_t> buf(kSize + 64, 0);

    // Deliberately use a pointer that is 1-byte off from 32-byte alignment.
    uint8_t* base = align_ptr(buf, 32);
    uint8_t* p    = base + 1;  // intentionally unaligned

    nt_memset(p, kFill, kSize);

    for (std::size_t i = 0; i < kSize; ++i)
        assert(p[i] == kFill);

    // Byte before and after must be untouched
    assert(base[0] == 0);
    assert(p[kSize] == 0);

    printf("PASS  test_nt_memset_unaligned  (offset=1  size=%zu)\n", kSize);
}

// ---------------------------------------------------------------------------
// Test 5: small sizes (< 16 bytes) — scalar-only path
// ---------------------------------------------------------------------------
static void test_nt_memset_small() {
    for (std::size_t sz : {std::size_t{0}, std::size_t{1}, std::size_t{3}, std::size_t{7}, std::size_t{15}}) {
        std::vector<uint8_t> buf(sz + 32, 0xFF);
        uint8_t* p = align_ptr(buf, 32);
        nt_memset(p, 0x42, sz);
        for (std::size_t i = 0; i < sz; ++i)
            assert(p[i] == 0x42);
    }
    printf("PASS  test_nt_memset_small  (sizes: 0,1,3,7,15)\n");
}

// ---------------------------------------------------------------------------
// Test 6: size boundaries around vector width (16, 32, 64, 128 bytes)
// ---------------------------------------------------------------------------
static void test_nt_memset_boundaries() {
    for (std::size_t sz : {std::size_t{16}, std::size_t{32}, std::size_t{48}, std::size_t{64}, std::size_t{96}, std::size_t{128}, std::size_t{256}}) {
        auto raw = make_aligned(sz + 32, 32);
        uint8_t* p = align_ptr(raw, 32);
        std::memset(p, 0xFF, sz);
        nt_memset(p, 0x00, sz);
        for (std::size_t i = 0; i < sz; ++i)
            assert(p[i] == 0x00);
    }
    printf("PASS  test_nt_memset_boundaries  (16,32,48,64,96,128,256 bytes)\n");
}

// ---------------------------------------------------------------------------
// Test 7: nt_memcpy copies correctly
// ---------------------------------------------------------------------------
static void test_nt_memcpy_basic() {
    constexpr std::size_t kSize = 512;

    auto src_raw = make_aligned(kSize, 32);
    auto dst_raw = make_aligned(kSize, 32);
    uint8_t* src = align_ptr(src_raw, 32);
    uint8_t* dst = align_ptr(dst_raw, 32);

    for (std::size_t i = 0; i < kSize; ++i)
        src[i] = static_cast<uint8_t>(i ^ 0xA5u);
    std::memset(dst, 0, kSize);

    nt_memcpy(dst, src, kSize);

    assert(std::memcmp(src, dst, kSize) == 0);
    printf("PASS  test_nt_memcpy_basic  (%zu bytes)\n", kSize);
}

// ---------------------------------------------------------------------------
// Test 8: nt_memcpy with unaligned dst
// ---------------------------------------------------------------------------
static void test_nt_memcpy_unaligned() {
    constexpr std::size_t kSize = 300;

    std::vector<uint8_t> src_buf(kSize + 64, 0);
    std::vector<uint8_t> dst_buf(kSize + 64, 0);

    uint8_t* src = align_ptr(src_buf, 32);
    uint8_t* dst = align_ptr(dst_buf, 32) + 3;  // 3 bytes off alignment

    for (std::size_t i = 0; i < kSize; ++i)
        src[i] = static_cast<uint8_t>(i);

    nt_memcpy(dst, src, kSize);

    assert(std::memcmp(src, dst, kSize) == 0);
    printf("PASS  test_nt_memcpy_unaligned  (%zu bytes, dst offset=3)\n", kSize);
}

// ---------------------------------------------------------------------------
// Test 9: large nt_memset (exercises the full SIMD loop)
// ---------------------------------------------------------------------------
static void test_nt_memset_large() {
    constexpr std::size_t kSize = 4u << 20;  // 4 MiB
    constexpr uint8_t     kFill = 0xCD;

    auto raw = make_aligned(kSize, 32);
    uint8_t* p = align_ptr(raw, 32);

    nt_memset(p, kFill, kSize);

    // Spot-check every 4 KB page boundary and random interior
    for (std::size_t i = 0; i < kSize; i += 4096)
        assert(p[i] == kFill);
    assert(p[kSize - 1] == kFill);

    printf("PASS  test_nt_memset_large  (4 MiB)\n");
}

// ---------------------------------------------------------------------------
// Test 10: raw nt_store<__m128i> + sfence (when SSE2 available)
// ---------------------------------------------------------------------------
static void test_raw_nt_store() {
#if defined(__SSE2__)
    // 16-byte aligned storage
    alignas(16) int32_t buf[4] = {0, 0, 0, 0};

    __m128i val = _mm_set1_epi32(static_cast<int>(0xDEADBEEFu));
    cpu_engine::nt_store<__m128i>(reinterpret_cast<__m128i*>(buf), val);
    sfence();

    for (int i = 0; i < 4; ++i)
        assert(buf[i] == static_cast<int32_t>(0xDEADBEEF));

    printf("PASS  test_raw_nt_store  (__m128i  SSE2)\n");
#else
    printf("SKIP  test_raw_nt_store  (SSE2 not available)\n");
#endif
}

int main() {
    printf("ISA: ");
#if defined(__AVX2__)
    printf("AVX2 (256-bit NT stores)\n");
#elif defined(__SSE2__)
    printf("SSE2 (128-bit NT stores)\n");
#else
    printf("scalar fallback\n");
#endif

    test_sfence();
    test_nt_memset_zero();
    test_nt_memset_pattern();
    test_nt_memset_unaligned();
    test_nt_memset_small();
    test_nt_memset_boundaries();
    test_nt_memcpy_basic();
    test_nt_memcpy_unaligned();
    test_nt_memset_large();
    test_raw_nt_store();

    printf("\nAll nt_store tests passed.\n");
}
