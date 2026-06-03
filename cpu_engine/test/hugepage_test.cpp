#include "hugepage/hugepage.h"

#include <cassert>
#include <cstdio>
#include <cstring>
#include <utility>

using cpu_engine::HugeRegion;

// ---------------------------------------------------------------------------
// Test 1: small allocation succeeds and is readable/writable
// ---------------------------------------------------------------------------
static void test_alloc_small() {
    auto r = HugeRegion::alloc(64u << 10);  // 64 KiB
    assert(r.ok());
    assert(r.data() != nullptr);
    assert(r.size() >= 64u << 10);

    // Write and read back
    auto* b = static_cast<char*>(r.data());
    b[0]            = static_cast<char>(0xAB);
    b[r.size() - 1] = static_cast<char>(0xCD);
    assert(static_cast<unsigned char>(b[0])            == 0xABu);
    assert(static_cast<unsigned char>(b[r.size() - 1]) == 0xCDu);

    printf("PASS  test_alloc_small  (size=%zu  huge=%s)\n",
           r.size(), r.is_huge() ? "yes" : "no");
}

// ---------------------------------------------------------------------------
// Test 2: large allocation (exceeds any TLB capacity with 4 KB pages)
// ---------------------------------------------------------------------------
static void test_alloc_large() {
    auto r = HugeRegion::alloc(128u << 20, /*try_huge=*/false);  // 128 MiB, 4 KB pages
    assert(r.ok());
    assert(!r.is_huge());
    assert(r.size() == 128u << 20);

    // Write pattern to first and last page
    auto* b = static_cast<char*>(r.data());
    std::memset(b, 0x42, HugeRegion::kSmallPageSize);
    assert(b[0] == 0x42);

    printf("PASS  test_alloc_large  (128 MiB, 4KB pages)\n");
}

// ---------------------------------------------------------------------------
// Test 3: hugepage attempt — Linux gets real huge pages; macOS gracefully skips
// ---------------------------------------------------------------------------
static void test_alloc_hugepage() {
    auto r = HugeRegion::alloc(HugeRegion::kHugePageSize * 4, /*try_huge=*/true);
    assert(r.ok());
    assert(r.size() >= HugeRegion::kHugePageSize * 4);

#ifdef __linux__
    // On Linux, we either got huge pages or fell back — both are valid.
    printf("PASS  test_alloc_hugepage  (huge=%s  size=%zu)\n",
           r.is_huge() ? "yes" : "no (pool empty — set nr_hugepages)",
           r.size());
#else
    // macOS: always 4 KB pages
    assert(!r.is_huge());
    printf("PASS  test_alloc_hugepage  (macOS: no huge pages, 4KB fallback)\n");
#endif
}

// ---------------------------------------------------------------------------
// Test 4: expected_tlb_entries() math
// ---------------------------------------------------------------------------
static void test_tlb_entries() {
    // 4 KB pages
    {
        auto r = HugeRegion::alloc(64u << 20, /*try_huge=*/false);  // 64 MiB
        assert(r.ok() && !r.is_huge());
        std::size_t expected = (64u << 20) / HugeRegion::kSmallPageSize;  // 16384
        assert(r.expected_tlb_entries() == expected);
    }

#ifdef __linux__
    // 2 MB huge pages (only meaningful on Linux)
    {
        auto r = HugeRegion::alloc(64u << 20, /*try_huge=*/true);
        assert(r.ok());
        if (r.is_huge()) {
            std::size_t expected = (64u << 20) / HugeRegion::kHugePageSize;  // 32
            assert(r.expected_tlb_entries() == expected);
            printf("PASS  test_tlb_entries  "
                   "(4KB: 16384 entries, 2MB: 32 entries — 512x reduction)\n");
            return;
        }
    }
#endif
    printf("PASS  test_tlb_entries  (4KB: 16384 entries checked)\n");
}

// ---------------------------------------------------------------------------
// Test 5: prefault() makes all pages resident before benchmarking
// ---------------------------------------------------------------------------
static void test_prefault() {
    auto r = HugeRegion::alloc(8u << 20, /*try_huge=*/false);  // 8 MiB
    assert(r.ok());

    r.prefault();

    // All pages should now be writable without triggering faults
    auto* b = static_cast<char*>(r.data());
    for (std::size_t off = 0; off < r.size(); off += HugeRegion::kSmallPageSize)
        b[off] = static_cast<char>(off & 0xFF);

    // Read back last page
    assert(b[r.size() - HugeRegion::kSmallPageSize] ==
           static_cast<char>((r.size() - HugeRegion::kSmallPageSize) & 0xFF));

    printf("PASS  test_prefault  (8 MiB, all pages touched)\n");
}

// ---------------------------------------------------------------------------
// Test 6: move semantics transfer ownership correctly
// ---------------------------------------------------------------------------
static void test_move() {
    auto r1 = HugeRegion::alloc(4u << 20, /*try_huge=*/false);
    assert(r1.ok());
    void* original_ptr = r1.data();

    auto r2 = std::move(r1);
    assert(!r1.ok());         // source is now empty
    assert(r1.data() == nullptr);
    assert(r2.ok());
    assert(r2.data() == original_ptr);  // same backing memory

    // Move assignment
    HugeRegion r3 = HugeRegion::alloc(4u << 20, /*try_huge=*/false);
    assert(r3.ok());
    r3 = std::move(r2);
    assert(!r2.ok());
    assert(r3.data() == original_ptr);

    printf("PASS  test_move\n");
}

// ---------------------------------------------------------------------------
// Test 7: write/read full pattern through a region
// ---------------------------------------------------------------------------
static void test_write_read_pattern() {
    constexpr std::size_t kSize = 2u << 20;  // 2 MiB
    auto r = HugeRegion::alloc(kSize, /*try_huge=*/false);
    assert(r.ok());

    auto* b = static_cast<uint32_t*>(r.data());
    std::size_t n = kSize / sizeof(uint32_t);
    for (std::size_t i = 0; i < n; ++i)
        b[i] = static_cast<uint32_t>(i ^ 0xDEADBEEFu);
    for (std::size_t i = 0; i < n; ++i)
        assert(b[i] == static_cast<uint32_t>(i ^ 0xDEADBEEFu));

    printf("PASS  test_write_read_pattern  (%zu uint32 values)\n", n);
}

// ---------------------------------------------------------------------------
// Test 8: zero-size alloc returns empty (not a crash)
// ---------------------------------------------------------------------------
static void test_zero_size() {
    auto r = HugeRegion::alloc(0);
    assert(!r.ok());
    assert(r.data() == nullptr);
    printf("PASS  test_zero_size\n");
}

int main() {
    test_alloc_small();
    test_alloc_large();
    test_alloc_hugepage();
    test_tlb_entries();
    test_prefault();
    test_move();
    test_write_read_pattern();
    test_zero_size();
    printf("\nAll hugepage tests passed.\n");
}
