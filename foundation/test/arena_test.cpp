#include "arena/arena.h"

#include <cassert>
#include <cstdio>
#include <cstring>
#include <latch>
#include <thread>
#include <vector>

using foundation::Arena;
using foundation::SizeClassedArena;
using foundation::ThreadLocalArena;

// ---------------------------------------------------------------------------
// Test 1: basic alloc returns non-null and is within the slab
// ---------------------------------------------------------------------------
static void test_arena_basic() {
    Arena a(64 * 1024);
    assert(a.ok());
    assert(a.used() == 0);
    assert(a.capacity() >= 64 * 1024);

    void* p = a.alloc(16);
    assert(p != nullptr);
    assert(a.owns(p));
    assert(a.used() >= 16);

    void* q = a.alloc(32);
    assert(q != nullptr);
    assert(q != p);
    assert(a.owns(q));

    printf("PASS  test_arena_basic\n");
}

// ---------------------------------------------------------------------------
// Test 2: alloc respects alignment
// ---------------------------------------------------------------------------
static void test_arena_alignment() {
    Arena a(1024 * 1024);

    for (std::size_t align : {1u, 8u, 16u, 32u, 64u, 128u, 256u}) {
        for (std::size_t size : {1u, 3u, 7u, 16u, 100u}) {
            void* p = a.alloc(size, align);
            assert(p != nullptr);
            assert(reinterpret_cast<uintptr_t>(p) % align == 0);
        }
    }

    printf("PASS  test_arena_alignment\n");
}

// ---------------------------------------------------------------------------
// Test 3: alloc returns nullptr when slab is full
// ---------------------------------------------------------------------------
static void test_arena_full() {
    Arena a(1024);  // tiny slab

    // Drain it
    while (a.alloc(64)) {}
    void* p = a.alloc(64);
    assert(p == nullptr);

    printf("PASS  test_arena_full\n");
}

// ---------------------------------------------------------------------------
// Test 4: reset rewinds cursor; subsequent allocs reuse the space
// ---------------------------------------------------------------------------
static void test_arena_reset() {
    Arena a(4096);

    void* first = a.alloc(256);
    assert(first != nullptr);
    std::size_t used_before = a.used();

    a.reset();
    assert(a.used() == 0);

    void* again = a.alloc(256);
    assert(again == first);  // same address — cursor rewound
    assert(a.used() == used_before);

    printf("PASS  test_arena_reset\n");
}

// ---------------------------------------------------------------------------
// Test 5: reset with page release doesn't corrupt subsequent allocs
// ---------------------------------------------------------------------------
static void test_arena_reset_release_pages() {
    Arena a(4096);

    void* p = a.alloc(128);
    assert(p != nullptr);
    std::memset(p, 0xAB, 128);

    a.reset(true);  // release physical pages back to OS

    // On next alloc the pages may be zero-filled (OS reclaimed them)
    void* q = a.alloc(128);
    assert(q != nullptr);  // must not crash
    // Write and read back to verify the page is accessible
    std::memset(q, 0xCD, 128);
    assert(static_cast<unsigned char*>(q)[0] == 0xCD);

    printf("PASS  test_arena_reset_release_pages\n");
}

// ---------------------------------------------------------------------------
// Test 6: owns() correctly identifies arena vs. external pointers
// ---------------------------------------------------------------------------
static void test_arena_owns() {
    Arena a(4096);
    void* p = a.alloc(64);
    assert(a.owns(p));

    int stack_var = 42;
    assert(!a.owns(&stack_var));

    auto* heap = new int(1);
    assert(!a.owns(heap));
    delete heap;

    printf("PASS  test_arena_owns\n");
}

// ---------------------------------------------------------------------------
// Test 7: size class index lookup
// ---------------------------------------------------------------------------
static void test_sc_class_index() {
    using foundation::SizeClassedArena;

    // Exact powers of 2
    assert(SizeClassedArena::class_index(8)    == 0);
    assert(SizeClassedArena::class_index(16)   == 1);
    assert(SizeClassedArena::class_index(32)   == 2);
    assert(SizeClassedArena::class_index(64)   == 3);
    assert(SizeClassedArena::class_index(128)  == 4);
    assert(SizeClassedArena::class_index(256)  == 5);
    assert(SizeClassedArena::class_index(512)  == 6);
    assert(SizeClassedArena::class_index(1024) == 7);

    // Round-up: anything in (N/2, N] maps to class N
    assert(SizeClassedArena::class_index(1)   == 0);  // rounds up to 8
    assert(SizeClassedArena::class_index(9)   == 1);  // rounds up to 16
    // 33 -> bit_ceil(33) = 64 -> ctz(64)-3 = 6-3 = 3
    assert(SizeClassedArena::class_index(33)  == 3);  // rounds up to 64
    assert(SizeClassedArena::class_index(513) == 7);  // rounds up to 1024

    // Large: returns -1
    assert(SizeClassedArena::class_index(1025) == -1);
    assert(SizeClassedArena::class_index(4096) == -1);

    printf("PASS  test_sc_class_index\n");
}

// ---------------------------------------------------------------------------
// Test 8: freelist recycling — free then alloc returns the same pointer
// ---------------------------------------------------------------------------
static void test_sc_freelist_recycling() {
    SizeClassedArena sc(64 * 1024);

    void* p = sc.alloc(32);
    assert(p != nullptr);
    assert(reinterpret_cast<uintptr_t>(p) % 32 == 0);  // 32-byte aligned

    sc.free(p, 32);

    void* q = sc.alloc(32);
    assert(q == p);  // same block recycled from freelist

    printf("PASS  test_sc_freelist_recycling\n");
}

// ---------------------------------------------------------------------------
// Test 9: allocation alignment matches size class
// ---------------------------------------------------------------------------
static void test_sc_alignment() {
    SizeClassedArena sc(1024 * 1024);

    for (std::size_t i = 0; i < SizeClassedArena::kNumClasses; ++i) {
        std::size_t sz = SizeClassedArena::kSizes[i];
        void* p = sc.alloc(sz);
        assert(p != nullptr);
        assert(reinterpret_cast<uintptr_t>(p) % sz == 0);
    }

    printf("PASS  test_sc_alignment\n");
}

// ---------------------------------------------------------------------------
// Test 10: large allocation (>1024) bypasses freelist, still works
// ---------------------------------------------------------------------------
static void test_sc_large_alloc() {
    SizeClassedArena sc(1024 * 1024);

    void* p = sc.alloc(2048);
    assert(p != nullptr);
    assert(sc.arena().owns(p));

    // free does nothing for large objects (no freelist)
    sc.free(p, 2048);

    // Alloc again at the same size — gets a fresh bump (not recycled)
    void* q = sc.alloc(2048);
    assert(q != nullptr);
    assert(q != p);  // NOT recycled — large objects are not on any freelist

    printf("PASS  test_sc_large_alloc\n");
}

// ---------------------------------------------------------------------------
// Test 11: reset clears freelists and rewinds arena
// ---------------------------------------------------------------------------
static void test_sc_reset() {
    SizeClassedArena sc(64 * 1024);

    void* p = sc.alloc(64);
    void* q = sc.alloc(64);
    sc.free(p, 64);
    sc.free(q, 64);

    sc.reset();
    assert(sc.arena().used() == 0);

    // After reset, freelist is cleared — next alloc bumps fresh from arena
    void* r = sc.alloc(64);
    // r could equal p (first address in the slab) because we rewound.
    // It must NOT be on a stale freelist (which would point into reset memory).
    assert(sc.arena().owns(r));

    printf("PASS  test_sc_reset\n");
}

// ---------------------------------------------------------------------------
// Test 12: thread-local arenas are independent (no address overlap)
// ---------------------------------------------------------------------------
static void test_thread_local_isolation() {
    // Threads must all be ALIVE simultaneously when we check pointers.
    // Without this, a finished thread's arena is destroyed (munmap), and
    // the next thread may get the same VA from mmap — a false collision.
    constexpr std::size_t kN = 4;
    std::vector<void*> ptrs(kN, nullptr);
    std::latch allocated(kN);  // all threads have allocated
    std::latch done(1);         // main signals threads to exit
    std::vector<std::thread> threads;

    for (std::size_t i = 0; i < kN; ++i) {
        threads.emplace_back([&ptrs, &allocated, &done, i]{
            ptrs[i] = ThreadLocalArena::alloc(64);
            allocated.count_down();
            done.wait();           // keep arena alive while main checks
        });
    }

    allocated.wait();  // all threads have their arenas live

    for (std::size_t i = 0; i < kN; ++i) assert(ptrs[i] != nullptr);
    for (std::size_t i = 0; i < kN; ++i)
        for (std::size_t j = i + 1; j < kN; ++j)
            assert(ptrs[i] != ptrs[j]);

    done.count_down();
    for (auto& t : threads) t.join();

    printf("PASS  test_thread_local_isolation\n");
}

// ---------------------------------------------------------------------------
// Test 13: hugepage request — either succeeds or falls back gracefully
// ---------------------------------------------------------------------------
static void test_hugepage_fallback() {
    // 2 MiB slab, try hugepage.  On Linux with hugepages configured this
    // will use MAP_HUGETLB; otherwise (most dev machines / macOS) it falls
    // back to regular mmap.  Either way the arena must work correctly.
    Arena a(2u << 20, /*try_hugepage=*/true);
    assert(a.ok());

    void* p = a.alloc(1024, 64);
    assert(p != nullptr);
    assert(reinterpret_cast<uintptr_t>(p) % 64 == 0);

    // is_hugepage() may be true or false — just verify it doesn't crash.
    (void)a.is_hugepage();

    printf("PASS  test_hugepage_fallback  (hugepage=%s)\n",
           a.is_hugepage() ? "yes" : "no (graceful fallback)");
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main() {
    test_arena_basic();
    test_arena_alignment();
    test_arena_full();
    test_arena_reset();
    test_arena_reset_release_pages();
    test_arena_owns();
    test_sc_class_index();
    test_sc_freelist_recycling();
    test_sc_alignment();
    test_sc_large_alloc();
    test_sc_reset();
    test_thread_local_isolation();
    test_hugepage_fallback();
    return 0;
}
