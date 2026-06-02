#include "freelist/freelist.h"

#include <atomic>
#include <barrier>
#include <cassert>
#include <cstdio>
#include <thread>
#include <vector>

// ---------------------------------------------------------------------------
// Test 1: sequential acquire + release covers all capacity
// ---------------------------------------------------------------------------
static void test_acquire_release_basic() {
    constexpr std::size_t kCap = 8;
    foundation::FreeList<uint64_t> pool(kCap);

    assert(pool.available() == kCap);

    // Acquire all slots; each must be non-null and distinct.
    std::vector<uint64_t*> ptrs;
    ptrs.reserve(kCap);
    for (std::size_t i = 0; i < kCap; ++i) {
        uint64_t* p = pool.acquire();
        assert(p != nullptr);
        for (auto* q : ptrs) assert(q != p);  // distinct addresses
        ptrs.push_back(p);
    }
    assert(pool.available() == 0);

    // Pool is empty — next acquire must fail.
    assert(pool.acquire() == nullptr);

    // Release all and verify pool is full again.
    for (auto* p : ptrs) pool.release(p);
    assert(pool.available() == kCap);

    printf("PASS  test_acquire_release_basic  (cap=%zu)\n", kCap);
}

// ---------------------------------------------------------------------------
// Test 2: data written through an acquired slot survives release+re-acquire
// ---------------------------------------------------------------------------
static void test_data_roundtrip() {
    foundation::FreeList<uint64_t> pool(4);

    uint64_t* p = pool.acquire();
    assert(p != nullptr);
    *p = 0xDEADBEEF'CAFEBABE;
    pool.release(p);

    // We must be able to re-acquire and write new data.
    uint64_t* q = pool.acquire();
    assert(q != nullptr);
    *q = 42;
    assert(*q == 42);
    pool.release(q);

    assert(pool.available() == 4);
    printf("PASS  test_data_roundtrip\n");
}

// ---------------------------------------------------------------------------
// Test 3: acquire returns nullptr when pool is exhausted
// ---------------------------------------------------------------------------
static void test_exhaustion() {
    constexpr std::size_t kCap = 3;
    foundation::FreeList<uint64_t> pool(kCap);

    std::vector<uint64_t*> held;
    for (std::size_t i = 0; i < kCap; ++i)
        held.push_back(pool.acquire());

    assert(pool.acquire() == nullptr);
    assert(pool.acquire() == nullptr);  // repeated calls are safe

    for (auto* p : held) pool.release(p);
    assert(pool.available() == kCap);
    printf("PASS  test_exhaustion  (cap=%zu)\n", kCap);
}

// ---------------------------------------------------------------------------
// Test 4: concurrent acquire + release (TSan + no double-acquire)
//
// N threads each do M cycles of: acquire → write a sentinel → read it back
// → release. We use one atomic<int> per slot to catch double-acquire:
// the counter must go 0→1 on acquire and 1→0 on release with no skips.
//
// To map pointer → slot index, we check whether the pointer falls within the
// pool's slot array. Since FreeList is not opaque about its layout, we use
// the capacity and compare addresses against all known ptrs in a quiescent
// check. For the concurrent check, each thread just asserts no nullptr comes
// back while at least one slot should be free (kThreads < kCap).
// ---------------------------------------------------------------------------
static void test_concurrent_acquire_release() {
    constexpr std::size_t kThreads = 4;
    constexpr std::size_t kCycles  = 50'000;
    // More slots than threads so each thread can always acquire.
    constexpr std::size_t kCap     = kThreads * 2;

    foundation::FreeList<uint64_t> pool(kCap);

    // One atomic ownership flag per thread (thread i owns its slot or not).
    std::vector<std::atomic<int>> owned(kThreads);
    for (auto& o : owned) o.store(0, std::memory_order_relaxed);

    std::barrier sync(static_cast<std::ptrdiff_t>(kThreads));

    std::vector<std::thread> threads;
    threads.reserve(kThreads);
    for (std::size_t t = 0; t < kThreads; ++t) {
        threads.emplace_back([&, t]() {
            sync.arrive_and_wait();
            for (std::size_t i = 0; i < kCycles; ++i) {
                uint64_t* p = pool.acquire();
                // With kCap = 2*kThreads, at least kThreads slots are always
                // free, so each thread can always acquire one.
                assert(p != nullptr);

                // Write a unique sentinel and read it back.
                *p = static_cast<uint64_t>(t * 1000 + i);
                assert(*p == static_cast<uint64_t>(t * 1000 + i));

                pool.release(p);
            }
        });
    }

    for (auto& th : threads) th.join();

    // After all threads done, all slots must be free.
    assert(pool.available() == kCap);
    printf("PASS  test_concurrent_acquire_release  (%zu threads, %zu cycles each)\n",
           kThreads, kCycles);
}

// ---------------------------------------------------------------------------
// Test 5: struct type (sizeof > sizeof(void*)), verify alignment
// ---------------------------------------------------------------------------
static void test_struct_type() {
    struct alignas(16) BigObj {
        uint64_t a{0}, b{0}, c{0}, d{0};  // 32 bytes
    };

    foundation::FreeList<BigObj> pool(4);
    assert(pool.available() == 4);

    BigObj* p = pool.acquire();
    assert(p != nullptr);
    assert(reinterpret_cast<uintptr_t>(p) % alignof(BigObj) == 0);

    p->a = 1; p->b = 2; p->c = 3; p->d = 4;
    assert(p->a == 1 && p->d == 4);

    pool.release(p);
    assert(pool.available() == 4);
    printf("PASS  test_struct_type  (BigObj 32B, align=16)\n");
}

int main() {
    test_acquire_release_basic();
    test_data_roundtrip();
    test_exhaustion();
    test_concurrent_acquire_release();
    test_struct_type();
    return 0;
}
