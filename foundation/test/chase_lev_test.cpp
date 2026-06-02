#include "chase_lev/chase_lev.h"

#include <atomic>
#include <barrier>
#include <cassert>
#include <cstdio>
#include <numeric>
#include <thread>
#include <vector>

// ---------------------------------------------------------------------------
// Test 1: LIFO order — push/pop on a single thread
// ---------------------------------------------------------------------------
static void test_lifo_order() {
    foundation::ChaseLevDeque<int> dq;

    assert(dq.empty());
    dq.push(1); dq.push(2); dq.push(3);
    assert(dq.size() == 3);

    auto v = dq.pop(); assert(v && *v == 3);
    v = dq.pop();      assert(v && *v == 2);
    v = dq.pop();      assert(v && *v == 1);
    v = dq.pop();      assert(!v);          // empty
    assert(dq.empty());

    printf("PASS  test_lifo_order\n");
}

// ---------------------------------------------------------------------------
// Test 2: steal is FIFO from the top — thief on a second thread
// ---------------------------------------------------------------------------
static void test_steal_fifo() {
    foundation::ChaseLevDeque<int> dq;

    // Push from owner (single thread, before spawning thief)
    dq.push(10); dq.push(20); dq.push(30);

    std::vector<int> stolen;
    std::thread thief([&]() {
        for (int i = 0; i < 3; ++i) {
            std::optional<int> v;
            while (!(v = dq.steal())) {}  // spin until success
            stolen.push_back(*v);
        }
    });
    thief.join();

    assert(stolen.size() == 3);
    assert(stolen[0] == 10 && stolen[1] == 20 && stolen[2] == 30);
    printf("PASS  test_steal_fifo\n");
}

// ---------------------------------------------------------------------------
// Test 3: concurrent — owner push/pop, thieves steal.
// Verify all N items dequeued exactly once (TSan primary check).
// ---------------------------------------------------------------------------
static void test_concurrent_owner_thieves() {
    constexpr int         kItems    = 100'000;
    constexpr std::size_t kThieves  = 3;

    foundation::ChaseLevDeque<int> dq;
    std::vector<std::atomic<int>> seen(static_cast<std::size_t>(kItems));
    for (auto& a : seen) a.store(0, std::memory_order_relaxed);

    std::atomic<bool> owner_done{false};
    std::barrier sync(static_cast<std::ptrdiff_t>(kThieves + 1));

    // Thieves: steal until owner signals done AND deque is empty.
    std::vector<std::thread> thieves;
    thieves.reserve(kThieves);
    for (std::size_t i = 0; i < kThieves; ++i) {
        thieves.emplace_back([&]() {
            sync.arrive_and_wait();
            while (!owner_done.load(std::memory_order_acquire) || !dq.empty()) {
                auto v = dq.steal();
                if (v) {
                    assert(*v >= 0 && *v < kItems);
                    seen[static_cast<std::size_t>(*v)].fetch_add(1, std::memory_order_relaxed);
                }
            }
        });
    }

    // Owner: push all items, then pop remaining ones.
    sync.arrive_and_wait();
    for (int i = 0; i < kItems; ++i)
        dq.push(i);

    // Pop from the owner side (some items may already have been stolen).
    while (true) {
        auto v = dq.pop();
        if (!v) break;
        assert(*v >= 0 && *v < kItems);
        seen[static_cast<std::size_t>(*v)].fetch_add(1, std::memory_order_relaxed);
    }

    owner_done.store(true, std::memory_order_release);
    for (auto& t : thieves) t.join();

    // Every item must have been seen exactly once.
    for (int i = 0; i < kItems; ++i)
        assert(seen[static_cast<std::size_t>(i)].load() == 1);

    printf("PASS  test_concurrent_owner_thieves  (%d items, %zu thieves)\n",
           kItems, kThieves);
}

// ---------------------------------------------------------------------------
// Test 4: grow — push past the initial capacity
// ---------------------------------------------------------------------------
static void test_grow() {
    constexpr std::size_t kInitCap = 4;     // intentionally tiny
    constexpr int         kItems   = 1000;  // forces multiple resizes

    foundation::ChaseLevDeque<int> dq(kInitCap);

    for (int i = 0; i < kItems; ++i) dq.push(i);
    assert(dq.size() == static_cast<std::size_t>(kItems));

    // Pop all in LIFO order.
    for (int i = kItems - 1; i >= 0; --i) {
        auto v = dq.pop();
        assert(v && *v == i);
    }
    assert(dq.empty());

    printf("PASS  test_grow  (%d items, initial_cap=%zu)\n", kItems, kInitCap);
}

// ---------------------------------------------------------------------------
// Test 5: pop vs. steal race on the last element
//
// Owner pushes exactly 1 item. A thief attempts to steal it concurrently.
// Exactly one of them should get it; the other should see nullopt.
// Run many times to exercise the race window.
// ---------------------------------------------------------------------------
static void test_last_element_race() {
    constexpr int kRounds = 10'000;

    for (int r = 0; r < kRounds; ++r) {
        foundation::ChaseLevDeque<int> dq;
        dq.push(42);

        std::atomic<int> got{0};

        std::thread thief([&]() {
            auto v = dq.steal();
            if (v) got.fetch_add(1, std::memory_order_relaxed);
        });

        auto v = dq.pop();
        if (v) got.fetch_add(1, std::memory_order_relaxed);

        thief.join();
        assert(got.load() == 1);  // exactly one thread got it
    }

    printf("PASS  test_last_element_race  (%d rounds)\n", kRounds);
}

// ---------------------------------------------------------------------------
// Test 6: multiple concurrent thieves — verify no item stolen twice
// ---------------------------------------------------------------------------
static void test_multiple_thieves_no_duplicate() {
    constexpr int         kItems   = 50'000;
    constexpr std::size_t kThieves = 4;

    foundation::ChaseLevDeque<int> dq;
    for (int i = 0; i < kItems; ++i) dq.push(i);

    std::atomic<bool> owner_done{false};
    std::vector<std::atomic<int>> seen(static_cast<std::size_t>(kItems));
    for (auto& a : seen) a.store(0, std::memory_order_relaxed);

    std::vector<std::thread> thieves;
    thieves.reserve(kThieves);
    for (std::size_t i = 0; i < kThieves; ++i) {
        thieves.emplace_back([&]() {
            while (!owner_done.load(std::memory_order_acquire) || !dq.empty()) {
                auto v = dq.steal();
                if (v) {
                    int prev = seen[static_cast<std::size_t>(*v)].fetch_add(1, std::memory_order_relaxed);
                    assert(prev == 0);  // no duplicate steals
                }
            }
        });
    }

    // Owner doesn't pop — let thieves take everything.
    owner_done.store(true, std::memory_order_release);
    for (auto& t : thieves) t.join();

    int total = 0;
    for (auto& a : seen) total += a.load();
    assert(total == kItems);

    printf("PASS  test_multiple_thieves_no_duplicate  (%d items, %zu thieves)\n",
           kItems, kThieves);
}

int main() {
    test_lifo_order();
    test_steal_fifo();
    test_concurrent_owner_thieves();
    test_grow();
    test_last_element_race();
    test_multiple_thieves_no_duplicate();
    return 0;
}
