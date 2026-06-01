#include "epoch/epoch_reclaim.h"
#include "epoch/epoch_stack.h"

#include <atomic>
#include <barrier>
#include <cassert>
#include <cstdio>
#include <thread>
#include <vector>

template <typename T>
struct Tracked {
    T data;
    inline static std::atomic<int> live{0};

    explicit Tracked(T v) : data(v) { live.fetch_add(1, std::memory_order_relaxed); }
    Tracked(const Tracked& o)       : data(o.data) { live.fetch_add(1, std::memory_order_relaxed); }
    Tracked(Tracked&& o) noexcept   : data(std::move(o.data)) { live.fetch_add(1, std::memory_order_relaxed); }
    ~Tracked()                      { live.fetch_sub(1, std::memory_order_relaxed); }
    Tracked& operator=(const Tracked&) = default;
    Tracked& operator=(Tracked&&)      = default;
};

// Retire enough dummies to force all pending items through the 2-epoch lag.
// After this call, all objects retired before it will have been freed.
static void drain_epoch(foundation::EpochDomain& domain) {
    // 2 advances needed: one to move retired slot out of the current window,
    // one more to actually reclaim it. Each retire() triggers one advance
    // (when no active guards are present). 3 extra retires is sufficient
    // regardless of the current epoch position.
    for (int i = 0; i < 3; ++i) {
        int* d = new int(0);
        domain.retire(d);
    }
}

// ---------------------------------------------------------------------------
// Test 1: epoch advances on each retire (no active guards)
// ---------------------------------------------------------------------------
static void test_epoch_advances() {
    using T = Tracked<int>;
    assert(T::live.load() == 0);

    foundation::EpochDomain domain;
    assert(domain.epoch() == 0);

    foundation::EpochStack<T> stack(domain);
    stack.push(T{1});
    stack.push(T{2});
    stack.push(T{3});
    assert(T::live.load() == 3);

    // pop() opens and closes a guard, then retires the node.
    // With no competing active guards, each retire triggers an epoch advance.
    {
        T val{0};
        stack.pop(val);
        stack.pop(val);
        stack.pop(val);
    }

    // Force reclamation through the 2-epoch lag.
    drain_epoch(domain);

    assert(T::live.load() == 0);
    printf("PASS  test_epoch_advances  (epoch=%zu)\n",
           static_cast<std::size_t>(domain.epoch()));
}

// ---------------------------------------------------------------------------
// Test 2: active guard blocks a second epoch advance within the same section
//
// After the first retire inside a guard advances the epoch to G+1, our
// local_epoch is still G (we haven't refreshed it). A second retire inside
// the same guard calls try_advance(), which checks all active threads:
// our thread is active with local_epoch = G != G+1 → advance blocked.
// ---------------------------------------------------------------------------
static void test_guard_blocks_second_advance() {
    foundation::EpochDomain domain;

    uint64_t epoch_before = domain.epoch();

    {
        foundation::EpochDomain::Guard guard(domain);
        // local_epoch = G = epoch_before, active = true.

        int* p1 = new int(1);
        domain.retire(p1);
        // try_advance: our local_epoch = G matches current G → advance G→G+1. p1 queued.
        uint64_t mid = domain.epoch();
        assert(mid == epoch_before + 1);

        int* p2 = new int(2);
        domain.retire(p2);
        // try_advance: our local_epoch = epoch_before != G = epoch_before+1 → BLOCKED.
        assert(domain.epoch() == epoch_before + 1);  // still the same
    }
    // After the guard exits (active = false), retires can advance again.

    int* p3 = new int(3);
    domain.retire(p3);
    assert(domain.epoch() >= epoch_before + 2);  // at least one more advance

    drain_epoch(domain);
    printf("PASS  test_guard_blocks_second_advance\n");
}

// ---------------------------------------------------------------------------
// Test 3: EpochStack LIFO + live count
// ---------------------------------------------------------------------------
static void test_epoch_stack_lifo() {
    using T = Tracked<int>;
    assert(T::live.load() == 0);

    foundation::EpochDomain domain;
    foundation::EpochStack<T> stack(domain);

    for (int i = 0; i < 8; ++i) stack.push(T{i});
    assert(T::live.load() == 8);

    for (int i = 7; i >= 0; --i) {
        T val{-1};
        bool ok = stack.pop(val);
        assert(ok && val.data == i);
    }
    {
        T dummy{0};
        assert(!stack.pop(dummy));
    }
    assert(stack.empty());

    drain_epoch(domain);
    assert(T::live.load() == 0);
    printf("PASS  test_epoch_stack_lifo\n");
}

// ---------------------------------------------------------------------------
// Test 4: concurrent push-then-pop stress with live-count verification
// ---------------------------------------------------------------------------
static void test_concurrent_reclamation() {
    constexpr std::size_t kThreads        = 4;
    constexpr std::size_t kNodesPerThread = 10'000;
    constexpr std::size_t kTotal          = kThreads * kNodesPerThread;

    using T = Tracked<int>;
    assert(T::live.load() == 0);

    foundation::EpochDomain domain;
    foundation::EpochStack<T> stack(domain);

    std::atomic<int> pop_count{0};
    std::barrier sync(static_cast<std::ptrdiff_t>(kThreads));

    std::vector<std::thread> threads;
    threads.reserve(kThreads);
    for (std::size_t t = 0; t < kThreads; ++t) {
        threads.emplace_back([&, t]() {
            for (std::size_t i = 0; i < kNodesPerThread; ++i)
                stack.push(T{static_cast<int>(t * kNodesPerThread + i)});

            sync.arrive_and_wait();

            {
                T val{0};
                while (stack.pop(val))
                    pop_count.fetch_add(1, std::memory_order_relaxed);
            }
        });
    }
    for (auto& t : threads) t.join();

    drain_epoch(domain);

    assert(pop_count.load() == static_cast<int>(kTotal));
    assert(stack.empty());
    assert(T::live.load() == 0);
    printf("PASS  test_concurrent_reclamation  (%zu nodes, all freed)\n", kTotal);
}

// ---------------------------------------------------------------------------
// Test 5: concurrent mixed push+pop (EBR protects concurrent readers)
// ---------------------------------------------------------------------------
static void test_concurrent_push_pop_mixed() {
    constexpr std::size_t kThreads   = 4;
    constexpr std::size_t kPerThread = 10'000;

    using T = Tracked<int>;
    assert(T::live.load() == 0);

    foundation::EpochDomain domain;
    foundation::EpochStack<T> stack(domain);

    std::atomic<int> pop_count{0};
    std::atomic<bool> done{false};

    std::vector<std::thread> producers;
    producers.reserve(kThreads / 2);
    for (std::size_t t = 0; t < kThreads / 2; ++t) {
        producers.emplace_back([&, t]() {
            for (std::size_t i = 0; i < kPerThread; ++i)
                stack.push(T{static_cast<int>(t * kPerThread + i)});
        });
    }

    std::vector<std::thread> consumers;
    consumers.reserve(kThreads / 2);
    for (std::size_t t = 0; t < kThreads / 2; ++t) {
        consumers.emplace_back([&]() {
            T val{0};
            while (!done.load(std::memory_order_acquire) || !stack.empty()) {
                if (stack.pop(val))
                    pop_count.fetch_add(1, std::memory_order_relaxed);
            }
        });
    }

    for (auto& p : producers) p.join();
    done.store(true, std::memory_order_release);
    {
        T val{0};
        while (stack.pop(val))
            pop_count.fetch_add(1, std::memory_order_relaxed);
    }
    for (auto& c : consumers) c.join();

    drain_epoch(domain);

    const std::size_t expected = (kThreads / 2) * kPerThread;
    assert(pop_count.load() == static_cast<int>(expected));
    assert(stack.empty());
    assert(T::live.load() == 0);
    printf("PASS  test_concurrent_push_pop_mixed  (%zu nodes, all freed)\n", expected);
}

int main() {
    test_epoch_advances();
    test_guard_blocks_second_advance();
    test_epoch_stack_lifo();
    test_concurrent_reclamation();
    test_concurrent_push_pop_mixed();
    return 0;
}
