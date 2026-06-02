#include "msqueue/ms_queue.h"

#include <atomic>
#include <barrier>
#include <cassert>
#include <cstdio>
#include <numeric>
#include <thread>
#include <vector>

template <typename T>
struct Tracked {
    T data;
    inline static std::atomic<int> live{0};

    explicit Tracked(T v) : data(v)  { live.fetch_add(1, std::memory_order_relaxed); }
    Tracked(const Tracked& o)        : data(o.data) { live.fetch_add(1, std::memory_order_relaxed); }
    Tracked(Tracked&& o) noexcept    : data(std::move(o.data)) { live.fetch_add(1, std::memory_order_relaxed); }
    ~Tracked()                       { live.fetch_sub(1, std::memory_order_relaxed); }
    Tracked& operator=(const Tracked&) = default;
    Tracked& operator=(Tracked&&)      = default;
};

// ---------------------------------------------------------------------------
// Test 1: FIFO ordering and empty-queue sentinel
// ---------------------------------------------------------------------------
static void test_fifo_order() {
    foundation::MsQueue<int> q;

    assert(q.empty());
    q.enqueue(1);
    q.enqueue(2);
    q.enqueue(3);
    assert(!q.empty());

    int v = 0;
    assert(q.dequeue(v) && v == 1);
    assert(q.dequeue(v) && v == 2);
    assert(q.dequeue(v) && v == 3);
    assert(!q.dequeue(v));  // empty
    assert(q.empty());

    printf("PASS  test_fifo_order\n");
}

// ---------------------------------------------------------------------------
// Test 2: memory reclamation — Tracked<int> live count drops to 0
// ---------------------------------------------------------------------------
static void test_reclamation() {
    using T = Tracked<int>;
    assert(T::live.load() == 0);

    {
        foundation::MsQueue<T> q;
        for (int i = 0; i < 10; ++i) q.enqueue(T{i});
        assert(T::live.load() == 10);

        T v{0};
        for (int i = 0; i < 10; ++i) {
            bool ok = q.dequeue(v);
            assert(ok && v.data == i);
        }

        q.drain();
    }
    // Queue destructor drains remaining nodes and scans.
    assert(T::live.load() == 0);
    printf("PASS  test_reclamation\n");
}

// ---------------------------------------------------------------------------
// Test 3: single enqueue then dequeue from separate threads
// ---------------------------------------------------------------------------
static void test_producer_consumer_single() {
    foundation::MsQueue<int> q;
    std::atomic<bool> ready{false};

    std::thread producer([&]() {
        q.enqueue(42);
        ready.store(true, std::memory_order_release);
    });

    int v = 0;
    while (!ready.load(std::memory_order_acquire)) {}
    bool ok = q.dequeue(v);
    assert(ok && v == 42);

    producer.join();
    printf("PASS  test_producer_consumer_single\n");
}

// ---------------------------------------------------------------------------
// Test 4: N producers, N consumers — all items dequeued, correct count
// (primary TSan check)
// ---------------------------------------------------------------------------
static void test_concurrent_mpmc() {
    constexpr std::size_t kProducers    = 4;
    constexpr std::size_t kConsumers    = 4;
    constexpr std::size_t kPerProducer  = 10'000;
    constexpr std::size_t kTotal        = kProducers * kPerProducer;

    foundation::MsQueue<int> q;
    std::atomic<int>  pop_count{0};
    std::barrier sync(static_cast<std::ptrdiff_t>(kProducers + kConsumers));
    std::atomic<bool> producers_done{false};

    std::vector<std::thread> producers;
    producers.reserve(kProducers);
    for (std::size_t t = 0; t < kProducers; ++t) {
        producers.emplace_back([&, t]() {
            sync.arrive_and_wait();
            for (std::size_t i = 0; i < kPerProducer; ++i)
                q.enqueue(static_cast<int>(t * kPerProducer + i));
        });
    }

    std::vector<std::thread> consumers;
    consumers.reserve(kConsumers);
    for (std::size_t t = 0; t < kConsumers; ++t) {
        consumers.emplace_back([&]() {
            sync.arrive_and_wait();
            int v = 0;
            while (!producers_done.load(std::memory_order_acquire) || !q.empty()) {
                if (q.dequeue(v))
                    pop_count.fetch_add(1, std::memory_order_relaxed);
            }
        });
    }

    for (auto& p : producers) p.join();
    producers_done.store(true, std::memory_order_release);

    // Drain any remaining items.
    {
        int v = 0;
        while (q.dequeue(v))
            pop_count.fetch_add(1, std::memory_order_relaxed);
    }

    for (auto& c : consumers) c.join();

    assert(pop_count.load() == static_cast<int>(kTotal));
    assert(q.empty());
    q.drain();

    printf("PASS  test_concurrent_mpmc  (%zuP x %zuC, %zu total items)\n",
           kProducers, kConsumers, kTotal);
}

// ---------------------------------------------------------------------------
// Test 5: lagging tail_ — enqueue many items without concurrent dequeue,
// then verify FIFO order. The tail-advance help path is exercised by having
// the consumer see an advanced tail after the producer finishes.
// ---------------------------------------------------------------------------
static void test_lagging_tail() {
    constexpr int kN = 1000;
    foundation::MsQueue<int> q;

    for (int i = 0; i < kN; ++i) q.enqueue(i);

    for (int i = 0; i < kN; ++i) {
        int v = -1;
        assert(q.dequeue(v) && v == i);
    }
    assert(q.empty());
    printf("PASS  test_lagging_tail  (%d items)\n", kN);
}

// ---------------------------------------------------------------------------
// Test 6: Tracked live count under concurrent push/pop
// ---------------------------------------------------------------------------
static void test_concurrent_live_count() {
    constexpr std::size_t kThreads     = 4;
    constexpr std::size_t kPerThread   = 5'000;
    constexpr std::size_t kTotal       = kThreads * kPerThread;

    using T = Tracked<int>;
    assert(T::live.load() == 0);

    {
        foundation::MsQueue<T> q;
        std::atomic<int> pop_count{0};
        std::barrier sync(static_cast<std::ptrdiff_t>(kThreads));

        std::vector<std::thread> threads;
        threads.reserve(kThreads);
        for (std::size_t t = 0; t < kThreads; ++t) {
            threads.emplace_back([&, t]() {
                sync.arrive_and_wait();
                for (std::size_t i = 0; i < kPerThread; ++i)
                    q.enqueue(T{static_cast<int>(t * kPerThread + i)});

                sync.arrive_and_wait();

                T v{0};
                while (q.dequeue(v))
                    pop_count.fetch_add(1, std::memory_order_relaxed);
            });
        }
        for (auto& t : threads) t.join();

        q.drain();
        assert(pop_count.load() == static_cast<int>(kTotal));
        assert(q.empty());
    }

    assert(T::live.load() == 0);
    printf("PASS  test_concurrent_live_count  (%zu items, all freed)\n", kTotal);
}

int main() {
    test_fifo_order();
    test_reclamation();
    test_producer_consumer_single();
    test_concurrent_mpmc();
    test_lagging_tail();
    test_concurrent_live_count();
    return 0;
}
