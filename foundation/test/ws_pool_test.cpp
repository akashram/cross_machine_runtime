#include "ws_pool/ws_pool.h"

#include <atomic>
#include <cassert>
#include <chrono>
#include <cstdio>
#include <numeric>
#include <thread>
#include <vector>

// ---------------------------------------------------------------------------
// Test 1: basic submit + wait
// ---------------------------------------------------------------------------
static void test_submit_wait() {
    foundation::WorkStealingPool pool(4);

    std::atomic<int> counter{0};
    for (int i = 0; i < 1000; ++i)
        pool.submit([&]{ counter.fetch_add(1, std::memory_order_relaxed); });

    pool.wait();
    assert(counter.load() == 1000);
    printf("PASS  test_submit_wait  (1000 tasks)\n");
}

// ---------------------------------------------------------------------------
// Test 2: parallel_for correctness
// ---------------------------------------------------------------------------
static void test_parallel_for() {
    foundation::WorkStealingPool pool(4);

    constexpr std::size_t N = 10'000;
    std::vector<std::atomic<int>> hits(N);
    for (auto& h : hits) h.store(0, std::memory_order_relaxed);

    pool.parallel_for(N, [&](std::size_t i){
        hits[i].fetch_add(1, std::memory_order_relaxed);
    });

    for (std::size_t i = 0; i < N; ++i)
        assert(hits[i].load() == 1);

    printf("PASS  test_parallel_for  (%zu tasks, each executed exactly once)\n", N);
}

// ---------------------------------------------------------------------------
// Test 3: TaskGroup — multiple waves, each must complete before the next
// ---------------------------------------------------------------------------
static void test_task_group_waves() {
    foundation::WorkStealingPool pool(4);
    std::atomic<int> stage{0};

    for (int wave = 0; wave < 5; ++wave) {
        foundation::TaskGroup group(pool);
        int expected_stage = wave;
        for (int t = 0; t < 100; ++t) {
            group.run([&, expected_stage]{
                assert(stage.load(std::memory_order_acquire) == expected_stage);
            });
        }
        group.wait();
        stage.fetch_add(1, std::memory_order_release);
    }

    assert(stage.load() == 5);
    printf("PASS  test_task_group_waves  (5 waves of 100 tasks)\n");
}

// ---------------------------------------------------------------------------
// Test 4: tasks spawning child tasks (worker→worker locality path)
//
// Each outer task submits kInner child tasks via pool.submit() WITHOUT
// blocking. Blocking inside a task (e.g. inner.wait()) would deadlock if all
// workers are occupied waiting — nobody left to run the inner tasks.
// The non-blocking pattern: outer tasks fire child tasks and return; the pool's
// pending_ counter tracks all submitted work including children.
// ---------------------------------------------------------------------------
static void test_nested_submit() {
    foundation::WorkStealingPool pool(4);

    constexpr int kOuter = 50;
    constexpr int kInner = 50;
    std::atomic<int> total{0};

    {
        foundation::TaskGroup outer(pool);
        for (int i = 0; i < kOuter; ++i) {
            outer.run([&]{
                // Running on a worker thread: child submits go to own Chase-Lev
                // deque (LIFO locality). No blocking — we return immediately.
                for (int j = 0; j < kInner; ++j)
                    pool.submit([&]{
                        total.fetch_add(1, std::memory_order_relaxed);
                    });
            });
        }
        outer.wait();  // wait for outer tasks (they've submitted but not blocked)
    }
    pool.wait();  // drain all child tasks (tracked by pool's pending_ counter)

    assert(total.load() == kOuter * kInner);
    printf("PASS  test_nested_submit  (%d outer x %d inner = %d child tasks)\n",
           kOuter, kInner, kOuter * kInner);
}

// ---------------------------------------------------------------------------
// Test 5: parallel sum — correctness of a divide-and-conquer pattern
// ---------------------------------------------------------------------------
static void test_parallel_sum() {
    foundation::WorkStealingPool pool(4);

    constexpr std::size_t N = 1'000'000;
    std::vector<int> data(N);
    std::iota(data.begin(), data.end(), 0);

    // Partition into chunks, sum each chunk in parallel.
    constexpr std::size_t kChunks = 100;
    constexpr std::size_t kChunkSize = N / kChunks;
    std::vector<std::atomic<long long>> chunk_sums(kChunks);
    for (auto& s : chunk_sums) s.store(0, std::memory_order_relaxed);

    pool.parallel_for(kChunks, [&](std::size_t c){
        long long s = 0;
        std::size_t start = c * kChunkSize;
        std::size_t end   = start + kChunkSize;
        for (std::size_t i = start; i < end; ++i) s += data[i];
        chunk_sums[c].store(s, std::memory_order_relaxed);
    });

    long long total = 0;
    for (auto& s : chunk_sums) total += s.load();
    long long expected = static_cast<long long>(N) * (N - 1) / 2;
    assert(total == expected);

    printf("PASS  test_parallel_sum  (N=%zu, sum=%lld)\n", N, total);
}

// ---------------------------------------------------------------------------
// Test 6: wait() with no submitted tasks returns immediately
// ---------------------------------------------------------------------------
static void test_wait_empty() {
    foundation::WorkStealingPool pool(2);
    pool.wait();  // must not block
    printf("PASS  test_wait_empty\n");
}

// ---------------------------------------------------------------------------
// Test 7: stress — many concurrent submits from multiple external threads
// ---------------------------------------------------------------------------
static void test_concurrent_external_submit() {
    constexpr int kSubmitters = 4;
    constexpr int kPerSubmitter = 5'000;

    foundation::WorkStealingPool pool(4);
    std::atomic<int> count{0};

    std::vector<std::thread> submitters;
    submitters.reserve(kSubmitters);
    for (int t = 0; t < kSubmitters; ++t) {
        submitters.emplace_back([&]{
            for (int i = 0; i < kPerSubmitter; ++i)
                pool.submit([&]{ count.fetch_add(1, std::memory_order_relaxed); });
        });
    }
    for (auto& t : submitters) t.join();

    pool.wait();
    assert(count.load() == kSubmitters * kPerSubmitter);
    printf("PASS  test_concurrent_external_submit  (%d tasks)\n",
           kSubmitters * kPerSubmitter);
}

int main() {
    test_submit_wait();
    test_parallel_for();
    test_task_group_waves();
    test_nested_submit();
    test_parallel_sum();
    test_wait_empty();
    test_concurrent_external_submit();
    return 0;
}
