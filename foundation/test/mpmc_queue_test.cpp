#include "mpmc_queue.h"

#include <algorithm>
#include <atomic>
#include <cassert>
#include <cstdio>
#include <numeric>
#include <thread>
#include <vector>

using foundation::MpmcQueue;

// ---------------------------------------------------------------------------
// Minimal test harness (same pattern as spsc_queue_test)
// ---------------------------------------------------------------------------

static int g_pass = 0;
static int g_fail = 0;

#define CHECK(cond)                                                          \
    do {                                                                     \
        if (cond) {                                                          \
            ++g_pass;                                                        \
        } else {                                                             \
            ++g_fail;                                                        \
            fprintf(stderr, "FAIL  %s:%d  %s\n", __FILE__, __LINE__, #cond); \
        }                                                                    \
    } while (0)

// ---------------------------------------------------------------------------
// Single-threaded correctness
// ---------------------------------------------------------------------------

static void test_empty_on_construction() {
    MpmcQueue<int, 4> q;
    int val;
    CHECK(!q.pop(val));
}

static void test_capacity() {
    // MpmcQueue capacity is exactly N (no sentinel slot wasted)
    static_assert(MpmcQueue<int, 4>::capacity() == 4);
    static_assert(MpmcQueue<int, 8>::capacity() == 8);
    CHECK(true);
}

static void test_push_pop_single() {
    MpmcQueue<int, 4> q;
    CHECK(q.push(99));
    int val = 0;
    CHECK(q.pop(val));
    CHECK(val == 99);
    CHECK(!q.pop(val));
}

static void test_full_queue() {
    MpmcQueue<int, 4> q;  // capacity = 4
    CHECK(q.push(1));
    CHECK(q.push(2));
    CHECK(q.push(3));
    CHECK(q.push(4));
    CHECK(!q.push(5));  // full
}

static void test_fifo_order_single_thread() {
    MpmcQueue<int, 8> q;
    for (int i = 0; i < 8; ++i) CHECK(q.push(i));
    for (int i = 0; i < 8; ++i) {
        int val = -1;
        CHECK(q.pop(val));
        CHECK(val == i);
    }
    int val;
    CHECK(!q.pop(val));
}

static void test_wraparound() {
    MpmcQueue<uint32_t, 4> q;  // capacity = 4
    constexpr int kRounds = 30;
    for (int r = 0; r < kRounds; ++r) {
        for (uint32_t i = 0; i < 4; ++i) CHECK(q.push(i));
        for (uint32_t i = 0; i < 4; ++i) {
            uint32_t val = 0xDEAD;
            CHECK(q.pop(val));
            CHECK(val == i);
        }
    }
}

// ---------------------------------------------------------------------------
// Multi-threaded correctness
//
// This is the core TSan target. It exercises concurrent push/pop and verifies:
//   1. No items lost — exactly kItems items are received.
//   2. No duplicates — each value appears exactly once in the received set.
//   3. No data races — TSan validates the memory ordering is correct.
//
// We do NOT require FIFO ordering across producers: with N concurrent
// producers, the interleaving is non-deterministic. What we require is that
// each individual value is transferred exactly once and intact.
//
// Termination:
//   Producers use fetch_add on a shared counter to claim a unique value. Each
//   producer stops when it claims a value >= kItems. Consumers use a shared
//   pop_count to know when all kItems have been received.
// ---------------------------------------------------------------------------

static void test_mpsc(int n_producers) {
    constexpr std::size_t kItems = 50'000;
    MpmcQueue<uint64_t, 1024> q;

    std::atomic<uint64_t> push_idx{0};
    std::vector<uint64_t> received(kItems);
    std::atomic<std::size_t> pop_count{0};

    std::vector<std::thread> producers;
    producers.reserve(static_cast<std::size_t>(n_producers));
    for (int t = 0; t < n_producers; ++t) {
        producers.emplace_back([&]() {
            for (;;) {
                uint64_t val = push_idx.fetch_add(1, std::memory_order_relaxed);
                if (val >= kItems) break;
                while (!q.push(val)) {}
            }
        });
    }

    std::thread consumer([&]() {
        while (pop_count.load(std::memory_order_relaxed) < kItems) {
            uint64_t val;
            if (q.pop(val)) {
                std::size_t idx = pop_count.fetch_add(1, std::memory_order_relaxed);
                received[idx] = val;
            }
        }
    });

    for (auto& t : producers) t.join();
    consumer.join();

    std::sort(received.begin(), received.end());
    bool ok = true;
    for (std::size_t i = 0; i < kItems; ++i)
        if (received[i] != i) { ok = false; break; }
    CHECK(ok);
}

static void test_spmc(int n_consumers) {
    constexpr std::size_t kItems = 50'000;
    MpmcQueue<uint64_t, 1024> q;

    std::atomic<std::size_t> pop_count{0};
    std::vector<std::vector<uint64_t>> per_consumer(
        static_cast<std::size_t>(n_consumers));

    std::thread producer([&]() {
        for (uint64_t i = 0; i < kItems; ++i)
            while (!q.push(i)) {}
    });

    std::vector<std::thread> consumers;
    consumers.reserve(static_cast<std::size_t>(n_consumers));
    for (int t = 0; t < n_consumers; ++t) {
        consumers.emplace_back([&, t]() {
            while (pop_count.load(std::memory_order_relaxed) < kItems) {
                uint64_t val;
                if (q.pop(val)) {
                    per_consumer[static_cast<std::size_t>(t)].push_back(val);
                    pop_count.fetch_add(1, std::memory_order_relaxed);
                }
            }
        });
    }

    producer.join();
    for (auto& t : consumers) t.join();

    std::vector<uint64_t> all;
    all.reserve(kItems);
    for (auto& v : per_consumer)
        all.insert(all.end(), v.begin(), v.end());
    std::sort(all.begin(), all.end());
    CHECK(all.size() == kItems);
    bool ok = true;
    for (std::size_t i = 0; i < kItems; ++i)
        if (all[i] != i) { ok = false; break; }
    CHECK(ok);
}

static void test_mpmc(int n_producers, int n_consumers) {
    constexpr std::size_t kItems = 100'000;
    MpmcQueue<uint64_t, 2048> q;

    std::atomic<uint64_t> push_idx{0};
    std::atomic<std::size_t> pop_count{0};
    std::vector<std::vector<uint64_t>> per_consumer(
        static_cast<std::size_t>(n_consumers));

    std::vector<std::thread> producers;
    producers.reserve(static_cast<std::size_t>(n_producers));
    for (int t = 0; t < n_producers; ++t) {
        producers.emplace_back([&]() {
            for (;;) {
                uint64_t val = push_idx.fetch_add(1, std::memory_order_relaxed);
                if (val >= kItems) break;
                while (!q.push(val)) {}
            }
        });
    }

    std::vector<std::thread> consumers;
    consumers.reserve(static_cast<std::size_t>(n_consumers));
    for (int t = 0; t < n_consumers; ++t) {
        consumers.emplace_back([&, t]() {
            while (pop_count.load(std::memory_order_relaxed) < kItems) {
                uint64_t val;
                if (q.pop(val)) {
                    per_consumer[static_cast<std::size_t>(t)].push_back(val);
                    pop_count.fetch_add(1, std::memory_order_relaxed);
                }
            }
        });
    }

    for (auto& t : producers) t.join();
    for (auto& t : consumers) t.join();

    std::vector<uint64_t> all;
    all.reserve(kItems);
    for (auto& v : per_consumer)
        all.insert(all.end(), v.begin(), v.end());
    std::sort(all.begin(), all.end());
    CHECK(all.size() == kItems);
    bool ok = true;
    for (std::size_t i = 0; i < kItems; ++i)
        if (all[i] != i) { ok = false; break; }
    CHECK(ok);
}

// ---------------------------------------------------------------------------

int main() {
    test_empty_on_construction();
    test_capacity();
    test_push_pop_single();
    test_full_queue();
    test_fifo_order_single_thread();
    test_wraparound();

    // MP-SC: 2 and 4 producers, 1 consumer
    test_mpsc(2);
    test_mpsc(4);

    // SP-MC: 1 producer, 2 and 4 consumers
    test_spmc(2);
    test_spmc(4);

    // MP-MC: the full concurrent case
    test_mpmc(2, 2);
    test_mpmc(4, 4);

    printf("mpmc_queue_test: %d passed, %d failed\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}
