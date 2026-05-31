#include "spsc_queue.h"

#include <atomic>
#include <cassert>
#include <cstdio>
#include <thread>
#include <vector>

using foundation::SpscQueue;

// ---------------------------------------------------------------------------
// Minimal test harness
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
    SpscQueue<int, 4> q;
    int val;
    CHECK(q.empty());
    CHECK(!q.pop(val));
    CHECK(q.size_approx() == 0);
}

static void test_capacity() {
    // N=4 → capacity=3
    SpscQueue<int, 4> q;
    CHECK(q.capacity() == 3);
}

static void test_push_pop_single() {
    SpscQueue<int, 4> q;
    CHECK(q.push(42));
    CHECK(!q.empty());
    int val = 0;
    CHECK(q.pop(val));
    CHECK(val == 42);
    CHECK(q.empty());
}

static void test_full_queue() {
    SpscQueue<int, 4> q;  // capacity = 3
    CHECK(q.push(1));
    CHECK(q.push(2));
    CHECK(q.push(3));
    CHECK(!q.push(4));  // full
    CHECK(q.size_approx() == 3);
}

static void test_fifo_order() {
    SpscQueue<int, 8> q;
    for (int i = 0; i < 7; ++i) CHECK(q.push(i));
    for (int i = 0; i < 7; ++i) {
        int val = -1;
        CHECK(q.pop(val));
        CHECK(val == i);
    }
    CHECK(q.empty());
}

static void test_wraparound() {
    // Push/pop enough items to wrap the ring buffer indices at least twice.
    SpscQueue<uint32_t, 8> q;  // capacity = 7
    constexpr int kRounds = 20;
    for (int r = 0; r < kRounds; ++r) {
        for (uint32_t i = 0; i < 7; ++i) CHECK(q.push(i));
        for (uint32_t i = 0; i < 7; ++i) {
            uint32_t val = 0xDEAD;
            CHECK(q.pop(val));
            CHECK(val == i);
        }
    }
}

static void test_interleaved_push_pop() {
    SpscQueue<int, 4> q;
    int val;
    CHECK(q.push(1));
    CHECK(q.pop(val)); CHECK(val == 1);
    CHECK(q.push(2));
    CHECK(q.push(3));
    CHECK(q.pop(val)); CHECK(val == 2);
    CHECK(q.push(4));
    CHECK(q.pop(val)); CHECK(val == 3);
    CHECK(q.pop(val)); CHECK(val == 4);
    CHECK(!q.pop(val));
}

// ---------------------------------------------------------------------------
// Multi-threaded correctness
//
// This test is specifically useful under TSan (tsan preset). If the memory
// ordering in push/pop were wrong (e.g. relaxed instead of acquire/release),
// TSan would flag a data race on the slot access.
//
// The test verifies two properties:
//   1. No items are lost — consumer receives exactly kItems items.
//   2. FIFO order is preserved — items arrive in the same order they were sent.
// ---------------------------------------------------------------------------

static void test_threaded_correctness() {
    constexpr std::size_t kItems = 100'000;
    SpscQueue<uint64_t, 4096> q;

    std::atomic<bool> consumer_done{false};
    bool order_ok = true;

    std::thread producer([&]() {
        for (uint64_t i = 0; i < kItems; ++i) {
            while (!q.push(i)) {}
        }
    });

    std::thread consumer([&]() {
        uint64_t expected = 0;
        while (expected < kItems) {
            uint64_t val;
            if (q.pop(val)) {
                if (val != expected) order_ok = false;
                ++expected;
            }
        }
        consumer_done.store(true, std::memory_order_release);
    });

    producer.join();
    consumer.join();

    CHECK(consumer_done.load(std::memory_order_acquire));
    CHECK(order_ok);
}

// ---------------------------------------------------------------------------
// Static interface checks (compile-time)
// ---------------------------------------------------------------------------

static void test_static_properties() {
    // Non-copyable
    static_assert(!std::is_copy_constructible_v<SpscQueue<int, 4>>);
    static_assert(!std::is_copy_assignable_v<SpscQueue<int, 4>>);

    // capacity() is constexpr
    static_assert(SpscQueue<int, 8>::capacity() == 7);
    static_assert(SpscQueue<int, 1024>::capacity() == 1023);

    CHECK(true);  // reached here means all static_asserts passed
}

// ---------------------------------------------------------------------------

int main() {
    test_empty_on_construction();
    test_capacity();
    test_push_pop_single();
    test_full_queue();
    test_fifo_order();
    test_wraparound();
    test_interleaved_push_pop();
    test_threaded_correctness();
    test_static_properties();

    printf("spsc_queue_test: %d passed, %d failed\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}
