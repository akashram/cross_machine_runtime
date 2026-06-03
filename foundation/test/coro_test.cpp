#include "coro/coro.h"

#include <atomic>
#include <cassert>
#include <cstdio>
#include <latch>
#include <stdexcept>
#include <vector>

// ---------------------------------------------------------------------------
// sync_wait: run a root Task synchronously on the calling thread.
//
// Task uses symmetric transfer (await_suspend returns the child's handle), so
// the ENTIRE co_await chain runs inline.  raw_handle().resume() drives the
// chain; when it returns the task is in final_suspend and the result is ready.
// ---------------------------------------------------------------------------
template <typename T>
static T sync_wait(foundation::Task<T> task) {
    task.raw_handle().resume();
    return task.await_resume();
}

static void sync_wait(foundation::Task<void> task) {
    task.raw_handle().resume();
    task.await_resume();
}

// ---------------------------------------------------------------------------
// Test 1: single Task<int> returning a value
// ---------------------------------------------------------------------------
static void test_basic_task() {
    if (sync_wait([]() -> foundation::Task<int> { co_return 42; }()) != 42)
        __builtin_trap();
    printf("PASS  test_basic_task\n");
}

// ---------------------------------------------------------------------------
// Test 2: chained tasks A -> B -> C via symmetric transfer
// ---------------------------------------------------------------------------
static void test_task_chain() {
    auto c = []() -> foundation::Task<int> { co_return 1; };
    auto b = [&]() -> foundation::Task<int> { co_return (co_await c()) + 10; };
    auto chain = [&]() -> foundation::Task<int> { co_return (co_await b()) + 100; };

    if (sync_wait(chain()) != 111) __builtin_trap();
    printf("PASS  test_task_chain\n");
}

// ---------------------------------------------------------------------------
// Test 3: exception in inner task propagates through co_await
// ---------------------------------------------------------------------------
static void test_task_exception() {
    auto inner = []() -> foundation::Task<int> {
        throw std::runtime_error("kaboom");
        co_return 0;
    };
    auto outer = [&]() -> foundation::Task<int> {
        co_return co_await inner();
    };

    int caught = 0;
    try {
        sync_wait(outer());
    } catch (const std::runtime_error& e) {
        caught = (std::string(e.what()) == "kaboom") ? 1 : -1;
    }
    if (caught != 1) { printf("FAIL  test_task_exception\n"); __builtin_trap(); }
    printf("PASS  test_task_exception\n");
}

// ---------------------------------------------------------------------------
// Test 4: Task<void> completes without exception
// ---------------------------------------------------------------------------
static void test_void_task() {
    std::atomic<int> flag{0};
    auto make = [&]() -> foundation::Task<void> {
        flag.store(1, std::memory_order_relaxed);
        co_return;
    };

    sync_wait(make());
    assert(flag.load() == 1);
    printf("PASS  test_void_task\n");
}

// ---------------------------------------------------------------------------
// Test 5: task destroyed before started -- no crash
// ---------------------------------------------------------------------------
static void test_task_destroyed_before_start() {
    auto make = []() -> foundation::Task<int> { co_return 99; };
    { auto t = make(); /* destroyed without resume -- handle_.destroy() */ }
    printf("PASS  test_task_destroyed_before_start\n");
}

// ---------------------------------------------------------------------------
// Test 6: AwaitableEvent already set before co_await -> no suspension
// ---------------------------------------------------------------------------
static void test_event_preset() {
    foundation::AwaitableEvent ev;
    ev.set();

    std::atomic<int> reached{0};
    auto coro = [&]() -> foundation::Task<void> {
        co_await ev;
        reached.store(1, std::memory_order_relaxed);
    };

    sync_wait(coro());
    assert(reached.load() == 1);
    printf("PASS  test_event_preset\n");
}

// ---------------------------------------------------------------------------
// Test 7: set() from an external thread after co_await suspends
// ---------------------------------------------------------------------------
static void test_event_external_set() {
    foundation::WorkStealingPool pool(2);
    foundation::AwaitableEvent ev;
    std::atomic<int> result{0};
    std::latch done(1);

    auto make_waiter = [&]() -> foundation::Task<void> {
        co_await ev.on(pool);
        result.store(42, std::memory_order_relaxed);
        done.count_down();
    };

    auto waiter = make_waiter();
    schedule(pool, waiter.raw_handle());

    // Signal from the main thread; set() sees the stored handle and schedules it.
    ev.set(&pool);

    done.wait();
    // pool.wait() ensures the pool lambda ([h]{ h.resume(); }) submitted by
    // ev.set() has returned.  h.resume() returns only after the coroutine
    // reaches final_suspend (after return_void()).  Without this, ~Task()
    // could call handle_.destroy() while the worker is still in return_void().
    pool.wait();
    assert(result.load() == 42);
    printf("PASS  test_event_external_set\n");
}

// ---------------------------------------------------------------------------
// Test 8: double set() does not resume twice or crash
// ---------------------------------------------------------------------------
static void test_event_double_set() {
    foundation::AwaitableEvent ev;
    ev.set();
    ev.set();  // second call: old == kSet, nothing to resume
    printf("PASS  test_event_double_set\n");
}

// ---------------------------------------------------------------------------
// Test 9: AwaitableMutex basic acquire + release with RAII Guard
//
// co_await mu.lock() returns a Guard; capturing it keeps the mutex locked
// until the Guard goes out of scope.  NOT capturing it would unlock immediately
// (Guard destructs at the semicolon) -- always capture with `auto g = ...`.
// ---------------------------------------------------------------------------
static void test_mutex_basic() {
    foundation::AwaitableMutex mu;
    int counter = 0;

    auto coro = [&]() -> foundation::Task<void> {
        {
            auto g = co_await mu.lock();
            ++counter;
        }  // g.~Guard() -> mu.unlock()
        {
            auto g = co_await mu.lock();
            ++counter;
        }
    };

    sync_wait(coro());
    assert(counter == 2);
    printf("PASS  test_mutex_basic\n");
}

// ---------------------------------------------------------------------------
// Test 10: Guard RAII -- lock survives a nested co_await
// ---------------------------------------------------------------------------
static void test_mutex_guard_survives_await() {
    foundation::AwaitableMutex mu;
    std::atomic<int> step{0};

    auto inner = [&]() -> foundation::Task<void> {
        step.store(1, std::memory_order_relaxed);
        co_return;
    };

    auto coro = [&]() -> foundation::Task<void> {
        auto g = co_await mu.lock();
        co_await inner();   // guard still live across this co_await
        step.store(2, std::memory_order_relaxed);
        // g destructs here -> unlock
    };

    sync_wait(coro());
    assert(step.load() == 2);
    // Verify mutex is now free by locking again.
    {
        auto coro2 = [&]() -> foundation::Task<void> {
            auto g = co_await mu.lock();
            step.store(3, std::memory_order_relaxed);
        };
        sync_wait(coro2());
    }
    assert(step.load() == 3);
    printf("PASS  test_mutex_guard_survives_await\n");
}

// ---------------------------------------------------------------------------
// Test 11: N coroutines on pool increment shared counter with mutex
//
// counter is a plain int -- the mutex provides all synchronization.
// TSan must see acquire/release through the mutex CAS pairs and report no race.
//
// Coroutines may suspend (waiting for mutex) and be resumed on different pool
// workers; the Task objects are kept alive in `tasks` for the full duration.
// ---------------------------------------------------------------------------
static void test_mutex_stress_pool() {
    foundation::WorkStealingPool pool(4);
    foundation::AwaitableMutex mu;
    int counter = 0;

    constexpr int N     = 8;
    constexpr int ITERS = 200;
    std::latch done(N);

    auto make_worker = [&]() -> foundation::Task<void> {
        for (int i = 0; i < ITERS; ++i) {
            auto g = co_await mu.lock(&pool);
            ++counter;
        }  // Guard unlocks with pool=&pool -> schedule next waiter on pool
        done.count_down();
    };

    std::vector<foundation::Task<void>> tasks;
    tasks.reserve(N);
    for (std::size_t i = 0; i < N; ++i)
        tasks.push_back(make_worker());

    for (std::size_t i = 0; i < N; ++i)
        schedule(pool, tasks[i].raw_handle());

    done.wait();
    pool.wait();  // all h.resume() lambdas returned → frames in final_suspend → safe to ~Task()
    assert(counter == N * ITERS);
    printf("PASS  test_mutex_stress_pool  (N=%d, ITERS=%d, counter=%d)\n",
           N, ITERS, N * ITERS);
}

// ---------------------------------------------------------------------------
// Test 12: pool integration -- N root tasks fan out, latch tracks completion
// ---------------------------------------------------------------------------
static void test_pool_fanout() {
    foundation::WorkStealingPool pool(4);
    std::atomic<int> total{0};

    constexpr int N     = 16;
    constexpr int ITERS = 500;
    std::latch done(N);

    auto make_task = [&]() -> foundation::Task<void> {
        for (int i = 0; i < ITERS; ++i)
            total.fetch_add(1, std::memory_order_relaxed);
        done.count_down();
        co_return;
    };

    std::vector<foundation::Task<void>> tasks;
    tasks.reserve(N);
    for (std::size_t i = 0; i < N; ++i)
        tasks.push_back(make_task());

    for (std::size_t i = 0; i < N; ++i)
        schedule(pool, tasks[i].raw_handle());

    done.wait();
    pool.wait();  // frames in final_suspend before tasks vector destructs
    assert(total.load() == N * ITERS);
    printf("PASS  test_pool_fanout  (N=%d, ITERS=%d, total=%d)\n",
           N, ITERS, N * ITERS);
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main() {
    test_basic_task();
    test_task_chain();
    test_task_exception();
    test_void_task();
    test_task_destroyed_before_start();
    test_event_preset();
    test_event_external_set();
    test_event_double_set();
    test_mutex_basic();
    test_mutex_guard_survives_await();
    test_mutex_stress_pool();
    test_pool_fanout();
    return 0;
}
