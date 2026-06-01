#include "hazard/hazard_ptr.h"
#include "hazard/hazard_stack.h"

#include <atomic>
#include <barrier>
#include <cassert>
#include <cstdio>
#include <thread>
#include <vector>

// ---------------------------------------------------------------------------
// Tracked<T>: a wrapper that counts live instances.
// Used to verify that hazard pointer reclamation actually frees memory.
// ---------------------------------------------------------------------------
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
// Test 1: single-thread retire + scan frees memory
// ---------------------------------------------------------------------------
static void test_retire_frees_memory() {
    foundation::HazardDomain domain;

    {
        using T = Tracked<int>;
        // Push 10 items via HazardStack (each push allocates a Node<T>).
        foundation::HazardStack<T> stack(domain);
        for (int i = 0; i < 10; ++i) stack.push(T{i});

        assert(T::live.load() == 10);

        // Pop all — each pop() retires the node.
        // val is scoped so its destructor runs before the live-count assert.
        int count = 0;
        {
            T val{0};
            while (stack.pop(val)) ++count;
        }
        assert(count == 10);

        // Force a scan to ensure pending retires are resolved.
        domain.scan();
        assert(domain.pending_count() == 0);
        assert(T::live.load() == 0);
    }

    printf("PASS  test_retire_frees_memory\n");
}

// ---------------------------------------------------------------------------
// Test 2: LIFO ordering (single thread)
// ---------------------------------------------------------------------------
static void test_lifo_ordering() {
    foundation::HazardDomain domain;
    foundation::HazardStack<int> stack(domain);

    for (int i = 0; i < 8; ++i) stack.push(i);

    for (int i = 7; i >= 0; --i) {
        int val = -1;
        bool ok = stack.pop(val);
        assert(ok);
        assert(val == i);
    }
    int dummy;
    assert(!stack.pop(dummy));
    assert(stack.empty());
    domain.scan();
    assert(domain.pending_count() == 0);

    printf("PASS  test_lifo_ordering\n");
}

// ---------------------------------------------------------------------------
// Test 3: concurrent push-then-pop stress with live-count verification
//
// Phase 1: all threads push kNodesPerThread nodes.
// Phase 2: all threads race to pop until empty.
// After joining: verify live_count == 0 (all nodes freed).
// ---------------------------------------------------------------------------
static void test_concurrent_reclamation() {
    constexpr std::size_t kThreads        = 4;
    constexpr std::size_t kNodesPerThread = 10'000;
    constexpr std::size_t kTotal          = kThreads * kNodesPerThread;

    using T = Tracked<int>;
    assert(T::live.load() == 0);

    foundation::HazardDomain domain;
    foundation::HazardStack<T> stack(domain);

    std::atomic<int> pop_count{0};
    std::barrier sync(static_cast<std::ptrdiff_t>(kThreads));

    std::vector<std::thread> threads;
    threads.reserve(kThreads);

    for (std::size_t t = 0; t < kThreads; ++t) {
        threads.emplace_back([&, t]() {
            // Phase 1: push
            for (std::size_t i = 0; i < kNodesPerThread; ++i)
                stack.push(T{static_cast<int>(t * kNodesPerThread + i)});

            sync.arrive_and_wait();

            // Phase 2: pop until empty
            {
                T val{0};
                while (stack.pop(val))
                    pop_count.fetch_add(1, std::memory_order_relaxed);
            }
        });
    }

    for (auto& t : threads) t.join();

    // All threads exited — their thread_local ThreadState destructors ran,
    // calling scan() one final time each. Force another scan from this thread.
    domain.scan();

    assert(pop_count.load() == static_cast<int>(kTotal));
    assert(stack.empty());
    // Tracked<int> live count must be 0: all nodes freed via retire().
    assert(T::live.load() == 0);

    printf("PASS  test_concurrent_reclamation  (%zu nodes, all freed)\n", kTotal);
}

// ---------------------------------------------------------------------------
// Test 4: concurrent push AND pop (hazard pointers protect cross-thread accesses)
//
// Unlike the previous test, pushes and pops happen simultaneously on all
// threads. This is the case that requires hazard pointers — without them,
// a popped node could be freed while another thread is reading ->next.
// ASan will catch any use-after-free here.
// ---------------------------------------------------------------------------
static void test_concurrent_push_pop_mixed() {
    constexpr std::size_t kThreads  = 4;
    constexpr std::size_t kPerThread = 10'000;

    using T = Tracked<int>;
    assert(T::live.load() == 0);

    foundation::HazardDomain domain;
    foundation::HazardStack<T> stack(domain);
    std::atomic<int> pop_count{0};
    std::atomic<bool> done{false};

    // Producer threads: push kPerThread items each.
    std::vector<std::thread> producers;
    producers.reserve(kThreads / 2);
    for (std::size_t t = 0; t < kThreads / 2; ++t) {
        producers.emplace_back([&, t]() {
            for (std::size_t i = 0; i < kPerThread; ++i)
                stack.push(T{static_cast<int>(t * kPerThread + i)});
        });
    }

    // Consumer threads: pop until producers are done and stack is empty.
    std::vector<std::thread> consumers;
    consumers.reserve(kThreads / 2);
    for (std::size_t t = 0; t < kThreads / 2; ++t) {
        consumers.emplace_back([&]() {
            // val is scoped so its destructor runs before join (and before
            // the live-count assert).
            T val{0};
            while (!done.load(std::memory_order_acquire) || !stack.empty()) {
                if (stack.pop(val))
                    pop_count.fetch_add(1, std::memory_order_relaxed);
            }
        });
    }

    for (auto& p : producers) p.join();
    done.store(true, std::memory_order_release);
    // Drain anything pushed after consumers last checked done.
    {
        T val{0};
        while (stack.pop(val))
            pop_count.fetch_add(1, std::memory_order_relaxed);
    }
    for (auto& c : consumers) c.join();

    domain.scan();

    const std::size_t expected = (kThreads / 2) * kPerThread;
    assert(pop_count.load() == static_cast<int>(expected));
    assert(stack.empty());
    assert(T::live.load() == 0);

    printf("PASS  test_concurrent_push_pop_mixed  (%zu nodes, all freed)\n", expected);
}

// ---------------------------------------------------------------------------
// Test 5: pending_count drops to 0 after scan()
// ---------------------------------------------------------------------------
static void test_pending_count() {
    foundation::HazardDomain domain;
    foundation::HazardStack<int> stack(domain);

    for (int i = 0; i < 5; ++i) stack.push(i);

    {
        int val = 0;
        while (stack.pop(val)) {}
    }

    domain.scan();
    assert(domain.pending_count() == 0);

    printf("PASS  test_pending_count\n");
}

int main() {
    test_retire_frees_memory();
    test_lifo_ordering();
    test_concurrent_reclamation();
    test_concurrent_push_pop_mixed();
    test_pending_count();
    return 0;
}
