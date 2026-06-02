#include "rcu/rcu_domain.h"
#include "rcu/rcu_ptr.h"

#include <atomic>
#include <barrier>
#include <cassert>
#include <chrono>
#include <cstdio>
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
// Test 1: retire + reclaim_all frees memory (no readers)
// ---------------------------------------------------------------------------
static void test_retire_reclaim_basic() {
    using T = Tracked<int>;
    assert(T::live.load() == 0);

    foundation::RcuDomain domain;
    T* p1 = new T{1};
    T* p2 = new T{2};
    T* p3 = new T{3};
    assert(T::live.load() == 3);

    domain.retire(p1);
    domain.retire(p2);
    domain.retire(p3);
    assert(domain.pending_count() == 3);

    domain.reclaim_all();

    assert(domain.pending_count() == 0);
    assert(T::live.load() == 0);
    printf("PASS  test_retire_reclaim_basic\n");
}

// ---------------------------------------------------------------------------
// Test 2: active ReadGuard blocks reclaim_all until the guard exits.
//
// The reader holds a guard for 20 ms. The main thread calls reclaim_all()
// after confirming the reader is inside the guard. reclaim_all() must not
// return until after the guard releases.
// ---------------------------------------------------------------------------
static void test_guard_blocks_reclaim() {
    using T = Tracked<int>;
    assert(T::live.load() == 0);

    foundation::RcuDomain domain;
    T* p = new T{42};
    assert(T::live.load() == 1);

    std::atomic<bool> reader_in{false};

    std::thread reader([&]() {
        foundation::RcuDomain::ReadGuard guard(domain);
        reader_in.store(true, std::memory_order_release);
        // Hold the guard long enough for the main thread to enter synchronize().
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
    });

    // Wait until reader is inside the guard.
    while (!reader_in.load(std::memory_order_acquire))
        std::this_thread::yield();

    domain.retire(p);  // reader is active — synchronize() will block on them

    // reclaim_all() → synchronize() → spins until reader's ctr changes → frees p
    domain.reclaim_all();

    reader.join();

    assert(T::live.load() == 0);
    assert(domain.pending_count() == 0);
    printf("PASS  test_guard_blocks_reclaim\n");
}

// ---------------------------------------------------------------------------
// Test 3: RcuPtr store/get correctness (single thread)
// ---------------------------------------------------------------------------
static void test_rcu_ptr_basic() {
    using T = Tracked<int>;
    assert(T::live.load() == 0);

    foundation::RcuDomain domain;

    {
        foundation::RcuPtr<T> ptr(domain, new T{0});
        assert(T::live.load() == 1);

        {
            foundation::RcuDomain::ReadGuard guard(domain);
            assert(ptr.get()->data == 0);
        }

        ptr.store(new T{10});
        // Old value (0) is now in pending_. New value (10) is live.
        assert(T::live.load() >= 1);  // new value definitely alive

        {
            foundation::RcuDomain::ReadGuard guard(domain);
            assert(ptr.get()->data == 10);
        }

        ptr.store(new T{20});
        {
            foundation::RcuDomain::ReadGuard guard(domain);
            assert(ptr.get()->data == 20);
        }

        domain.reclaim_all();
        assert(T::live.load() == 1);  // only the current value remains

    }  // ~RcuPtr deletes current value (20)

    assert(T::live.load() == 0);
    printf("PASS  test_rcu_ptr_basic\n");
}

// ---------------------------------------------------------------------------
// Test 4: concurrent readers + writer (primary TSan safety check)
//
// N reader threads continuously read via ReadGuard. 1 writer thread stores
// new values. TSan must detect no use-after-free or data race.
// After all threads join and reclaim_all is called, all objects are freed.
// ---------------------------------------------------------------------------
static void test_concurrent_readers_writer() {
    constexpr std::size_t kReaders   = 4;
    constexpr std::size_t kWriteOps  = 2'000;

    using T = Tracked<int>;
    assert(T::live.load() == 0);

    {
        foundation::RcuDomain domain;
        foundation::RcuPtr<T> ptr(domain, new T{0});

        std::atomic<bool> done{false};
        std::barrier sync(static_cast<std::ptrdiff_t>(kReaders + 1));

        std::vector<std::thread> readers;
        readers.reserve(kReaders);
        for (std::size_t i = 0; i < kReaders; ++i) {
            readers.emplace_back([&]() {
                sync.arrive_and_wait();
                while (!done.load(std::memory_order_acquire)) {
                    foundation::RcuDomain::ReadGuard guard(domain);
                    T* p = ptr.get();
                    if (p) {
                        // Access the object while holding the guard.
                        // TSan will catch any use-after-free here.
                        [[maybe_unused]] volatile int x = p->data;
                    }
                }
            });
        }

        sync.arrive_and_wait();  // release all readers

        for (std::size_t i = 0; i < kWriteOps; ++i)
            ptr.store(new T{static_cast<int>(i + 1)});

        done.store(true, std::memory_order_release);
        for (auto& t : readers) t.join();

        domain.reclaim_all();
        assert(T::live.load() == 1);  // only the current value remains

    }  // ~RcuPtr deletes the final value

    assert(T::live.load() == 0);
    printf("PASS  test_concurrent_readers_writer  (%zu writers, %zu readers)\n",
           kWriteOps, kReaders);
}

// ---------------------------------------------------------------------------
// Test 5: automatic batch reclaim at threshold
//
// Retire kRcuReclaimThreshold objects one by one without calling reclaim_all()
// explicitly. The last retire must trigger internal reclamation.
// ---------------------------------------------------------------------------
static void test_auto_reclaim_at_threshold() {
    using T = Tracked<int>;
    assert(T::live.load() == 0);

    foundation::RcuDomain domain;

    for (std::size_t i = 0; i < foundation::kRcuReclaimThreshold; ++i)
        domain.retire(new T{static_cast<int>(i)});

    // The last retire should have triggered reclaim_all() internally,
    // bringing pending_count() back to 0 and freeing all objects.
    assert(domain.pending_count() == 0);
    assert(T::live.load() == 0);
    printf("PASS  test_auto_reclaim_at_threshold  (threshold=%zu)\n",
           foundation::kRcuReclaimThreshold);
}

int main() {
    test_retire_reclaim_basic();
    test_guard_blocks_reclaim();
    test_rcu_ptr_basic();
    test_concurrent_readers_writer();
    test_auto_reclaim_at_threshold();
    return 0;
}
