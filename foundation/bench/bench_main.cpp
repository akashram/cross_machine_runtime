#include "bench.h"
#include "spsc_queue.h"
#include "mpmc_queue.h"
#include <atomic>
#include <cstdio>
#include <thread>
#include <chrono>

// Prevents the compiler from optimizing away a value entirely.
// Without this, the optimizer sees that the result of a computation is never
// used and deletes the computation — giving you a benchmark of "nothing".
// Writing to a volatile variable forces the value to be materialized.
// This is the standard low-tech alternative to compiler-specific escape
// intrinsics like __asm__ __volatile__("" : "+r"(x)).
template<typename T>
inline void do_not_optimize(T const& val) {
    volatile T sink = val;
    (void)sink;
}

int main() {
    // Print the calibrated frequency so we can sanity-check it.
    // On a modern Intel/Apple Silicon Mac at ~3GHz you should see ~3.0 ticks/ns.
    // If this prints something wildly off (e.g. 0.1 or 300), the calibration
    // or the TSC read is broken.
    printf("TSC frequency: %.3f ticks/ns\n\n", bench::tsc_ticks_per_ns());

    // --- Benchmark 1: harness overhead floor ---
    //
    // The lambda does nothing. The measured time is the cost of two RDTSCP
    // instructions plus the call overhead. This is the noise floor — any
    // real benchmark result should be larger than this. On a 3GHz machine
    // expect roughly 5–15 ns.
    auto result_nop = bench::run_bench("nop (harness floor)", 100'000, 1'000,
        []() {
            // nothing
        });
    bench::print_result(result_nop);

    // --- Benchmark 2: single integer addition ---
    //
    // One ALU operation. The actual add takes <1ns, so what you're measuring
    // here is still dominated by harness overhead. The interesting thing is
    // comparing it to the nop above — it should be indistinguishable, because
    // the add gets folded into the surrounding loop by the CPU's out-of-order
    // engine. do_not_optimize forces the result to be written to memory,
    // which adds a store but prevents the computation from disappearing.
    int x = 1;
    auto result_add = bench::run_bench("integer add + store", 100'000, 1'000,
        [&x]() {
            do_not_optimize(x + 1);
        });
    bench::print_result(result_add);

    // --- Benchmark 3: sleep_for(1ns) ---
    //
    // sleep_for is NOT a nanosecond-precision sleep on any real OS. On macOS,
    // the scheduler granularity is ~1ms (1,000,000 ns). This benchmark is here
    // to verify that the harness can faithfully measure *long* durations — if
    // the mean comes back near 1,000,000 ns, the timer is working correctly at
    // scale. The p99 will be much higher due to scheduler jitter.
    auto result_sleep = bench::run_bench("sleep_for(1ns) [expect ~10-1000us on macOS]", 100, 5,
        []() {
            std::this_thread::sleep_for(std::chrono::nanoseconds(1));
        });
    bench::print_result(result_sleep);

    // --- Benchmark 4: SPSC single-thread roundtrip ---
    //
    // Push one uint64_t then immediately pop it — same thread. This is the
    // absolute floor for SPSC cost: no cross-core coherence traffic, no
    // scheduling jitter. On x86, acquire/release compile to plain MOVs, so
    // you're measuring two atomic loads, two atomic stores, and a cache-line
    // read/write to the slot array.
    //
    // Expected: roughly 5–20 ns on a 3 GHz machine. If it reads higher, check
    // whether the queue spans multiple cache lines (head_/tail_ false sharing).
    {
        foundation::SpscQueue<uint64_t, 4096> q;
        uint64_t sink = 0;
        auto result = bench::run_bench("spsc roundtrip (1 thread)", 500'000, 10'000,
            [&]() {
                q.push(1);
                q.pop(sink);
                do_not_optimize(sink);
            });
        bench::print_result(result);
    }

    // --- Benchmark 5: SPSC producer-consumer throughput (2 threads) ---
    //
    // Producer spins on push, consumer spins on pop. Measures steady-state
    // throughput: items transferred per nanosecond when both threads are hot.
    //
    // This is the inter-core case. The slot data and the tail_ index travel
    // across the coherence fabric between producer core and consumer core.
    // Expected: 100–500 ns/item depending on LLC latency and core topology.
    // A large gap vs. the single-thread roundtrip above is normal — it reflects
    // the cost of cache-line ownership transfer between cores (MESI state
    // transitions from Modified on producer to Exclusive/Shared on consumer).
    {
        constexpr std::size_t kItems = 2'000'000;
        foundation::SpscQueue<uint64_t, 4096> q;
        std::atomic<bool> go{false};

        std::thread producer([&]() {
            while (!go.load(std::memory_order_acquire)) {}
            for (uint64_t i = 0; i < kItems; ++i) {
                while (!q.push(i)) {}
            }
        });

        std::thread consumer([&]() {
            while (!go.load(std::memory_order_acquire)) {}
            std::size_t received = 0;
            while (received < kItems) {
                uint64_t val;
                if (q.pop(val)) {
                    do_not_optimize(val);
                    ++received;
                }
            }
        });

        auto t0 = std::chrono::steady_clock::now();
        go.store(true, std::memory_order_release);
        producer.join();
        consumer.join();
        auto t1 = std::chrono::steady_clock::now();

        double total_ns = static_cast<double>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count());
        double ns_per_item = total_ns / static_cast<double>(kItems);
        double m_items_per_sec = static_cast<double>(kItems) / total_ns * 1000.0;
        printf("%-30s  %.1f ns/item  (%.0f M items/sec)\n",
               "spsc throughput (2 threads)", ns_per_item, m_items_per_sec);
    }

    // --- Benchmark 6: SPSC ping-pong latency (2 threads, 2 queues) ---
    //
    // Thread A pushes to q_fwd, Thread B pops from q_fwd and pushes to q_bck,
    // Thread A pops from q_bck. One round trip = one item traveling A→B→A.
    // The measured latency is the full cross-core round-trip time for a single
    // cache line — a proxy for the inter-core communication latency on this
    // machine. Expect 100–400 ns on modern Intel depending on core placement.
    {
        constexpr std::size_t kPings = 200'000;
        constexpr std::size_t kWarmup = 1'000;
        foundation::SpscQueue<uint64_t, 2> q_fwd;
        foundation::SpscQueue<uint64_t, 2> q_bck;

        std::thread relay([&]() {
            for (std::size_t i = 0; i < kPings + kWarmup; ++i) {
                uint64_t val;
                while (!q_fwd.pop(val)) {}
                while (!q_bck.push(val)) {}
            }
        });

        // Warmup
        for (std::size_t i = 0; i < kWarmup; ++i) {
            while (!q_fwd.push(0)) {}
            uint64_t val;
            while (!q_bck.pop(val)) {}
        }

        std::vector<double> samples;
        samples.reserve(kPings);
        for (std::size_t i = 0; i < kPings; ++i) {
            uint64_t t0 = bench::tsc_now();
            while (!q_fwd.push(static_cast<uint64_t>(i))) {}
            uint64_t val;
            while (!q_bck.pop(val)) {}
            uint64_t t1 = bench::tsc_now();
            samples.push_back(bench::tsc_to_ns(t1 - t0));
        }

        relay.join();

        auto result = bench::compute_stats("spsc ping-pong RTT (2 threads)", samples);
        bench::print_result(result);
    }

    printf("\n--- MpmcQueue ---\n\n");

    // --- Benchmark 7: MPMC single-thread roundtrip ---
    //
    // Same-thread push+pop. Isolates the per-operation cost of the CAS loop
    // and sequence number check. On a single thread there is no contention on
    // tail_/head_, so the CAS always succeeds on the first attempt. The gap
    // between this and the SPSC roundtrip above is the raw overhead of CAS +
    // sequence load vs. plain atomic store.
    {
        foundation::MpmcQueue<uint64_t, 4096> q;
        uint64_t sink = 0;
        auto result = bench::run_bench("mpmc roundtrip (1 thread)", 500'000, 10'000,
            [&]() {
                q.push(1);
                q.pop(sink);
                do_not_optimize(sink);
            });
        bench::print_result(result);
    }

    // --- Benchmark 8: MPMC 1P-1C throughput ---
    //
    // Direct comparison with the SPSC throughput benchmark. Same structure,
    // same item count. The difference shows MPMC's overhead in the 1P-1C case
    // where SPSC is theoretically optimal. Typical expectation: MPMC is
    // 1.5–3x slower than SPSC here due to the CAS on tail_/head_.
    {
        auto run_mpmc_throughput = [](const char* label, int n_prod, int n_cons) {
            constexpr std::size_t kItems = 2'000'000;
            foundation::MpmcQueue<uint64_t, 4096> q;
            std::atomic<bool> go{false};
            std::atomic<uint64_t> push_idx{0};
            std::atomic<std::size_t> pop_count{0};

            std::vector<std::thread> producers;
            for (int i = 0; i < n_prod; ++i) {
                producers.emplace_back([&]() {
                    while (!go.load(std::memory_order_acquire)) {}
                    for (;;) {
                        uint64_t val = push_idx.fetch_add(1, std::memory_order_relaxed);
                        if (val >= kItems) break;
                        while (!q.push(val)) {}
                    }
                });
            }

            std::vector<std::thread> consumers;
            for (int i = 0; i < n_cons; ++i) {
                consumers.emplace_back([&]() {
                    while (!go.load(std::memory_order_acquire)) {}
                    while (pop_count.load(std::memory_order_relaxed) < kItems) {
                        uint64_t val;
                        if (q.pop(val)) {
                            do_not_optimize(val);
                            pop_count.fetch_add(1, std::memory_order_relaxed);
                        }
                    }
                });
            }

            auto t0 = std::chrono::steady_clock::now();
            go.store(true, std::memory_order_release);
            for (auto& t : producers) t.join();
            for (auto& t : consumers) t.join();
            auto t1 = std::chrono::steady_clock::now();

            double total_ns = static_cast<double>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count());
            double ns_per_item = total_ns / static_cast<double>(kItems);
            double m_items_per_sec = static_cast<double>(kItems) / total_ns * 1000.0;
            printf("%-36s  %.1f ns/item  (%.0f M items/sec)\n",
                   label, ns_per_item, m_items_per_sec);
        };

        run_mpmc_throughput("mpmc 1P-1C throughput", 1, 1);
        run_mpmc_throughput("mpmc 2P-2C throughput", 2, 2);
        run_mpmc_throughput("mpmc 4P-4C throughput", 4, 4);
    }

    return 0;
}
