#include "bench.h"
#include "spsc_queue.h"
#include "mpmc_queue.h"
#include "aba/aba_demo.h"
#include "aba/aba_stack.h"
#include "hazard/hazard_stack.h"
#include "epoch/epoch_stack.h"
#include "rcu/rcu_ptr.h"
#include "freelist/freelist.h"
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

    printf("\n--- ABA tagged-pointer stack ---\n\n");

    // --- Benchmark 9: BuggyStack single-thread roundtrip (8-byte CAS) ---
    //
    // push + pop on a single node, same thread, no contention.
    // Measures the raw cost of the 8-byte compare_exchange_weak on the
    // head_ pointer. CAS always succeeds on the first try (no contention).
    // The node is pushed and immediately popped — one write + one CAS each way.
    {
        foundation::BuggyStack<uint64_t> stack;
        foundation::BuggyStack<uint64_t>::Node node{0, nullptr};
        auto result = bench::run_bench("buggy_stack roundtrip (8B CAS)", 500'000, 10'000,
            [&]() {
                stack.push(&node);
                do_not_optimize(stack.pop());
            });
        bench::print_result(result);
    }

    // --- Benchmark 10: AbaStack single-thread roundtrip (16-byte CAS) ---
    //
    // Same structure as above but uses a 16-byte tagged pointer and cmpxchg16b.
    // The gap between this and the buggy stack roundtrip is the pure cost of
    // the 128-bit CAS vs 64-bit CAS — no contention, no coherence traffic.
    // Expected: cmpxchg16b is ~1.5–3x more expensive than cmpxchg on x86-64
    // because it requires exclusive ownership of a full 16-byte cache line.
    {
        foundation::AbaStack<uint64_t> stack;
        foundation::AbaStack<uint64_t>::Node node{0, nullptr};
        auto result = bench::run_bench("aba_stack  roundtrip (16B CAS)", 500'000, 10'000,
            [&]() {
                stack.push(&node);
                do_not_optimize(stack.pop());
            });
        bench::print_result(result);
    }

    printf("\n--- HazardStack (hazard pointer reclamation) ---\n\n");

    // --- Benchmark 11: HazardStack single-thread push+pop roundtrip ---
    //
    // Each push allocates a Node with new; each pop protects via a hazard
    // pointer, does a 16-byte CAS, then retires the node. The retire triggers
    // a scan once the list exceeds threshold (2 * record_count * slots_per_thread).
    // With 1 thread and 2 slots, threshold = 4. Every 4th pop causes a scan.
    //
    // Compare to AbaStack: AbaStack leaks nodes (no retire), so it avoids
    // both the scan cost and the heap allocation. The gap shows the combined
    // cost of:
    //   (1) new + delete (heap round-trip per push+pop)
    //   (2) retire list management
    //   (3) amortized hazard pointer scan (1/threshold scans per pop)
    {
        foundation::HazardDomain domain;
        foundation::HazardStack<uint64_t> stack(domain);
        uint64_t sink = 0;
        auto result = bench::run_bench("hazard_stack roundtrip (HP + new/del)", 500'000, 10'000,
            [&]() {
                stack.push(1ULL);
                stack.pop(sink);
                do_not_optimize(sink);
            });
        bench::print_result(result);
    }

    printf("\n--- EpochStack (epoch-based reclamation) ---\n\n");

    // --- Benchmark 12: EpochStack single-thread push+pop roundtrip ---
    //
    // Compare with HazardStack (bench 11). Key differences:
    //   - Enter/exit: one seq_cst store each (cheaper than HP's validate loop)
    //   - No per-node hazard publication; any number of nodes safe in one guard
    //   - try_advance() on each retire: O(T) scan of thread records
    //
    // With T=1 thread the scan is trivial (one record). The dominant costs
    // are new+delete (same as HazardStack) and the retire mutex + advance.
    // Expect similar latency to HazardStack; the real EBR win shows at
    // high concurrency with many nodes accessed per critical section.
    {
        foundation::EpochDomain domain;
        foundation::EpochStack<uint64_t> stack(domain);
        uint64_t sink = 0;
        auto result = bench::run_bench("epoch_stack  roundtrip (EBR + new/del)", 500'000, 10'000,
            [&]() {
                stack.push(1ULL);
                stack.pop(sink);
                do_not_optimize(sink);
            });
        bench::print_result(result);
    }

    printf("\n--- RcuPtr (read-copy-update) ---\n\n");

    // --- Benchmark 13: RCU read-side roundtrip (no contention, 1 thread) ---
    //
    // ReadGuard::ReadGuard() + rcu_ptr.get() + ReadGuard::~ReadGuard().
    // This measures the pure reader overhead: two seq_cst fetch_add operations
    // (read_lock + read_unlock) plus one acquire load for get().
    //
    // On x86, seq_cst fetch_add compiles to LOCK XADD — same as acq_rel RMW,
    // about 10–15 ns each. The acquire load is a plain MOV (~1 ns).
    // Expected total: 20–35 ns.
    //
    // Compare with EpochStack enter+exit:
    //   enter = acquire load + relaxed store + seq_cst store (~15 ns)
    //   exit  = release store (~5 ns)
    // EBR's read-side is slightly cheaper (~20 ns) because it stores a fixed
    // epoch value rather than doing an atomic RMW. The RCU trade-off is that
    // readers never touch a shared epoch counter, making RCU more scalable
    // under extremely high reader concurrency.
    {
        foundation::RcuDomain domain;
        foundation::RcuPtr<uint64_t> ptr(domain, new uint64_t{42});

        auto result = bench::run_bench("rcu read-side  (ReadGuard + get, 1T)", 500'000, 10'000,
            [&]() {
                foundation::RcuDomain::ReadGuard guard(domain);
                do_not_optimize(ptr.get());
            });
        bench::print_result(result);

        domain.reclaim_all();
        // ~RcuPtr deletes the current value
    }

    // --- Benchmark 14: RCU write roundtrip (store new, retire old, 1 thread) ---
    //
    // Each iteration: new uint64_t allocation + ptr_.exchange + domain.retire.
    // retire() accumulates into a batch; every kRcuReclaimThreshold (64) items
    // it calls synchronize() + bulk free. The amortized cost per store:
    //   most iterations: exchange + mutex + push_back          (~30 ns)
    //   every 64th:      + synchronize() [fence + scan + fence]
    //
    // With no active readers, synchronize() scans an empty waitlist and returns
    // after just the two fences (~5 ns extra). On x86 MFENCE is ~10–15 ns.
    // Amortized over 64 items: ~(30 + 25/64) ≈ 30 ns/write expected.
    //
    // Note: this benchmark forces reclaim_all() at the end to free the last
    // partial batch. The ~RcuPtr then deletes the final live pointer.
    {
        foundation::RcuDomain domain;
        foundation::RcuPtr<uint64_t> ptr(domain, new uint64_t{0});

        uint64_t counter = 0;
        auto result = bench::run_bench("rcu write      (store+retire, amortized, 1T)", 50'000, 1'000,
            [&]() {
                ptr.store(new uint64_t{++counter});
            });
        bench::print_result(result);

        domain.reclaim_all();
        // ~RcuPtr deletes the final value
    }

    printf("\n--- FreeList (lock-free object pool) ---\n\n");

    // --- Benchmark 15: FreeList acquire+release vs new+delete (1 thread) ---
    //
    // FreeList: acquire() = one 64-bit CAS pop; release() = one 64-bit CAS push.
    // No heap interaction after construction — all slots are pre-allocated.
    // The CAS is uncontended (single thread), so it always succeeds first try.
    // Cost: 2× LOCK CMPXCHG (8B) ≈ 20–30 ns on x86.
    //
    // new/delete: macOS libmalloc uses per-thread magazines for small (<256B)
    // allocations — essentially a thread-local bump pointer. For an 8-byte
    // uint64_t, this is extremely fast (~10 ns). For large/irregular objects
    // the allocator must search size classes and occasionally synchronize.
    //
    // Result: FreeList is SLOWER than malloc for tiny objects on a single thread.
    // The FreeList advantage emerges when:
    //   1. Object construction is expensive (construct once, reuse many times)
    //   2. Multiple threads contend on malloc (magazine exhaustion triggers a lock)
    //   3. Latency determinism matters (malloc has unbounded tail latency;
    //      FreeList acquire/release are O(1) worst-case with bounded CAS retries)
    //   4. Objects are large enough that malloc bookkeeping dominates
    //
    // The struct benchmark below (64B) shows a crossover point.
    {
        foundation::FreeList<uint64_t> pool(1);
        auto result = bench::run_bench("freelist  acquire+release 8B  (1T)", 500'000, 10'000,
            [&]() {
                uint64_t* p = pool.acquire();
                *p = 1;
                do_not_optimize(*p);
                pool.release(p);
            });
        bench::print_result(result);
    }

    {
        auto result = bench::run_bench("new/delete 8B uint64_t        (1T, baseline)", 500'000, 10'000,
            [&]() {
                auto* p = new uint64_t{1};
                do_not_optimize(*p);
                delete p;
            });
        bench::print_result(result);
    }

    // 64-byte struct: malloc bookkeeping is proportionally smaller fraction
    // of total cost, but FreeList still avoids all heap interaction.
    {
        struct alignas(64) Obj64 { std::byte data[64]; };
        foundation::FreeList<Obj64> pool(1);
        auto result = bench::run_bench("freelist  acquire+release 64B (1T)", 500'000, 10'000,
            [&]() {
                Obj64* p = pool.acquire();
                do_not_optimize(p->data[0]);
                pool.release(p);
            });
        bench::print_result(result);
    }

    {
        struct alignas(64) Obj64 { std::byte data[64]; };
        auto result = bench::run_bench("new/delete 64B struct          (1T, baseline)", 500'000, 10'000,
            [&]() {
                auto* p = new Obj64;
                do_not_optimize(p->data[0]);
                delete p;
            });
        bench::print_result(result);
    }

    // --- Benchmark 16: FreeList throughput under contention (N threads) ---
    //
    // Each of N threads spins on acquire → write → release. The gap between
    // 1-thread and N-thread throughput reflects:
    //   (a) CAS retry overhead (failed CAS on contended head_)
    //   (b) MESI invalidation: head_ (8 bytes) bounces between cores on every
    //       successful CAS — ~100–300 ns cross-core round trip on Intel.
    //
    // This is the primary scenario where FreeList beats malloc: malloc under
    // contention must take a lock or use an OS primitive. FreeList never blocks
    // (only spins), giving better worst-case latency at the cost of higher
    // average retry count.
    {
        auto run = [](const char* label, int n_threads) {
            constexpr std::size_t kOps = 2'000'000;
            // 4× slots per thread avoids exhaustion at peak concurrency.
            foundation::FreeList<uint64_t> pool(static_cast<std::size_t>(n_threads) * 4);
            std::atomic<bool> go{false};
            std::atomic<std::size_t> done{0};

            std::vector<std::thread> threads;
            threads.reserve(static_cast<std::size_t>(n_threads));
            for (int t = 0; t < n_threads; ++t) {
                threads.emplace_back([&]() {
                    while (!go.load(std::memory_order_acquire)) {}
                    std::size_t local = 0;
                    while (done.load(std::memory_order_relaxed) < kOps) {
                        uint64_t* p = pool.acquire();
                        if (p) {
                            *p = static_cast<uint64_t>(local++);
                            pool.release(p);
                            done.fetch_add(1, std::memory_order_relaxed);
                        }
                    }
                });
            }

            auto t0 = std::chrono::steady_clock::now();
            go.store(true, std::memory_order_release);
            for (auto& t : threads) t.join();
            auto t1 = std::chrono::steady_clock::now();

            double ns = static_cast<double>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count());
            printf("%-40s  %.1f ns/op  (%.0f M ops/sec)\n",
                   label, ns / kOps, kOps / ns * 1000.0);
        };

        run("freelist throughput 1T", 1);
        run("freelist throughput 2T", 2);
        run("freelist throughput 4T", 4);
    }

    return 0;
}
