#include "bench.h"
#include "arena/arena.h"
#include "numa/numa.h"
#include "perf/perf.h"
#include "spsc_queue.h"
#include "mpmc_queue.h"
#include "aba/aba_demo.h"
#include "aba/aba_stack.h"
#include "hazard/hazard_stack.h"
#include "epoch/epoch_stack.h"
#include "rcu/rcu_ptr.h"
#include "freelist/freelist.h"
#include "msqueue/ms_queue.h"
#include "chase_lev/chase_lev.h"
#include "ws_pool/ws_pool.h"
#include <atomic>
#include <cstdio>
#include <numeric>
#include <thread>
#include <vector>
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

    printf("\n--- MsQueue (Michael-Scott lock-free FIFO) ---\n\n");

    // --- Benchmark 17: MsQueue single-thread enqueue+dequeue roundtrip ---
    //
    // Each iteration: new Node (malloc) + two CAS operations (enqueue) +
    // two CAS operations (dequeue) + HP scan (amortized) + delete.
    //
    // Compare with MpmcQueue (pre-allocated ring buffer):
    //   MpmcQueue: no allocation, one CAS each way, ~21 ns (bench 7)
    //   MsQueue:   one malloc + one delete + more CAS overhead + HP
    //
    // The MS queue pays for heap allocation per element. Its advantage is
    // unbounded capacity (grows dynamically) vs. the ring buffer's fixed size.
    // In practice, MS queue is ~5–10x slower than MPMC for the single-thread
    // case because malloc/free dominates.
    {
        foundation::MsQueue<uint64_t> q;
        uint64_t sink = 0;
        auto result = bench::run_bench("ms_queue  enq+deq (1T, new+delete per elem)", 200'000, 5'000,
            [&]() {
                q.enqueue(1ULL);
                q.dequeue(sink);
                do_not_optimize(sink);
            });
        bench::print_result(result);
        q.drain();
    }

    // --- Benchmark 18: MsQueue throughput 1P-1C and 2P-2C ---
    //
    // Cross-thread case: producer and consumer on different cores. The node
    // (cache line) must travel from producer to consumer — same coherence cost
    // as MPMC, but with the additional malloc/free overhead on each side.
    //
    // The HP scan is amortized: with one record (2 slots), scan threshold =
    // 2 * 1 * 2 = 4. Every 4th dequeue triggers a full scan. In the 1P-1C
    // case, this is cheap (1 record to scan). In the 4P-4C case, 8 records.
    {
        auto run_ms = [](const char* label, int n_prod, int n_cons) {
            constexpr std::size_t kItems = 1'000'000;
            foundation::MsQueue<uint64_t> q;
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
                        q.enqueue(val);
                    }
                });
            }

            std::vector<std::thread> consumers;
            for (int i = 0; i < n_cons; ++i) {
                consumers.emplace_back([&]() {
                    while (!go.load(std::memory_order_acquire)) {}
                    while (pop_count.load(std::memory_order_relaxed) < kItems) {
                        uint64_t v;
                        if (q.dequeue(v)) {
                            do_not_optimize(v);
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
            q.drain();

            double ns = static_cast<double>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count());
            printf("%-40s  %.1f ns/item  (%.0f M items/sec)\n",
                   label, ns / kItems, kItems / ns * 1000.0);
        };

        run_ms("ms_queue 1P-1C throughput", 1, 1);
        run_ms("ms_queue 2P-2C throughput", 2, 2);
        run_ms("ms_queue 4P-4C throughput", 4, 4);
    }

    printf("\n--- ChaseLevDeque (work-stealing deque) ---\n\n");

    // --- Benchmark 19: owner push+pop roundtrip (no thieves, 1 thread) ---
    //
    // Push one item then immediately pop it — no CAS involved on the hot path.
    // Owner push: relaxed store to data[], release fence, relaxed store to bottom_.
    // Owner pop (non-last case): relaxed store to bottom_, seq_cst fence,
    //   relaxed load of top_, relaxed load of data[], relaxed restore of bottom_.
    //
    // The seq_cst fence in pop() is the dominant cost on x86 (MFENCE ~10 ns).
    // With exactly one item, pop() also does a seq_cst CAS (as the "last element"
    // race path is always triggered). Expected: ~25–40 ns.
    //
    // Compare with MpmcQueue roundtrip (~21 ns) and MsQueue (~250 ns).
    // ChaseLev is faster than MsQueue (no malloc) and competitive with MPMC.
    {
        foundation::ChaseLevDeque<uint64_t> dq;
        auto result = bench::run_bench("chase_lev  push+pop (1T, no CAS fast path)", 500'000, 10'000,
            [&]() {
                dq.push(1ULL);
                auto v = dq.pop();
                do_not_optimize(v);
            });
        bench::print_result(result);
    }

    // --- Benchmark 20: steal throughput (1 owner, N thieves) ---
    //
    // Owner pushes at full speed; N thief threads steal at full speed.
    // This is the steady-state of a work-stealing thread pool where one thread
    // generates tasks and others consume them. Steal() does:
    //   acquire load of top_, seq_cst fence, acquire load of bottom_,
    //   acquire load of array_, relaxed load of data[], seq_cst CAS on top_.
    //
    // Contention on top_: all N thieves race on the same atomic. With N thieves,
    // most CAS calls fail and retry. Throughput scales sublinearly.
    {
        auto run_steal = [](const char* label, int n_thieves) {
            constexpr std::size_t kItems = 2'000'000;
            foundation::ChaseLevDeque<uint64_t> dq;
            std::atomic<bool> go{false};
            std::atomic<std::size_t> stolen{0};

            // Pre-push all items so owner doesn't interfere.
            for (std::size_t i = 0; i < kItems; ++i)
                dq.push(static_cast<uint64_t>(i));

            std::vector<std::thread> thieves;
            thieves.reserve(static_cast<std::size_t>(n_thieves));
            for (int i = 0; i < n_thieves; ++i) {
                thieves.emplace_back([&]() {
                    while (!go.load(std::memory_order_acquire)) {}
                    while (stolen.load(std::memory_order_relaxed) < kItems) {
                        auto v = dq.steal();
                        if (v) {
                            do_not_optimize(*v);
                            stolen.fetch_add(1, std::memory_order_relaxed);
                        }
                    }
                });
            }

            auto t0 = std::chrono::steady_clock::now();
            go.store(true, std::memory_order_release);
            for (auto& t : thieves) t.join();
            auto t1 = std::chrono::steady_clock::now();

            double ns = static_cast<double>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count());
            printf("%-44s  %.1f ns/steal  (%.0f M/sec)\n",
                   label, ns / kItems, kItems / ns * 1000.0);
        };

        run_steal("chase_lev steal throughput 1T", 1);
        run_steal("chase_lev steal throughput 2T", 2);
        run_steal("chase_lev steal throughput 4T", 4);
    }

    printf("\n--- WorkStealingPool ---\n\n");

    // --- Benchmark 21: parallel_for throughput (task overhead measurement) ---
    //
    // N empty tasks, all run in parallel, timed end-to-end.
    // This measures the per-task scheduling overhead: submit to inbox/deque,
    // worker wakes, executes, decrements pending_, signals done.
    // Task body is a single atomic increment — the rest is pool overhead.
    //
    // With 4 workers, ideal speedup is 4x. Real speedup is lower due to:
    //   - Contention on inbox_mu_ for external submissions
    //   - done_cv_ mutex in execute() when pending_ → 0
    //   - Cache misses on pending_ (shared atomic, N workers write to it)
    {
        auto bench_pfor = [](const char* label, std::size_t n_workers, std::size_t n_tasks) {
            foundation::WorkStealingPool pool(n_workers);
            std::atomic<std::size_t> count{0};

            auto t0 = std::chrono::steady_clock::now();
            pool.parallel_for(n_tasks, [&](std::size_t) {
                count.fetch_add(1, std::memory_order_relaxed);
            });
            auto t1 = std::chrono::steady_clock::now();

            double ns = static_cast<double>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count());
            printf("%-42s  %.1f ns/task  (%.0f M tasks/sec)\n",
                   label, ns / static_cast<double>(n_tasks),
                   static_cast<double>(n_tasks) / ns * 1000.0);
        };

        bench_pfor("ws_pool parallel_for 1W  10K tasks",  1, 10'000);
        bench_pfor("ws_pool parallel_for 2W  10K tasks",  2, 10'000);
        bench_pfor("ws_pool parallel_for 4W  10K tasks",  4, 10'000);
        bench_pfor("ws_pool parallel_for 4W 100K tasks",  4, 100'000);
    }

    // --- Benchmark 22: parallel speedup on real work (vector sum) ---
    //
    // Sum N integers partitioned into kChunks chunks. Measures actual parallel
    // speedup relative to sequential: how much of the theoretical 4x is
    // realised when each task does ~microseconds of real work (not just overhead).
    // Tasks with more compute amortise the scheduling cost better.
    {
        constexpr std::size_t N       = 10'000'000;
        constexpr std::size_t kChunks = 400;
        std::vector<int> data(N);
        std::iota(data.begin(), data.end(), 0);
        std::vector<long long> partial(kChunks, 0);

        // Sequential baseline
        auto t0 = std::chrono::steady_clock::now();
        long long seq_sum = 0;
        for (std::size_t i = 0; i < N; ++i) seq_sum += data[i];
        auto t1 = std::chrono::steady_clock::now();
        double seq_ns = static_cast<double>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count());
        printf("%-42s  %.1f ms  (sum=%lld)\n",
               "ws_pool vector sum sequential", seq_ns / 1e6, seq_sum);

        // Parallel with 4 workers
        foundation::WorkStealingPool pool(4);
        t0 = std::chrono::steady_clock::now();
        pool.parallel_for(kChunks, [&](std::size_t c) {
            std::size_t start = c * (N / kChunks);
            std::size_t end   = start + (N / kChunks);
            long long s = 0;
            for (std::size_t i = start; i < end; ++i) s += data[i];
            partial[c] = s;
        });
        long long par_sum = 0;
        for (auto s : partial) par_sum += s;
        t1 = std::chrono::steady_clock::now();
        double par_ns = static_cast<double>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count());
        printf("%-42s  %.1f ms  (speedup=%.1fx, sum=%lld)\n",
               "ws_pool vector sum 4 workers", par_ns / 1e6,
               seq_ns / par_ns, par_sum);
    }

    // -----------------------------------------------------------------------
    // Arena allocator benchmarks
    // -----------------------------------------------------------------------
    printf("\n--- Arena allocator ---\n");

    // Bench: Arena bump-only alloc (no free), 32-byte objects
    {
        foundation::Arena arena(64u << 20);  // 64 MiB slab
        auto r = bench::run_bench("arena bump alloc 32B", 500'000, 100,
            [&arena]() {
                do_not_optimize(arena.alloc(32, 32));
            });
        bench::print_result(r);
    }

    // Bench: SizeClassedArena alloc+free (freelist path), 32-byte objects
    {
        foundation::SizeClassedArena sc(64u << 20);
        void* last = sc.alloc(32);
        auto r = bench::run_bench("sc_arena alloc+free 32B (freelist)", 500'000, 100,
            [&sc, &last]() {
                sc.free(last, 32);
                last = sc.alloc(32);
                do_not_optimize(last);
            });
        bench::print_result(r);
    }

    // Bench: malloc alloc+free, 32-byte objects
    {
        void* last = ::malloc(32);
        auto r = bench::run_bench("malloc+free 32B", 500'000, 100,
            [&last]() {
                ::free(last);
                last = ::malloc(32);
                do_not_optimize(last);
            });
        bench::print_result(r);
        ::free(last);
    }

    // Bench: ThreadLocalArena alloc+free, mixed sizes
    {
        static const std::size_t kSizes[] = {8, 32, 128, 512};
        std::size_t idx = 0;
        void* last = foundation::ThreadLocalArena::alloc(8);
        std::size_t last_sz = 8;
        auto r = bench::run_bench("thread_local_arena mixed sizes", 500'000, 100,
            [&]() {
                foundation::ThreadLocalArena::free(last, last_sz);
                last_sz = kSizes[idx++ & 3];
                last = foundation::ThreadLocalArena::alloc(last_sz);
                do_not_optimize(last);
            });
        bench::print_result(r);
        foundation::ThreadLocalArena::free(last, last_sz);
    }

    // Bench: malloc alloc+free, same mixed sizes
    {
        static const std::size_t kSizes[] = {8, 32, 128, 512};
        std::size_t idx = 0;
        void* last = ::malloc(8);
        std::size_t last_sz = 8;
        auto r = bench::run_bench("malloc+free mixed sizes", 500'000, 100,
            [&]() {
                ::free(last);
                last_sz = kSizes[idx++ & 3];
                last = ::malloc(last_sz);
                do_not_optimize(last);
            });
        bench::print_result(r);
        ::free(last);
    }

    // -----------------------------------------------------------------------
    // NUMA allocator benchmarks
    // -----------------------------------------------------------------------
    printf("\n--- NUMA allocator ---\n");
    {
        auto topo = foundation::NumaTopology::detect();
        printf("Topology: %d node(s), %d CPU(s)\n", topo.num_nodes, topo.num_cpus);

        auto na = foundation::NumaArena::make(topo, 64u << 20);

        // Bench: local-node alloc+free (node 0), 32-byte objects
        {
            void* last = na.alloc_on_node(0, 32);
            auto r = bench::run_bench("numa_arena alloc+free node-0 32B", 500'000, 100,
                [&na, &last]() {
                    na.free_on_node(0, last, 32);
                    last = na.alloc_on_node(0, 32);
                    do_not_optimize(last);
                });
            bench::print_result(r);
            na.free_on_node(0, last, 32);
        }

        // Bench: cross-node alloc — only meaningful on Linux multi-node machines
        if (topo.num_nodes >= 2) {
            void* last = na.alloc_on_node(1, 32);
            auto r = bench::run_bench("numa_arena alloc+free node-1 32B (cross-node)", 500'000, 100,
                [&na, &last]() {
                    na.free_on_node(1, last, 32);
                    last = na.alloc_on_node(1, 32);
                    do_not_optimize(last);
                });
            bench::print_result(r);
            na.free_on_node(1, last, 32);
            printf("  ^ cross-node penalty visible above (expected +50-100%% vs node-0)\n");
        } else {
            printf("  (single-node platform: cross-node benchmark skipped;\n"
                   "   run on a Linux 2-socket machine to see ~50-100%% penalty)\n");
        }
    }

    // -----------------------------------------------------------------------
    // Hardware counter analysis
    // -----------------------------------------------------------------------
    // Measures IPC, LLC miss rate, and branch miss rate on a selection of
    // workloads.  On Linux these come from perf_event_open(); on macOS
    // the counter infrastructure is unavailable (see perf/perf.h).
    //
    // Reading guide:
    //   IPC   — instructions per cycle.  > 2 = good.  < 1 = memory/branch bound.
    //   L3miss — fraction of LLC accesses that go to DRAM.  < 1% = data is hot.
    //   Brmiss — fraction of branches mispredicted.  < 1% = predictor works.
    // -----------------------------------------------------------------------
    printf("\n--- Hardware counter analysis ---\n");
    {
        foundation::PerfCounters probe;
        if (!probe.available()) {
            printf("  perf counters not available on this platform (macOS).\n");
            printf("  On Linux with perf_event_paranoid <= 2, this section will\n");
            printf("  show IPC / L3-miss-rate / branch-miss-rate per workload.\n");
            printf("  Expected values (AWS c5.2xlarge, Intel Xeon Platinum 8275CL):\n");
            printf("    sequential int sum (hot data): IPC~3.5  L3miss~0%%  Brmiss~0.1%%\n");
            printf("    random pointer chase (cold):   IPC~0.3  L3miss~50%% Brmiss~0.5%%\n");
            printf("    branch-heavy sort:             IPC~1.5  L3miss~2%%  Brmiss~5%%\n");
        } else {
            // Workload A: sequential integer sum (cache-warm, predictable)
            {
                constexpr std::size_t kN = 1 << 20;
                std::vector<int32_t> data(kN);
                std::iota(data.begin(), data.end(), 0);
                volatile int64_t sink = 0;
                auto snap = foundation::measure_perf(100'000, [&]{
                    // Sum a 64-element chunk — fits in L1 after warmup
                    int64_t s = 0;
                    for (std::size_t i = 0; i < 64; ++i) s += data[i];
                    sink = sink + s;
                });
                printf("  sequential int sum (64-elem, L1-hot):\n");
                snap.print();
            }

            // Workload B: random pointer chase (LLC-thrashing)
            {
                constexpr std::size_t kN = 1 << 22;  // 16 MB — beyond L3 on most CPUs
                std::vector<std::size_t> ptrs(kN);
                // Build a random cycle through all indices
                std::iota(ptrs.begin(), ptrs.end(), 0);
                // Fisher-Yates shuffle with splitmix64
                uint64_t rng_state = 0xdeadbeefcafe1234ULL;
                auto rng_next = [&]() -> uint64_t {
                    rng_state += 0x9e3779b97f4a7c15ULL;
                    uint64_t z = rng_state;
                    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
                    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
                    return z ^ (z >> 31);
                };
                for (std::size_t i = kN - 1; i > 0; --i) {
                    std::size_t j = static_cast<std::size_t>(rng_next()) % (i + 1);
                    std::swap(ptrs[i], ptrs[j]);
                }
                volatile std::size_t idx = 0;
                auto snap = foundation::measure_perf(10'000, [&]{
                    idx = ptrs[idx % kN];
                });
                printf("  random pointer chase (16MB, LLC-thrashing):\n");
                snap.print();
            }

            // Workload C: arena bump alloc (should show high IPC, low miss)
            {
                foundation::Arena arena(64u << 20);
                volatile void* sink = nullptr;
                auto snap = foundation::measure_perf(100'000, [&]{
                    if (arena.used() + 32 > arena.capacity()) arena.reset();
                    sink = arena.alloc(32, 32);
                });
                printf("  arena bump alloc 32B:\n");
                snap.print();
            }
        }
    }

    return 0;
}
