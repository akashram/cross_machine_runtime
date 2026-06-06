// Busy-poll vs OS-wait latency comparison — Phase 2 step 13
//
// SCENARIO
// --------
// Single-producer / single-consumer.  The producer sends timestamped
// messages at a controlled inter-arrival rate; the consumer records the
// end-to-end latency (TSC delta from just before push to just after pop).
//
// TWO APPROACHES
// --------------
// busy-poll: consumer spins on SpscQueue::pop() with x86 PAUSE hints.
//   No OS involvement on the receive side — latency is bounded by the
//   cache-coherence round-trip between the two cores (~50–200 ns on a
//   shared-LLC machine).
//
// os-wait: consumer blocks on a condition variable; producer notifies
//   after each push.  The OS scheduler must wake the consumer before it
//   can read the message.  On macOS the wake-up overhead is 5–50 µs;
//   on Linux with SCHED_FIFO it can be as low as 2–5 µs.
//
// CROSSOVER POINT
// ---------------
// busy-poll wins when the inter-arrival gap is shorter than the OS wake-up
// overhead, because the consumer is already spinning and reacts instantly.
// os-wait becomes competitive when messages arrive infrequently enough that
// the wake-up cost is amortised — and it wins on CPU utilisation (the
// consumer core is fully released between messages).
//
// METRICS
// -------
// p50, p99, p999 latency across 5 inter-arrival rates:
//   0 (tight loop), 1 µs, 10 µs, 100 µs, 1 ms.
//
// TSan notes
// ----------
// busy-poll: SpscQueue uses acquire/release atomics — TSan clean.
// os-wait:   std::mutex + std::condition_variable — TSan clean.
// latencies vector: written by consumer, read by main after join()
//   (join establishes happens-before) — no data race.

#include "affinity/affinity.h"
#include "foundation/spsc_queue.h"
#include "foundation/bench/bench.h"

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

#if defined(__x86_64__) || defined(__i386__)
#  include <immintrin.h>
#  define CPU_PAUSE() _mm_pause()
#else
#  define CPU_PAUSE() __asm__ __volatile__("yield" ::: "memory")
#endif

using cpu_engine::ThreadPinner;
using bench::tsc_now;
using bench::tsc_ticks_per_ns;
using bench::tsc_to_ns;

// ---------------------------------------------------------------------------
// Message type
// ---------------------------------------------------------------------------

struct alignas(64) Msg {
    uint64_t send_tsc;
    int32_t  seq;
};
static_assert(std::is_trivially_copyable_v<Msg>);

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static constexpr std::size_t kQueueCap = 1u << 14; // 16K slots

static void spin_until(uint64_t target) noexcept {
    while (tsc_now() < target) CPU_PAUSE();
}

static double pct_ns(std::vector<uint64_t>& v, double p) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    auto idx = static_cast<std::size_t>(p * static_cast<double>(v.size()));
    if (idx >= v.size()) idx = v.size() - 1;
    return tsc_to_ns(v[idx]);
}

static void format_ia(char* buf, std::size_t n, uint64_t ns) {
    if (ns == 0)           std::snprintf(buf, n, "tight-loop");
    else if (ns < 1'000)   std::snprintf(buf, n, "%llu ns", static_cast<unsigned long long>(ns));
    else if (ns < 1'000'000) std::snprintf(buf, n, "%llu us", static_cast<unsigned long long>(ns/1'000));
    else                   std::snprintf(buf, n, "%llu ms", static_cast<unsigned long long>(ns/1'000'000));
}

// ---------------------------------------------------------------------------
// Busy-poll: consumer spins on SpscQueue::pop()
// ---------------------------------------------------------------------------

static void run_busy_poll(uint64_t ia_ticks, int n_samples,
                           std::vector<uint64_t>& out) {
    foundation::SpscQueue<Msg, kQueueCap> q;
    std::atomic<bool>  done{false};
    std::atomic<bool>  ready{false};

    std::thread consumer([&]{
        ThreadPinner::pin(1);
        ready.store(true, std::memory_order_release);
        Msg msg{};
        while (!done.load(std::memory_order_relaxed) || !q.empty()) {
            if (q.pop(msg)) {
                out.push_back(tsc_now() - msg.send_tsc);
            } else {
                CPU_PAUSE();
            }
        }
    });

    // Wait for consumer to start before first send
    while (!ready.load(std::memory_order_acquire)) CPU_PAUSE();

    uint64_t next = tsc_now() + ia_ticks;
    for (int seq = 0; seq < n_samples; ++seq) {
        if (ia_ticks > 0) spin_until(next);
        Msg msg{tsc_now(), seq};
        while (!q.push(msg)) CPU_PAUSE(); // back-pressure
        next += ia_ticks;
    }
    done.store(true, std::memory_order_release);
    consumer.join();
}

// ---------------------------------------------------------------------------
// OS-wait: consumer blocks on std::condition_variable
// ---------------------------------------------------------------------------

static void run_os_wait(uint64_t ia_ticks, int n_samples,
                         std::vector<uint64_t>& out) {
    std::queue<Msg>          q;
    std::mutex               mtx;
    std::condition_variable  cv;
    std::atomic<bool>        done{false};
    std::atomic<bool>        ready{false};

    std::thread consumer([&]{
        ThreadPinner::pin(1);
        ready.store(true, std::memory_order_release);
        while (true) {
            Msg msg{};
            {
                std::unique_lock<std::mutex> lock(mtx);
                cv.wait(lock, [&]{
                    return !q.empty() || done.load(std::memory_order_relaxed);
                });
                if (q.empty()) break; // done + empty → exit
                msg = q.front();
                q.pop();
            }
            out.push_back(tsc_now() - msg.send_tsc);
        }
    });

    while (!ready.load(std::memory_order_acquire)) CPU_PAUSE();

    uint64_t next = tsc_now() + ia_ticks;
    for (int seq = 0; seq < n_samples; ++seq) {
        if (ia_ticks > 0) spin_until(next);
        Msg msg{tsc_now(), seq};
        {
            std::lock_guard<std::mutex> lock(mtx);
            q.push(msg);
        }
        cv.notify_one(); // notify outside lock — avoids "hurry up and wait"
        next += ia_ticks;
    }

    done.store(true, std::memory_order_relaxed);
    cv.notify_all(); // wake consumer so it sees done=true
    consumer.join();
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main() {
    ThreadPinner::pin(0); // pin producer to CPU 0
    (void)tsc_ticks_per_ns(); // calibrate TSC

    struct Config {
        uint64_t ia_ns;
        int      n;
    };
    static constexpr Config kConfigs[] = {
        {0,           100'000},
        {1'000,       100'000},   // 1 µs
        {10'000,       20'000},   // 10 µs
        {100'000,       5'000},   // 100 µs
        {1'000'000,     1'000},   // 1 ms
    };

    printf("Busy-poll vs OS-wait — producer/consumer latency\n");
    printf("==================================================\n");
    printf("Platform: macOS (advisory pinning; Linux gives tighter numbers)\n");
    printf("Queue: foundation::SpscQueue<Msg, %zu> for busy-poll; "
           "std::queue+condvar for os-wait\n\n", kQueueCap);

    printf("%-12s  %-14s  %7s  %9s  %9s  %9s\n",
           "approach", "inter-arrival", "samples", "p50", "p99", "p999");
    printf("%s\n", std::string(72, '-').c_str());

    for (int pass = 0; pass < 2; ++pass) {
        if (pass == 1) printf("\n");

        for (const auto& cfg : kConfigs) {
            auto ia_ticks = static_cast<uint64_t>(
                static_cast<double>(cfg.ia_ns) * tsc_ticks_per_ns());

            std::vector<uint64_t> lats;
            lats.reserve(static_cast<std::size_t>(cfg.n));

            if (pass == 0)
                run_busy_poll(ia_ticks, cfg.n, lats);
            else
                run_os_wait(ia_ticks, cfg.n, lats);

            char ia_str[32];
            format_ia(ia_str, sizeof(ia_str), cfg.ia_ns);

            printf("%-12s  %-14s  %7d  %8.0f ns  %8.0f ns  %8.0f ns\n",
                   (pass == 0 ? "busy-poll" : "os-wait"),
                   ia_str,
                   static_cast<int>(lats.size()),
                   pct_ns(lats, 0.50),
                   pct_ns(lats, 0.99),
                   pct_ns(lats, 0.999));
        }
    }

    printf("\n");
    printf("Crossover: busy-poll wins when inter-arrival < OS wake-up latency.\n");
    printf("  Read p50(os-wait, tight-loop) to find the wake-up floor on this machine.\n");
    printf("  When inter-arrival >> p50(os-wait), os-wait is competitive and saves a CPU core.\n");

    ThreadPinner::unpin();
}
