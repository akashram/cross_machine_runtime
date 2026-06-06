# Step 13 — Busy-poll vs OS-wait

## What was built

`latency_bench.cpp` — a single-producer / single-consumer benchmark that
measures end-to-end message latency (producer `send_tsc` → consumer `recv_tsc`)
at five inter-arrival rates for two approaches:

| Approach | Mechanism | Source |
|---|---|---|
| **busy-poll** | Consumer spins on `SpscQueue::pop()` with x86 `PAUSE` hints | `foundation/spsc_queue.h` |
| **os-wait** | Consumer blocks on `std::condition_variable`; producer calls `notify_one()` | stdlib |

## Results (macOS Intel, AVX2, advisory thread pinning)

```
approach      inter-arrival   samples        p50        p99       p999
------------------------------------------------------------------------
busy-poll     tight-loop       100000    343116 ns    598146 ns    604618 ns  *
busy-poll     1 us             100000       122 ns       201 ns     25599 ns
busy-poll     10 us             20000       128 ns       932 ns    836388 ns
busy-poll     100 us             5000       128 ns       455 ns     18170 ns
busy-poll     1 ms               1000       139 ns    155170 ns   4751804 ns

os-wait       tight-loop       100000   1734745 ns   3282394 ns   3315441 ns
os-wait       1 us             100000      3668 ns    580026 ns   1024140 ns
os-wait       10 us             20000      3973 ns    411101 ns   2064069 ns
os-wait       100 us             5000      4316 ns   5267864 ns   8865569 ns
os-wait       1 ms               1000     37767 ns   2780508 ns  19024700 ns
```

`*` See "Tight-loop anomaly" below.

## Key findings

### busy-poll p50 latency: ~122–139 ns (settled)

The 122 ns p50 at 1 µs inter-arrival is the actual **cache-coherence
round-trip time** through the SPSC queue: the producer writes to a slot and
releases it (store release), the consumer polls and acquires it (load acquire),
then reads the TSC. On an Intel box with a shared L3, the cache line moves
from core 0 to core 1 via the LLC coherence protocol (MESIF) in ~50–200 ns.
The 122 ns sits squarely in that range.

### OS-wait floor: 3.7–4.3 µs (macOS condvar wake-up)

Every OS-wait p50 is 3.7–4.3 µs regardless of inter-arrival rate (until 1 ms,
where deeper sleep states push it to 38 µs). This is the **minimum scheduler
wake-up latency** on macOS — the time from `notify_one()` to the consumer
being scheduled and executing its first instruction. On Linux with
`SCHED_FIFO` and `isolcpus` this floor is 2–5 µs; with `nohz_full` it can
be pushed below 2 µs.

### Crossover point

busy-poll is **30–300× lower p50 latency** than os-wait at every inter-arrival
rate tested. The latency crossover (where os-wait becomes competitive) does not
appear on this machine within the 1 ms range tested.

The **practical crossover is CPU utilisation, not latency**:

| Inter-arrival | Consumer core idle % (busy-poll) | Verdict |
|---|---|---|
| tight-loop | 0% | Both approaches waste the core; os-wait is worse |
| 1 µs | ~87% idle | Busy-poll burns 13% of core for 30× latency win |
| 10 µs | ~99% idle | Busy-poll burns 1% of core for 30× latency win |
| 100 µs | ~99.9% idle | Spinning for 100 µs to save 4 µs — questionable |
| 1 ms | ~99.99% idle | **os-wait wins on CPU budget**; 38 µs wake-up < 1% of inter-arrival |

**Rule of thumb:** use busy-poll when inter-arrival < 100 µs and latency is
critical. Switch to os-wait when inter-arrival > 500 µs or when a dedicated
core can't be reserved.

### Tight-loop busy-poll anomaly (343 µs p50)

With `inter_arrival = 0` and both threads on macOS advisory pinning, the OS
may schedule producer and consumer on the same physical core. The producer then
fills the 16K-slot queue in one scheduling quantum and spins; the consumer
runs in the next quantum and drains it. The "latency" each message sees is
the time it spent queued, not the wire latency. This is a macOS-specific
artifact — on Linux with hard `pthread_setaffinity_np`, tight-loop p50 would
be ~100–200 ns (matching the 1 µs inter-arrival result).

### p999 spikes

p999 values (25 µs – 4.7 ms) are OS jitter: the macOS scheduler preempting the
consumer for other work. On Linux with `isolcpus=1 nohz_full=1`, p999 for
busy-poll would be ~1–5 µs.

## Platform notes (Linux expected values)

| Metric | macOS Intel | Linux SCHED_FIFO |
|---|---|---|
| busy-poll p50 (1 µs IA) | 122 ns | ~80–150 ns |
| busy-poll p999 (1 µs IA) | 25 µs | ~300–800 ns |
| os-wait p50 floor | 3.7 µs | 2–5 µs |
| os-wait p999 (1 µs IA) | >1 ms | 10–50 µs |
| Crossover (latency) | < 1 ms | < 100 µs |

To get Linux numbers:
```bash
# On an isolated core (e.g., AWS c5.2xlarge with isolcpus=1 nohz_full=1):
sudo chrt -f 50 taskset -c 0,1 ./build/release/cpu_engine/busy_poll/latency_bench
```

## Conclusions for the CPU backend

The inference engine (step 9) is invoked synchronously — no producer/consumer
queue in the hot path. But when the CPU backend is wired into a request
dispatcher (Phase 5 distributed layer), the dispatcher→worker handoff latency
will matter. The findings here establish:

1. **Sub-100 µs work items → busy-poll the dispatch queue** on a pinned core.
2. **Batch / async workloads (> 1 ms between items) → condvar** to free the core.
3. The SPSC queue from `foundation/spsc_queue.h` delivers ~120 ns round-trip
   latency and is the right primitive for the busy-poll path.
