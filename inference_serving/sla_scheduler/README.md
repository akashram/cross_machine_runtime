# sla_scheduler

**Status: code-complete AND locally run — pure CPU scheduling logic, no
GPU dependency.**

## What this measures

PLAN.md Phase 9 step 3: latency budget per request, preemption when
budget exceeded, priority queuing.

## Design

- `SlaScheduler` implements Earliest-Deadline-First (EDF): every request
  carries a deadline (`arrival_time_s + latency_budget_s`), and
  `schedule_tick()` recomputes the active set as exactly the
  `max_batch_size` not-yet-finished requests with the earliest deadlines
  among *all* pending ones (active or waiting) — not a bespoke heuristic;
  EDF is the textbook-optimal policy for single-resource, hard-deadline,
  preemption-allowed scheduling (if any policy can meet every deadline in
  a feasible workload, EDF will). Preemption falls out of that rule
  directly: a request that was active but is no longer among the
  earliest deadlines gets evicted back to waiting when a more urgent
  request arrives.
- `sla_scheduler_bench.cpp` measures the outcome preemption is *for* —
  SLA-violation rate — by running the same 200-request mixed-urgency
  trace (20% tight/interactive-latency-class requests at 1.5x their
  service time as budget, 80% loose at 8x) through two policies: plain
  FIFO admission with no priority or preemption (what
  `continuous_batching`'s `ContinuousBatcher` does — reimplemented
  standalone here so this file can track finish-time-vs-deadline without
  adding that concern to `batcher.h`), and the real `SlaScheduler`
  (EDF).
- Modeling choice, stated plainly: preempted requests *pause* in this
  simulation (their progress is preserved, they just wait longer) rather
  than losing decoded KV state and restarting. A real system's actual
  preempt/evict/resume cost is backend-specific (whether KV blocks get
  swapped to host memory or dropped and recomputed) — out of scope here;
  this step's question is whether EDF's *admission* priority improves
  deadline outcomes at all, which doesn't depend on that cost.

## Results (captured 2026-07-27, Apple clang 14 / `-std=c++2b`, this Mac)

```
policy         requests     violations   violation rate
fifo                200             73            36.5%
edf                 200              0             0.0%
```

## Findings

- EDF meets every deadline in this trace (0% violations) while FIFO
  misses more than a third (36.5%) — this workload is fully
  EDF-schedulable (the deadline misses under FIFO are a scheduling-policy
  artifact, not a genuine capacity shortfall), which is exactly what EDF's
  optimality guarantee predicts: FIFO has no way to know a just-arrived
  tight-budget request is more urgent than an already-running loose-budget
  one, so it only ever gets lucky.
- This is a stronger result than `continuous_batching`'s 1.85x throughput
  finding, and a different axis entirely: that step improved raw
  throughput by eliminating padding waste; this step improves deadline
  *outcomes* by reordering admission, at no throughput cost (both
  policies serve all 200 requests with the same total service-tick
  demand) — the two techniques compose rather than compete, matching
  PLAN.md's step ordering (batching first, then SLA-aware admission on
  top).

## Hardware notes
None. Pure scheduling-policy simulation; a real serving stack would layer
this admission logic under `paged_kv`'s block allocator and a real
decode kernel unchanged, same as `continuous_batching`.
