# continuous_batching

**Status: code-complete AND locally run — pure CPU scheduling logic, no
GPU dependency.**

## What this measures

PLAN.md Phase 9 step 2: dynamic request arrival, batch formation across
variable sequence lengths, throughput improvement vs. static batching.

## Design

- `ContinuousBatcher`: a waiting queue plus an active-sequence set.
  `next_batch(max_batch_size)` admits waiting requests into any free slot
  and returns every active sequence id — the set that should run one
  decode step this tick. `finish_sequence()` frees a slot immediately, so
  the *next* call to `next_batch()` can admit a new request into it,
  rather than waiting for every sequence in some group to finish (the
  batch just returns whichever ids are currently active; there's no
  fixed "group" at all).
- `continuous_batching_bench.cpp` measures the throughput improvement
  PLAN.md asks for by running the same synthetic request trace (200
  requests, response lengths uniform 1..64 — the variance that makes
  static batching's cost visible; a uniform-length trace would hide it)
  through two policies:
  - **Static**: fixed-size batches of 16, strictly in arrival order. A
    batch runs for `max(response_len)` ticks — every slot stays occupied
    (and billed as device time) for the whole batch, even after its own
    sequence finished, since the batch can't return control until the
    longest sequence completes. This is the real, measurable cost the
    step exists to quantify, not a hypothetical.
  - **Continuous**: drives the real `ContinuousBatcher` class tick by
    tick — not a separate hand-derived formula, the actual admission
    logic under test.
  - Simulated in discrete ticks, not `std::chrono` wall-clock: there's no
    real GPU compute happening here, so timing this loop would measure
    how fast this Mac's CPU runs a scheduler, not the scheduling policy's
    effect. Device-slot-tick accounting (busy (tick, slot) units consumed
    per useful token) is the real quantity of interest, and it's exact
    given the discrete-event simulation, not estimated.

## Results (captured 2026-07-27, Apple clang 14 / `-std=c++2b`, this Mac)

```
policy                  total_ticks  device_slot_ticks         util %     tok/tick
static                          808              12416          52.8%         8.12
continuous                      436               6559         100.0%        15.04

throughput speedup (tok/tick): 1.85x
makespan speedup: 1.85x
```

## Findings

- Continuous batching reaches 100% device-slot utilization by
  construction (a slot-tick is only ever counted while a real active
  sequence occupies it) — the entire 1.85x throughput gain over static
  batching in this trace is static batching's padding waste: slots held
  busy-but-idle waiting for the longest sequence in their group to finish.
- This matches the vLLM paper's (Kwon et al.) core motivating finding
  qualitatively — variable response lengths make static batching's
  "wait for the stragglers" cost real and measurable, not a
  micro-optimization — reproduced here as an actual simulation result,
  not asserted from the paper.
- The speedup magnitude (1.85x) is specific to this trace's response-
  length distribution (uniform 1..64) and batch size (16); a
  narrower length distribution would shrink the gap, a wider one would
  grow it — `continuous_batching_bench.cpp`'s `make_trace()` is the knob
  to rerun this against a different assumed workload shape.

## Hardware notes
None. This step's throughput comparison is a discrete-event scheduling
simulation with no GPU dependency — a real GPU-backed serving stack would
plug `paged_kv`'s block allocator and a real attention/decode kernel in
underneath the same `ContinuousBatcher` admission logic tested here
unchanged.
