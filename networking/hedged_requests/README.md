# Hedged Requests

**Status: code-complete AND locally run — portable, builds and runs everywhere including this Mac.**

## What this measures
Duplicate slow requests to a second backend after a latency threshold,
measure tail latency improvement.

## Design
`hedgedCall()` (`hedged_request.cpp`) launches `backends[0]`; if it
hasn't returned within `hedgeDelay`, launches `backends[1..]` as well —
every backend races into one shared result slot (first writer wins via
`compare_exchange_strong`), and the call returns whichever finishes
first. Losing backends are **not** cancelled (a blocking `std::function`
has no cancellation hook) — they keep running in the background and their
result is discarded; a real deployment would need the backend call itself
to accept a deadline. `backends` are arbitrary callables, not tied to
`networking/common::Channel`, so this composes with any blocking call.

## Sanity-run output (Mac, 2026-07-19)

`hedged_request_test`: a flaky backend (5ms typically, 200ms straggler on
~8% of calls) raced against a reliable ~6ms backend, 200 requests each,
compared against the same flaky backend alone (same RNG seed, so both
runs see an identical straggler pattern — isolates hedging's effect from
run-to-run noise):

```
               p50 (ms)   p99 (ms)
no hedging         6.40     205.18
hedged             6.55      33.04
hedged on 14/200 requests (~7% straggler rate observed)
PASS
```

p99 drops from 205ms to 33ms (straggler calls get rescued by the reliable
backend ~20ms in, instead of running the full 200ms) while p50 barely
moves (6.40ms → 6.55ms) — most calls never hedge at all, since the
primary backend answers well within `hedgeDelay` the other ~93% of the
time. This is the whole point of hedging: it buys back tail latency
almost for free in the common case.

## Results
This step's real result is the sanity-run above — the tradeoff (tail
latency vs. duplicated backend load) doesn't change qualitatively with
real network hardware, only the absolute numbers. No hardware-gated
Results table.

## Hardware notes
- Builds and runs anywhere (validated on Mac); no hardware dependency at all.
