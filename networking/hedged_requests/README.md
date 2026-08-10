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

**A real finding, not just a design note**: losing backends running on
after `hedgedCall()` returns (see above) means a straggler from call `i`
can genuinely still be executing when call `i+1` starts. An early version
of `hedged_request_test.cpp` passed a single `std::mt19937&` by reference
into every call's backend lambda — two overlapping calls' threads then
touched that one generator with no synchronization, a real data race
caught by TSan (2026-08-10), not a false positive. Fixed by giving each
call its own independently-seeded generator (seed = base + call index)
instead: no shared mutable state to race on, while still giving the
no-hedging and hedging runs the identical straggler pattern call-for-call
that isolating hedging's effect depends on. `hedged_request.cpp` itself
needed no change — the race was in the test's traffic generation, not the
library.

## Sanity-run output (Mac, 2026-08-10)

`hedged_request_test`: a flaky backend (5ms typically, 200ms straggler on
~8% of calls) raced against a reliable ~6ms backend, 200 requests each,
compared against the same flaky backend alone (same per-call seed
sequence, so both runs see an identical straggler pattern — isolates
hedging's effect from run-to-run noise):

```
               p50 (ms)   p99 (ms)
no hedging         5.12     202.44
hedged             5.25      29.41
hedged on 7/200 requests (~4% straggler rate observed)
PASS
```

p99 drops from ~202ms to ~29ms (straggler calls get rescued by the
reliable backend ~20ms in, instead of running the full 200ms) while p50
barely moves (5.12ms → 5.25ms) — most calls never hedge at all, since the
primary backend answers well within `hedgeDelay` the other ~96% of the
time. This is the whole point of hedging: it buys back tail latency
almost for free in the common case. (Straggler rate observed dropped from
the earlier ~7% to ~4% purely because per-call seeding draws a different,
still-deterministic sequence than the old single shared generator did —
not a behavior change in the code being tested.)

## Results
This step's real result is the sanity-run above — the tradeoff (tail
latency vs. duplicated backend load) doesn't change qualitatively with
real network hardware, only the absolute numbers. No hardware-gated
Results table.

## Hardware notes
- Builds and runs anywhere (validated on Mac, including 3 clean runs
  under TSan); no hardware dependency at all.
