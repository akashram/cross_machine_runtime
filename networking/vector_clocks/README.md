# Vector Clocks

**Status: code-complete AND locally run — portable, builds and runs everywhere including this Mac.**

## What this measures
Lamport timestamps and vector clocks, integrated into event logging.

## Design
`LamportClock` (scalar, `max(local, received)+1`) and `VectorClock`
(one component per process, `tick()`/`onReceive()`/`compare()`) —
`vector_clock.cpp`. `compare()` operates on captured snapshots, not live
clocks (a live clock keeps changing — callers capture `values()` at the
moment of interest, e.g. when sending or logging an event). This is the
causality primitive `chandy_lamport` (step 18) builds its consistent-cut
detection on: a vector clock can distinguish "definitely causally
ordered" from "definitely concurrent," which a scalar Lamport timestamp
provably cannot (demonstrated directly in the test below, not just
asserted).

## Sanity-run output (Mac, 2026-07-19)

`vector_clock_test`: three simulated processes (P0 ticks, P0's message
causes P1's event, P2 ticks independently/concurrently), verifying vector
clocks correctly classify each pair, plus a side-by-side Lamport clock run
showing the same scenario's scalar timestamps can't make that distinction:

```
A happened-before B                           OK
B happened-after A                            OK
A concurrent with C                           OK
C concurrent with A                           OK
B concurrent with C                           OK
Lamport: la < lb (order preserved for causal pair) OK
(Lamport la=1 lc=1 — a real number exists for both, but nothing in
 a scalar timestamp says whether they're causally related or concurrent;
 that ambiguity is exactly what the vector clock comparisons above resolve.)
PASS
```

## Results
This step is a correctness primitive, not a performance benchmark — no
Results table. Integration into the project's actual event log (piggy-
backed on `networking/common::Channel` messages) happens as part of
`chandy_lamport/` (step 18) and `raft/` (step 19).

## Hardware notes
- Builds and runs anywhere (validated on Mac); no hardware dependency at all.
