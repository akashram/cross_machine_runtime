# Chandy-Lamport Distributed Snapshots

**Status: code-complete AND locally run — portable, builds and runs everywhere including this Mac.**

## What this measures
Distributed global state capture without stopping execution. Validates
consistency of captured snapshots.

## Design
Classic algorithm (Chandy & Lamport, 1985) over
`networking/common::Channel`, one receiver thread per peer feeding a
central marker/transfer state machine guarded by one mutex
(`chandy_lamport.cpp`). Any node can initiate. The subtle part, worth
documenting because it was a real correctness bug caught locally (not
just a design note): the algorithm's FIFO precondition — "a message sent
after this process's local snapshot arrives after the marker on that
channel" — only holds if *recording local state* and *broadcasting
markers to every peer* happen as one atomic unit relative to any
concurrent application-level send. The first implementation released the
lock between those two steps (to avoid holding it during socket I/O);
that leaves a window where a concurrent send could beat a marker to one
peer while already being excluded from the recorded local state,
double-counting or losing money in the invariant check below. Fixed by
holding the lock for the whole record-and-broadcast sequence, and by
adding `ChandyLamportNode::atomically()` — any caller mixing
`sendTransfer` with concurrent snapshots must route sends through it, not
call `sendTransfer` directly.

## Sanity-run output (Mac, loopback, 2026-07-19)

`chandy_lamport_test`: the classic bank-account invariant test — 3 ranks
start with balance=100 each (total=300), background threads on every rank
send small transfers to random peers *concurrently* with rank 0 initiating
a snapshot (ranks 1/2 respond passively). Regardless of which transfers
land pre-cut, post-cut, or genuinely in-flight, `sum(recorded local state)
+ sum(recorded in-flight channel messages)` must equal 300:

```
rank 0: recorded local state = 100
rank 1: recorded local state = 99
rank 1: channel from 2 recorded 1 in-flight message(s), sum=1
rank 2: recorded local state = 100
total recorded = 300 (expected 300)
PASS
```

Run 8 consecutive times (different scheduling/interleaving each run,
since the background load threads and the snapshot race for real) with no
failures — this is the property that actually matters, not any specific
outcome.

## Results
This step is a correctness primitive — no hardware-dependent performance
numbers to report. Multi-node validation would confirm the same invariant
holds over a real network (not just loopback), which doesn't change the
algorithm, only the RTTs involved.

## Hardware notes
- Builds and runs anywhere (validated on Mac, including 8 repeated runs
  under real concurrency); no hardware dependency at all.
