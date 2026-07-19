# Collective Library (Broadcast, Reduce-Scatter, All-Gather)

**Status: code-complete AND locally run — portable, builds and runs everywhere including this Mac.**

## What this measures
Complete collective library: assembles the algorithms steps 11-13 already
built (ring reduce-scatter/all-gather, binomial tree broadcast) into the
standard named collective operations.

## Design
Not new algorithms (`collectives.cpp` is ~20 lines) — `Broadcast` calls
`tree_broadcast` (step 13), `ReduceScatter` calls `ring_reduce_scatter`
(step 11), `AllGather` calls `ring_all_gather` (step 11) after placing each
rank's contribution at the right slot. That "right slot" is the one
non-obvious part: the ring algorithm's chunk ownership for rank `r` is
chunk `(r+1) % world_size`, not `r` — an artifact of which direction the
ring sends in, not worth an extra rotation step to hide. Both
`ReduceScatter` and `AllGather` document this in `collectives.h` and share
the convention, since `AllGather` is literally what running
`ReduceScatter`'s all-gather half looks like without a preceding reduce.
Getting this indexing wrong was an actual bug caught locally (see below)
before it ever needed hardware to surface.

## Sanity-run output (Mac, loopback, 2026-07-19)

`collectives_test`: Broadcast (root=rank 2), ReduceScatter, and AllGather,
each verified for correctness over a 4-rank loopback mesh:

```
Broadcast: PASS
ReduceScatter: PASS
AllGather: PASS
PASS
```

First run of `ReduceScatter`/`AllGather` actually **failed** — both
assumed rank `r`'s owned chunk was index `r`, when the ring algorithm
actually produces it at `(r+1) % world_size`. Diagnosed by dumping
`ring_reduce_scatter`'s raw output per rank (small 8-element buffer, 4
ranks) and reading off which chunk held the correct sum; fixed by
correcting the slot convention in `AllGather` and the doc comments in both
`ring_allreduce.h` and `collectives.h`. Left in this README because it's
exactly the kind of off-by-one that's cheap to catch locally and expensive
to debug against real EFA hardware.

## Results
TODO: run on multi-node setup — latency/bandwidth for each op at various
message sizes and world sizes.

## Hardware notes
- Builds and runs anywhere (validated on Mac). Real bandwidth numbers need
  2+ Linux nodes with a real network.
