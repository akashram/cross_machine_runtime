# Ring All-Reduce

**Status: code-complete AND locally run — portable, built on
`networking/common::Channel`.**

## What this measures
Ring all-reduce implemented from scratch. Bandwidth efficiency vs theoretical
maximum (2*(N-1)/N * message_size / bandwidth). Compare against NCCL baseline.

## Design
Two phases, `2*(world_size-1)` rounds total (`ring_allreduce.cpp`):
reduce-scatter (each rank ends up holding the fully-reduced sum for one
`1/N` slice) then all-gather (that slice circulates so every rank ends up
with the full buffer) — the standard bandwidth-optimal ring algorithm.
The one non-obvious part is deadlock avoidance: `TcpChannel` shares one
bidirectional socket per peer pair, so for `world_size == 2` a rank's
"left" and "right" neighbor are the same peer, and naive send-then-recv on
both ends deadlocks once a chunk exceeds the kernel socket buffer. Fixed
with the standard trick — even ranks send before receiving, odd ranks
receive before sending — which provably can't cycle around the ring (see
the comment in `ring_allreduce.cpp` for the trace).

## Sanity-run output (Mac, loopback, 2026-07-19)

`ring_allreduce_test`: 4 ranks, 1M floats (4MB) each, verifies every rank
converges on the correct element-wise sum, reports effective bandwidth:

```
rank 0: correct, effective bandwidth 0.100 GB/s
rank 1: correct, effective bandwidth 0.100 GB/s
rank 2: correct, effective bandwidth 0.103 GB/s
rank 3: correct, effective bandwidth 0.102 GB/s
PASS
```

~0.1 GB/s is loopback-TCP-through-threads overhead, not a real network
number — the point of this run is correctness (every rank agrees), not
throughput. Real bandwidth numbers need the multi-node setup below.

## Results
TODO: run on multi-node setup.

| Nodes | Message size | Our bandwidth (GB/s) | NCCL (GB/s) | % of theoretical |
|-------|-------------|---------------------|-------------|-----------------|
| 2 | 1 MB | TODO | TODO | TODO |
| 4 | 256 MB | TODO | TODO | TODO |
| 8 | 1 GB | TODO | TODO | TODO |

## Hardware notes
- Required: 2+ Linux nodes with high-bandwidth interconnect
