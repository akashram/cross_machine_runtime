# Recursive Halving-Doubling All-Reduce

**Status: code-complete AND locally run — portable, builds and runs everywhere including this Mac.**

## What this measures
Recursive halving-doubling all-reduce (Rabenseifner's algorithm), compared
against ring for small vs. large message sizes, documenting the crossover
point.

## Design
Two phases, `log2(world_size)` steps each (`halving_doubling.cpp`):
recursive-halving reduce-scatter (each step, split the currently-owned
range in half with a partner at halving distance, keep/reduce one half,
forward the other — applying "lower rank keeps lower half" consistently
means rank `r` ends up owning the same `r`-th `1/P` slice
`ring_allreduce.cpp` uses) then recursive-doubling all-gather (the mirror:
start with that small slice and double its extent each step). Same total
bytes moved as ring all-reduce, but `O(log P)` round trips instead of
`O(P)` — wins when per-round-trip latency dominates (small messages),
loses to ring once bandwidth dominates (large messages). Requires
`world_size` to be a power of 2 (the classic algorithm's precondition) and
`count` divisible by `world_size` (sidesteps off-by-one slice-size
bookkeeping for non-divisible counts — a real simplification, not a
correctness gap for the power-of-2 world sizes this project actually
deploys at); falls back to `ring_allreduce` for non-power-of-2 world sizes
rather than implementing the more complex general variant.

## Sanity-run output (Mac, loopback, 2026-07-19)

`halving_doubling_test`: 4 ranks, 1M floats (4MB) each, same correctness
check as `ring_allreduce_test`:

```
rank 0: correct, effective bandwidth 0.804 GB/s
rank 1: correct, effective bandwidth 0.825 GB/s
rank 2: correct, effective bandwidth 1.053 GB/s
rank 3: correct, effective bandwidth 0.847 GB/s
PASS
```

Faster than `ring_allreduce_test`'s loopback numbers (~0.1 GB/s) at this
same size/rank count — expected, since 4MB is well into the range where
`O(log P) = 2` round trips beats `O(P) = 3` round trips' overhead on a
loopback transport where per-round-trip fixed cost dominates. This is
*not* evidence about the real crossover point on an actual network — it's
a sanity check that both algorithms are correct and this one isn't
pathologically slow.

## Results
TODO: run on multi-node setup and find the real message-size crossover.

| Message size | Ring bandwidth (GB/s) | Halving-doubling bandwidth (GB/s) | Winner |
|--------------|------------------------|-------------------------------------|--------|
| 1 KB | TODO | TODO | TODO |
| 64 KB | TODO | TODO | TODO |
| 1 MB | TODO | TODO | TODO |
| 256 MB | TODO | TODO | TODO |

## Hardware notes
- Builds and runs anywhere (validated on Mac). Real crossover measurement
  needs 2+ Linux nodes with a real network (loopback has no meaningful
  latency-vs-bandwidth tradeoff to observe).
