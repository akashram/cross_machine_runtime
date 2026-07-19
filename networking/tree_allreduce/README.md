# Tree All-Reduce

**Status: code-complete AND locally run — portable, builds and runs everywhere including this Mac.**

## What this measures
Binomial-tree all-reduce: reduce-to-root then broadcast. Measured, three-way
compared against ring and halving-doubling with topology analysis.

## Design
`tree_reduce_to_root` + `tree_broadcast` (`tree_allreduce.cpp`), both using
rank *relative to root* so the same bit-mask logic works for any root, not
just 0. `O(log P)` round trips like halving-doubling (step 12), but each
round trip moves the **entire** buffer rather than a shrinking/growing
slice — `O(count * log P)` bandwidth vs. ring/halving-doubling's
`O(count)`. Included specifically for that tradeoff: at small message
sizes where round-trip latency dominates, tree's simplicity can still win
despite worse bandwidth scaling; at large sizes it loses badly. No
explicit send-before-recv ordering trick is needed here (unlike ring/
halving-doubling) — every round's pair roles are structurally asymmetric
(whoever's bit is set sends, the other unconditionally receives), so
there's no round where both sides might pick the same order and race.
`tree_broadcast` is reused directly by `collectives/` (step 14) as its
`Broadcast` primitive — no reason to write binomial broadcast twice.

## Sanity-run output (Mac, loopback, 2026-07-19)

`tree_allreduce_test`: correctness check at world_size=4 (power of 2) and
world_size=5 (non-power-of-2, exercises the bounds checks in both
functions — tree all-reduce, unlike halving-doubling, doesn't need a
power-of-2 fallback):

```
world_size=4: PASS
world_size=5: PASS
PASS
```

## Results
TODO: run on multi-node setup — three-way bandwidth comparison vs. ring
and halving-doubling across message sizes, plus topology analysis (does
the binomial tree's shape match physical rack/switch topology, or does it
create cross-rack hot links a topology-aware tree would avoid — see
`topo_scheduler/`, step 16).

| Message size | Ring (GB/s) | Halving-doubling (GB/s) | Tree (GB/s) | Winner |
|--------------|-------------|--------------------------|-------------|--------|
| 1 KB | TODO | TODO | TODO | TODO |
| 1 MB | TODO | TODO | TODO | TODO |
| 256 MB | TODO | TODO | TODO | TODO |

## Hardware notes
- Builds and runs anywhere (validated on Mac). Real bandwidth comparison
  needs 2+ Linux nodes with a real network.
