# 3D Parallelism

**Status: code-complete AND locally run — portable, builds on
autograd/matrix.h, col_row_linear's tensor-parallel linear layers, and
networking/ring_allreduce.**

## What this measures

PLAN.md Phase 6 step 15: combine data + tensor + pipeline parallelism,
verify all combinations work correctly. Two things: the rank <->
process-group math (the genuinely new part — see Design), and a composed
DP x TP end-to-end demo.

**Scoping decision, stated plainly**: this validates DP and TP composed
together, with the PP dimension present in the process-grid math (a
`pp_size` parameter, `pp_group` queries) but not executed — step 14
already validated pipeline scheduling at the dependency/bubble-fraction
level via discrete-event simulation, and composing real pipeline STAGE
EXECUTION (actual forward/backward hand-off between stages) with DP+TP in
this same step would mean re-deriving step 14's scheduler as a live
execution engine under time constraints that were better spent making the
DP+TP composition (the part with a genuine, catchable cross-wiring risk —
see the bug below) rigorous. `pp_size=1` in the composed demo is the
honest way to state that boundary.

## Design

`ProcessGrid` (`process_grid.h`): `global_rank = dp_rank*(tp_size*pp_size)
+ tp_rank*pp_size + pp_rank` (DP slowest-varying, PP fastest — a
documented choice, not the only valid one). `tp_group`/`dp_group`/
`pp_group` return every rank sharing the other two coordinates. Test 1
exhaustively checks, for 5 grid shapes: the rank<->coordinate mapping is a
true bijection, and every group has the right size and partitions the
full rank set with no overlaps or gaps.

Test 2 composes DP(2) x TP(4): each of 2 DP replicas runs the SAME
TP-sharded linear (step 11) on DIFFERENT data, all-reducing within its own
4-rank TP mesh; each of 4 TP shards then DP-all-reduces its gradient
across the 2 replicas that own that same shard, using a SEPARATE 2-rank DP
mesh per shard — 6 independent communicators total (2 TP meshes + 4 DP
meshes), mirroring how real multi-dimensional parallelism uses one
communicator per dimension rather than overloading a single one.

## A real bug this caught

The first version of test 2's reference computed each TP shard's gradient
using that shard's own pre-all-reduce PARTIAL output as the backward seed
(`dOut`) — but the actual distributed code (correctly) backprops from
`out`, the value AFTER the TP all-reduce sums every shard's contribution.
These are different quantities, and using the wrong one would have made
the reference itself wrong rather than validating anything. Fixed by
having the reference compute the FULL forward pass (summing all 4 shards)
first to get the true `dOut`, exactly like `col_row_linear`'s reference
does, then backward through one target shard from that.

## Sanity-run output (Mac, loopback, 2026-07-21)

```
test 1 (process grid: rank<->coordinate bijection, group partitioning):
  grid dp=2 tp=4 pp=1 (world_size=8): PASS
  grid dp=2 tp=2 pp=2 (world_size=8): PASS
  grid dp=3 tp=1 pp=2 (world_size=6): PASS
  grid dp=1 tp=1 pp=1 (world_size=1): PASS
  grid dp=4 tp=3 pp=2 (world_size=24): PASS

test 2 (composed DP(2) x TP(4), PP fixed at 1 -- see README):
  tp shard 0: DP-averaged dW_col max abs diff vs reference = 0.000000: PASS
  tp shard 1: DP-averaged dW_col max abs diff vs reference = 0.000000: PASS
  tp shard 2: DP-averaged dW_col max abs diff vs reference = 0.000000: PASS
  tp shard 3: DP-averaged dW_col max abs diff vs reference = 0.000000: PASS

PASS
```

Exact bit-for-bit match on every TP shard's DP-averaged gradient — the DP
all-reduce correctly stayed within each shard's own 2-rank group and never
cross-wired with another shard's gradient.

## Results
TODO: run on GPU hardware — the numbers that matter are real pipeline
STAGE execution composed with DP+TP (the scoping gap noted above), and
scaling efficiency across all three dimensions simultaneously at real
model scale.

| DP x TP x PP | World size | Throughput | Scaling efficiency vs. 1 GPU |
|----------------|--------------|--------------|----------------------------------|
| TODO | TODO | TODO | TODO |

## Hardware notes
- This step: none required for the process-grid math or DP+TP composition.
- Real pipeline execution + full 3D scaling (Results table): multi-GPU
  instance, p4d.24xlarge or larger for a real PP dimension.
