# ZeRO-Infinity

**Status: code-complete AND locally run — portable, no CUDA/Linux
dependency for the scheduling logic and cost model (see Design for what
this does and does not validate).**

## What this measures

PLAN.md Phase 6 step 10: CPU RAM offload, NVMe offload, measure throughput
degradation vs. memory capacity gain. Two genuinely different things get
validated separately here, matching the design comment at the top of
`offload_scheduler.h`:
1. The double-buffered prefetch **scheduling logic** — real, tested code.
2. An **analytical cost model** relating compute time, transfer time, and
   overlap (same spirit as `compiler/cost_model/`) — real arithmetic, but
   with illustrative, not measured, bandwidth inputs.

## Design

`run_double_buffered` executes the actual schedule (transfer shard 0 to
fill the pipeline, then for each shard prefetch the NEXT shard's transfer
before computing the CURRENT one) against real callbacks, and the test
verifies every shard is computed exactly once, in order, with transfer(i+1)
always issued strictly before compute(i) — the prefetch-ahead property
that is the entire reason overlap works. This holds regardless of what the
callbacks do or what device they target — real hardware would run
transfer_fn as an async DMA on one engine and compute_fn as a kernel launch
on another, letting them genuinely execute concurrently; this Mac test
proves the SCHEDULE is right, not that it survives real device dispatch.

`simulate_offload_schedule` is the accompanying cost model: naive
(transfer-then-compute, serialized) total time vs. double-buffered
(pipeline-fill transfer, then `num_shards` stages of `max(compute,
transfer)`) — the standard software-pipelining formula. Evaluated across
three illustrative regimes to show the formula behaves as pipelining
theory predicts: overlap helps most when compute and transfer are
comparable, and helps only marginally when one dominates (still bound by
whichever is larger, same as any pipeline).

## Sanity-run output (Mac, 2026-07-21)

```
test 1 (double-buffered schedule correctness): PASS

test 2 (cost model across regimes, 8 shards, illustrative inputs -- see README):
  compute-bound (fast NVMe/RAM)    compute=10.0ms transfer=2.0ms -> naive=96.0ms overlapped=82.0ms speedup=1.17x
  balanced (compute ~= transfer)   compute=5.0ms transfer=5.0ms -> naive=80.0ms overlapped=45.0ms speedup=1.78x
  transfer-bound (slow NVMe)       compute=2.0ms transfer=10.0ms -> naive=96.0ms overlapped=90.0ms speedup=1.07x
PASS
```

The balanced regime gets the largest speedup (1.78x) — exactly what
classic software-pipelining predicts: overlap has the most slack to
exploit when neither compute nor transfer dominates the other. The
transfer-bound regime (the one ZeRO-Infinity's NVMe tier is realistically
in, at real model scale) only gets 1.07x — overlap cannot make training
faster than the storage tier's bandwidth allows, only hide compute time
inside whatever transfer time is unavoidable.

## Results
TODO: run on GPU hardware — the compute_ms/transfer_ms inputs above are
illustrative round numbers, not measurements. Real numbers need an actual
optimizer-state shard size, real PCIe (CPU RAM offload) and NVMe bandwidth,
and real per-shard compute time from a GPU kernel.

| Offload tier | Shard size | Bandwidth | Compute/shard | Naive throughput | Overlapped throughput | Max model size enabled |
|---------------|------------|-----------|-----------------|--------------------|--------------------------|---------------------------|
| CPU RAM (PCIe) | TODO | TODO | TODO | TODO | TODO | TODO |
| NVMe | TODO | TODO | TODO | TODO | TODO | TODO |

## Hardware notes
- This step: none required for the scheduling-logic correctness check.
- Real bandwidth/throughput numbers (Results table): GPU instance with
  NVMe storage, real model at a scale where GPU memory is genuinely
  insufficient without offload.
