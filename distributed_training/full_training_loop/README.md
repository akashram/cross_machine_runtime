# Full Training Loop

**Status: code-complete AND locally run — portable, composes autograd/,
grad_clipping/, zero1/, and checkpoint/. Latency breakdown captured, but
reflects this Mac's loopback/thread overhead, not a GPU-representative
profile — see below.**

## What this measures

PLAN.md Phase 6 step 20: forward -> backward -> grad sync -> ZeRO
optimizer step -> checkpoint, with a latency breakdown per phase to
identify the bottleneck.

## Design

One real, coherent loop, 4 simulated data-parallel ranks, composing
almost everything built so far: forward+backward through the toy MLP
(autograd/, step 6) -> global gradient norm + clip (grad_clipping/, step
5) -> gradient all-reduce (ring_allreduce, step 3's pattern) -> ZeRO-1
optimizer step (zero1/, step 7) -> every 10 steps, an async sharded
checkpoint write (checkpoint/, step 17). Each phase is timed with
`std::chrono::steady_clock` on every step; the breakdown at the end
averages each phase across all ranks and steps.

## Sanity-run output (Mac, loopback, 2026-07-21)

Toy MLP (2 -> 16 -> 3), 3-class classification, 30 steps:

```
training: loss 2.6317 -> 0.0008

latency breakdown (mean across 4 ranks, 30 steps total):
  forward_backward     12.061 ms  ( 27.0%)
  grad_clip            16.502 ms  ( 37.0%)
  grad_sync             9.405 ms  ( 21.1%)
  optimizer_step         6.377 ms  ( 14.3%)
  checkpoint             0.245 ms  (  0.5%)
  total                44.589 ms  (100.0%)
bottleneck phase: grad_clip
final checkpoint restore: read 25 floats from /tmp/.../rank0_epoch29.bin

PASS
```

Training converges cleanly (loss 2.63 -> 0.0008), and the final checkpoint
restores correctly.

**Read the bottleneck finding for what it actually is**: `grad_clip` (37%)
beats `forward_backward` (27%) here specifically because clipping issues
its OWN one-scalar all-reduce round-trip (`global_grad_norm`, see
`grad_clipping/README.md`'s note on unpacked round-trips) on top of
`grad_sync`'s full-gradient all-reduce — on a toy model this small, TWO
loopback-TCP round-trips (clip's + sync's) cost more wall-clock time than
the actual compute, which is backwards from what a real model would show
(where forward/backward FLOPs dominate and an extra scalar round-trip is
noise). This is a genuine, useful finding about round-trip-count
sensitivity at small scale — exactly the kind of thing
`sync_batchnorm/README.md` flagged as a real, separate follow-up
(packing multiple all-reduces into one call) — but it is a statement
about THIS toy model's scale and THIS Mac's loopback overhead, not a
representative GPU training profile.

## Results
TODO: run on GPU hardware — the real bottleneck at real model scale (where
forward/backward compute is the dominant FLOP cost, not TCP round-trip
count) needs a real model and real interconnect to measure honestly.

| Model size | World size | forward_backward | grad_clip | grad_sync | optimizer_step | checkpoint | Bottleneck |
|--------------|--------------|---------------------|-------------|-------------|-------------------|--------------|---------------|
| TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO |

## Hardware notes
- This step: none required for correctness (convergence + checkpoint
  restore).
- Representative latency breakdown (Results table): multi-GPU instance,
  real model scale where compute dominates round-trip count.
