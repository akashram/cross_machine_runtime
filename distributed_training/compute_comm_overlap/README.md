# Compute/Communication Overlap

**Status: code-complete AND locally run — portable, builds on
autograd/matrix.h, data_loading's PrefetchQueue, and
networking/ring_allreduce. Correctness fully validated; wall-clock
speedup is not reliably measurable at this scale on this dev machine —
see below, same honest pattern as checkpoint/.**

## What this measures

PLAN.md Phase 6 step 18: double-buffer backward pass with all-reduce —
while computing layer N's gradient, communicate layer N+1's. Real
backprop's layer-by-layer sequential structure is what makes this
possible: by the time layer N's gradient is ready, layer N+1's all-reduce
(kicked off a step earlier) has had a whole layer's compute time to run in
the background.

## Design

`overlapped_backward` (`overlap_backward.h`) uses a **single dedicated
communication thread** draining a queue, not concurrent all-reduces from
multiple threads — deliberately, not incidentally. `TcpChannel` is one
socket per rank pair; two threads issuing all-reduces on the same channel
at once would interleave bytes on the wire and corrupt the stream. A
single comm thread while compute continues on the caller's thread is both
the correct way to avoid that and the realistic shape of the real
technique (a dedicated communication stream, not communication happening
wherever compute finishes). Reuses `data_loading::PrefetchQueue` as the
handoff — it was written generically enough (bounded, blocking,
producer/consumer) to be exactly the right queue here too.

`serial_backward` is the same math with nothing overlapped — used to prove
overlap changes only WHEN communication happens, never the result.

## Sanity-run output (Mac, loopback, 2026-07-21)

4 simulated data-parallel ranks, 4-layer linear chain, real sequential
per-layer gradient computation:

```
  rank 0, layer 0: overlapped vs serial max abs diff = 0.000000
  rank 0, layer 1: overlapped vs serial max abs diff = 0.000000
  rank 0, layer 2: overlapped vs serial max abs diff = 0.000000
  rank 0, layer 3: overlapped vs serial max abs diff = 0.000000
test 1 (overlap produces identical results to serial, all ranks): PASS
informational timing (not asserted -- see file comment): overlapped=0.0015s serial=0.0015s
PASS
```

Exact bit-for-bit match on every layer, every rank. The informational
timing shows essentially no difference between overlapped and serial —
expected and reported honestly, not asserted as a pass/fail condition (see
`checkpoint/README.md` for the fuller account of why wall-clock overlap
benefit is unreliable to measure at this scale on this dev machine): a
4-layer toy model's per-layer gradient computation and loopback all-reduce
are both far too fast (microseconds) for double-buffering to have
meaningful slack to exploit, and thread-scheduling overhead on 2 physical
cores dominates whatever there is.

## Results
TODO: run on GPU hardware — the number that matters is throughput
improvement at real layer count/size and real network latency, where
compute-per-layer (milliseconds) genuinely dwarfs the queue/thread-
handoff overhead this Mac run is dominated by.

| Layers | Compute time/layer | Comm time/layer | Serial throughput | Overlapped throughput | Improvement |
|--------|----------------------|--------------------|------------------------|----------------------------|----------------|
| TODO | TODO | TODO | TODO | TODO | TODO |

## Hardware notes
- This step: none required for correctness validation.
- Throughput improvement (Results table): multi-GPU instance with
  NVLink/EFA, real model layer sizes.
