# Data Parallel Training (baseline)

**Status: code-complete AND locally run — portable, built on
`networking/ring_allreduce`.**

## What this measures

PLAN.md Phase 6 step 3: manual gradient all-reduce after backward pass,
validated against a single-process ("single-GPU") baseline. `linreg.h` is a
closed-form linear regression (MSE loss, hand-derived gradient — no
autograd yet, that's step 6). Full-batch gradient descent, not mini-batch:
the point of this step is validating the *distributed mechanics*, and
full-batch keeps the comparison exact (the data-parallel gradient after
all-reducing per-shard sums is mathematically the same number as the
single-process full-batch gradient — no stochasticity to launder a mismatch
into looking like noise).

## Design

4 simulated ranks (real TCP loopback threads, `networking::TcpChannel`),
each owning a contiguous 1/4 shard of the same synthetic dataset (400
samples, 8 features, known `w_true` — see `linreg.h`). Each step: compute
local gradient **sum** (not mean) on the shard, `ring_allreduce` it (sums
across ranks), divide by the *global* sample count, apply identical SGD
update on every rank. Dividing by the global count after summing — not
dividing by the shard count before summing — is what makes this correct
for uneven shards in general, even though this test's shards happen to be
equal-sized.

## Sanity-run output (Mac, loopback, 2026-07-21)

```
step  baseline_loss  data_parallel_loss  rel_diff
   0       6.013300            6.013300   0.00000
  10       4.029556            4.029556   0.00000
  20       2.712640            2.712640   0.00000
  30       1.834297            1.834296   0.00000
  40       1.245779            1.245778   0.00000
final loss: baseline=0.882671 data_parallel=0.882671
PASS
```

The 4-rank data-parallel loss curve tracks the single-process baseline to
6 decimal places — the ring all-reduce's different summation order (partial
sums combined over rounds, vs. one loop over all samples) introduces only
float32-rounding-level divergence, exactly as expected.

## Results
TODO: run on multi-GPU hardware — real per-step wall-clock time and
scaling efficiency (1 -> 8 GPUs) vs. this correctness-only Mac run.

| GPUs | Throughput (samples/s) | Scaling efficiency |
|------|------------------------|---------------------|
| 1 | TODO | — |
| 8 | TODO | TODO |

## Hardware notes
- This step: none required — runs anywhere with threads and a loopback socket.
- Scaling-efficiency validation (Results table): multi-GPU instance.
