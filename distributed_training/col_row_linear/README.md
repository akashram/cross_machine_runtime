# Column/Row-Parallel Linear Layers

**Status: code-complete AND locally run — portable, builds on
autograd/matrix.h and networking/ring_allreduce.**

## What this measures

PLAN.md Phase 6 step 11: tensor-parallel linear layers, column-wise split
for the first linear, row-wise split for the second, validated
numerically. Implemented as the standard Megatron-LM pairing: a
column-parallel layer feeding directly into a row-parallel layer needs
only ONE all-reduce total in the forward pass (on the row-parallel
layer's output) rather than one after each layer — see Design for the
communication pattern this relies on.

## Design

The classic Megatron MLP block, hand-derived forward/backward (not new
autograd.h Tensor ops — see the design comment at the top of
`tensor_parallel_linear.h` for why a fused custom Function is the right
call here, same as real ML systems do for this exact pattern):
`ColumnParallelLinear` (splits weight by output columns; forward needs
no communication since the input is already replicated) -> ReLU
(elementwise, still no communication) -> `RowParallelLinear` (splits
weight by input rows; forward needs one all-reduce SUM to combine each
rank's partial output). This pairing needs exactly one communication op
per direction: one all-reduce in the forward pass (row-parallel's output),
one in the backward pass (column-parallel's dX) — not one after every
layer.

The row-parallel layer's bias is deliberately **replicated**, not sharded:
sharding it would save O(out) memory at the cost of another communication
op, for a bias vector that is already negligible next to the O(in*out)
weight matrix it sits next to.

## Sanity-run output (Mac, loopback, 2026-07-21)

4 simulated tensor-parallel ranks (real TCP loopback threads), hidden dim
16 split into 4 shards of 4, vs. a single-process reference with the full,
unsharded weights and a hand-derived relu-MLP backward:

```
  forward y (rank 0)       max abs diff = 0.000000: PASS
  dW_col (reassembled)     max abs diff = 0.000000: PASS
  db_col (reassembled)     max abs diff = 0.000000: PASS
  dW_row (reassembled)     max abs diff = 0.000000: PASS
  d(row_bias) (rank 0)     max abs diff = 0.000000: PASS
  dX (rank 0, all-reduced) max abs diff = 0.000000: PASS
PASS
```

Exact bit-for-bit match on every gradient, including the all-reduced dX —
tighter than steps 3/7-9's SGD/Adam comparisons because there is no
per-step accumulation of ring-summation-order float drift here: this test
checks one forward+backward pass, not many training steps compounding
small differences.

## Results
TODO: run on GPU hardware — the number that matters is communication
overhead (one all-reduce per direction) as a fraction of compute time at
real hidden-dimension scale, and scaling efficiency as tensor-parallel
world size grows.

| Hidden dim | TP world size | Compute time | All-reduce time | Overhead % |
|------------|----------------|---------------|-------------------|-------------|
| TODO | TODO | TODO | TODO | TODO |

## Hardware notes
- This step: none required.
- Overhead/scaling validation (Results table): multi-GPU instance with
  NVLink/EFA for the all-reduce.
