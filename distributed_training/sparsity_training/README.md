# 2:4 Structured Sparsity Training

**Status: code-complete AND locally run — portable, builds on autograd/.
Training-time convergence fully validated; real 2:4 Tensor Core throughput
is separately hardware-gated (Phase 3's `gpu_engine/sparsity/`) — this
step's own scope is the training-time question, not throughput.**

## What this measures

PLAN.md Phase 6 step 21: prune a weight matrix to 2:4 during training,
validate convergence, measure accuracy impact. The "measure 2x throughput"
half of PLAN.md's line is about real sparse Tensor Core execution, already
covered (hardware-gated, code-complete) by Phase 3's `gpu_engine/sparsity/`
— nothing new to build there; this step is specifically about whether the
model still learns with the mask applied, which is genuinely CPU-testable.

## Design

`compute_2_4_mask` groups along the weight's ROW axis (the matmul's K /
reduction dimension for `W[in x out]` used as `x @ W`) — not the output
axis — because that is the axis real sparse Tensor Cores actually skip
zeros along; grouping the wrong axis would validate a pattern that
doesn't correspond to any real hardware speedup. Recipe: train dense for
a warmup period, prune ONCE to 2:4 (NVIDIA's documented ASP recipe, not
re-pruning from scratch every step), then fine-tune with the mask
re-applied to the GRADIENT every step so pruned entries never drift from
exactly zero (masking only the weight, not the gradient, would let
gradient updates slowly un-prune them).

## Sanity-run output (Mac, 2026-07-21)

Toy MLP (4 -> 16 -> 3), 3-class classification, first layer's weight
(`[4 x 16]`, one group of 4 rows per output column) pruned to 2:4 after 80
dense warmup epochs, then 120 epochs of sparse fine-tuning:

```
after dense warmup (80 epochs): loss=0.0557 accuracy=100.0%
2:4 property immediately after pruning: PASS
after sparse fine-tune (120 epochs): loss=0.0165 accuracy=100.0%
2:4 property still holds after fine-tuning: PASS
sparsified model still learns well (accuracy >= 85%): PASS
PASS
```

Accuracy stays at 100% straight through pruning and fine-tuning, and loss
actually improves further (0.0557 -> 0.0165) despite 50% of the first
layer's weights being permanently pinned to zero — the surviving 50%
still have enough capacity for this toy problem, and the 2:4 property
provably holds throughout (verified structurally, not just trusted from
the pruning step).

## Results
TODO: run on GPU hardware — the number that matters is real
`cusparseLtMatmul` throughput vs. the dense baseline (already the scope
of `gpu_engine/sparsity/`'s own Results table) and accuracy impact at real
model scale, where 50% structured sparsity may cost more accuracy than it
did on this over-parameterized toy problem.

| Model size | Dense accuracy | 2:4 sparse accuracy (after fine-tune) | Accuracy delta |
|--------------|-------------------|------------------------------------------|---------------------|
| TODO | TODO | TODO | TODO |

## Hardware notes
- This step: none required for training-time convergence validation.
- Real throughput: see `gpu_engine/sparsity/README.md` (Phase 3).
- Real-model-scale accuracy impact (Results table): GPU instance.
