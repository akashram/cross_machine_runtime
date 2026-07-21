# Tensor-Parallel Attention

**Status: code-complete AND locally run — portable, builds on
autograd/matrix.h and networking/ring_allreduce.**

## What this measures

PLAN.md Phase 6 step 12: split attention heads across tensor-parallel
ranks, validated numerically against a single-process reference.

## Design

Splitting by HEAD is what makes attention embarrassingly parallel:
`softmax(QK^T/sqrt(d))V` for one head is independent of every other head,
so a rank owning a subset of heads needs **zero communication for the
attention math itself** (`attention.h`'s `single_head_attention_forward`/
`backward` — the standard scaled-dot-product forward and its hand-derived
backward, including the softmax Jacobian-vector product). Communication
only appears at the same two places step 11 already validated: the Q/K/V
projections are column-parallel (split by output columns = which heads a
rank owns, no forward communication), the output projection is
row-parallel (one all-reduce SUM in forward, one for dX in backward).

Simplified vs. real Megatron for this validation (stated plainly, also in
`attention.h`): Q/K/V use three separate weight matrices rather than one
fused matrix (mathematically equivalent — three column-parallel
projections instead of one wider one split three ways internally), and
there is no causal/padding mask (orthogonal to what is being validated —
gradient correctness of the tensor-parallel split — and unaffected by it).

## Sanity-run output (Mac, loopback, 2026-07-21)

4 simulated tensor-parallel ranks, 8 attention heads (2/rank), hidden dim
16, sequence length 5, vs. a single-process reference with full,
unsharded Wq/Wk/Wv/Wout:

```
  forward out      max abs diff = 0.000000: PASS
  dWq              max abs diff = 0.000000: PASS
  dWk              max abs diff = 0.000000: PASS
  dWv              max abs diff = 0.000000: PASS
  dWout            max abs diff = 0.000000: PASS
  dX               max abs diff = 0.000000: PASS
PASS
```

Exact bit-for-bit match on the first run — same structure as step 11's
result, and further evidence the hand-derived attention backward
(including the softmax Jacobian-vector product) is correct, not just the
tensor-parallel plumbing around it.

## Results
TODO: run on GPU hardware — the number that matters is attention
communication overhead (one all-reduce per direction, same as step 11) at
real head-count/sequence-length scale, plus interaction with Flash
Attention (Phase 3, `gpu_engine/flash_attn/`) once both are combined.

| Heads | TP world size | Seq length | Compute time | All-reduce time | Overhead % |
|-------|-----------------|-------------|---------------|-------------------|-------------|
| TODO | TODO | TODO | TODO | TODO | TODO |

## Hardware notes
- This step: none required.
- Overhead validation (Results table): multi-GPU instance with NVLink/EFA.
