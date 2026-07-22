# Minimal Transformer

**Status: code-complete AND locally run — portable, builds on
distributed_training/autograd, tensor_parallel_attn, and seq_parallel. Not
part of PLAN.md's original 12 phases — added specifically to give
distributed_training/'s RLHF steps (22-25) a real model to train against,
instead of stubbing those steps or reducing "language model" to a toy
single-step classifier.**

## What this is

A minimal decoder-only transformer (token + positional embedding, N
pre-LN blocks — causal multi-head self-attention + residual, MLP +
residual — final LayerNorm, output projection to vocab logits) and a
character-level tokenizer, both real and complete: real causal masking,
real multi-head attention, real backprop through the entire stack,
hand-derived at the Matrix level (`matrix.h`) rather than through
`autograd.h`'s generic Tensor tape — the same "fused custom Function"
pattern already used for `col_row_linear/`, `tensor_parallel_attn/`, and
`seq_parallel/`, composed here into a full model instead of one op.

No BPE tokenizer training pipeline (character-level vocabulary only,
built directly from a corpus's distinct bytes) and no batching (one
sequence per forward call) — both stated scope choices, not shortcuts
taken silently: BPE training is a separate, large undertaking or its own
right that nothing here needs to be a real tokenizer; batching would
multiply this file's hand-derived backward bookkeeping without changing
what it validates.

## Design

Reuses already-validated pieces rather than re-deriving them:
`seq_parallel::layernorm_forward/backward` for both LayerNorms in each
block, and `tensor_parallel_attn::single_head_attention_forward/backward`
for every attention head, with a causal mask (upper triangle set to
`-1e9`) applied to the score matrix before softmax in
`causal_attention_forward`. The backward formula is UNCHANGED from the
unmasked version in `tensor_parallel_attn/attention.h` — the mask's
effect is already fully captured in the cached post-mask softmax output,
which is all `single_head_attention_backward` reads.

## Sanity-run output (Mac, 2026-07-21)

Test 1: gradient check (central finite differences, median relative
error over 8 sampled elements per parameter — median, not max, for the
same reason as `autograd/autograd_test.cpp` and
`seq_parallel/seq_parallel_test.cpp`: this model has both ReLU and
LayerNorm, either of which can produce a rare finite-difference outlier
at a kink without indicating a wrong formula), covering every distinct
parameter type: embedding tables, attention weights, MLP weights,
LayerNorm affine params, output projection.

Test 2: an actual training run — 400 steps on the 20-character corpus
`"the quick fox jumps "`, then greedy autoregressive generation from the
first character.

```
  token_emb      median relative error (8 samples) = 0.000000
  pos_emb        median relative error (8 samples) = 0.001673
  block0.wq      median relative error (8 samples) = 0.000609
  block0.w1      median relative error (8 samples) = 0.004829
  block1.wo      median relative error (8 samples) = 0.011208
  block1.gamma2  median relative error (8 samples) = 0.004662
  final_gamma    median relative error (8 samples) = 0.002885
  w_out          median relative error (8 samples) = 0.000703
test 1 (gradient check): PASS
training: loss 3.1891 -> 0.0171
generated: "the quick fox jumps "
expected:  "the quick fox jumps "
test 2 (trains and greedy-generates the training corpus exactly): PASS
PASS
```

Every distinct backward path (embedding accumulation, causal attention,
both LayerNorms, both residual branches, the MLP, the output projection)
checks out numerically, AND the composed whole trains a real model that
greedy-generates its training corpus back exactly, character for
character — the strongest form of end-to-end validation available without
real hardware.

## Results
TODO: run on GPU hardware for anything at real model scale (real vocab
size via a real tokenizer, real `d_model`/depth, batched sequences,
perplexity on held-out text) — this Mac run validates correctness at toy
scale, not capability at real scale.

| Config | Params | Corpus | Final loss | Generation match |
|--------|--------|--------|-------------|----------------------|
| d_model=16, 2 layers, 2 heads | ~5K | 20 chars | 0.017 | exact |
| Real scale | TODO | TODO | TODO | TODO |

## Hardware notes
- This component: none required.
- Real-scale validation (Results table): GPU instance, real tokenizer.
