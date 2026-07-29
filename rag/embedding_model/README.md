# embedding_model

**Status: code-complete AND locally run — pure CPU, no external dependency.**

## What this measures

PLAN.md Phase 13 step 1: an embedding model that turns a token sequence
(query or document) into a fixed-size vector, trained so that a query's
embedding lands close to its matching document's embedding and far from
unrelated documents' embeddings — the retrieval half of RAG.

## Design

- **Architecture**: token + positional embedding -> N pre-LN blocks
  (bidirectional multi-head self-attention + residual, MLP + residual) ->
  final LayerNorm -> mean pooling over sequence positions -> linear
  projection (no bias) -> L2 normalize. Same block shape as
  `transformer/transformer_model.h`, but with **unmasked** attention
  (`distributed_training::single_head_attention_forward`, not
  `transformer`'s causal variant) — an embedding encoder must see the
  whole sequence in both directions, unlike a decoder. Reuses
  `seq_parallel`'s LayerNorm and `tensor_parallel_attn`'s attention
  primitive directly, same convention as `transformer/`.
- **Why not reuse `transformer::block_forward` directly**: it hard-codes
  causal masking inside the function body rather than taking the
  attention function as a parameter. Parameterizing it would touch a file
  `distributed_training`'s RLHF steps 22-25 already gradient-check and
  depend on, for a one-line difference; duplicating the (much shorter)
  block here was the smaller-blast-radius choice.
- **Siamese weight sharing**: one `EncoderParams` encodes both queries and
  documents — not two separate towers. A gradient step accumulates
  contributions from both the query-tower and document-tower backward
  passes into the same parameter set (`accumulate_encoder_grad`).
- **Loss**: symmetric InfoNCE (Oord et al. 2018; symmetric
  query<->document formulation as in Radford et al. 2021's CLIP) over
  in-batch negatives. Embeddings are L2-normalized, so their dot product
  is exactly cosine similarity; `diag_softmax_ce` is the same softmax
  cross-entropy shape as `transformer_model.h`'s `next_token_loss`, but
  the target for row `i` is always the row index itself (its true
  positive), not a looked-up token id.
- **Mean pooling and L2-normalize are parameter-free** — their backward
  is plain calculus (spread gradient evenly across positions; project out
  the radial component for the unit-norm constraint), not another learned
  layer.

## Results (captured 2026-07-29, Apple clang 14 / `-std=c++2b`, this Mac)

```
  token_emb      median relative error (8 samples) = 0.007773
  pos_emb        median relative error (8 samples) = 0.007984
  block0.wq      median relative error (8 samples) = 0.002197
  block0.w1      median relative error (8 samples) = 0.002099
  block1.wo      median relative error (8 samples) = 0.002000
  block1.gamma2  median relative error (8 samples) = 0.006082
  final_gamma    median relative error (8 samples) = 0.000362
  w_proj         median relative error (8 samples) = 0.001170
test 1 (gradient check): PASS
training: loss 2.0001 -> 0.0126
in-batch retrieval accuracy: before=0.333 after=1.000 (chance=0.167, batch=6)
test 2 (trains and improves in-batch retrieval accuracy): PASS
PASS
```

## Findings

- **A real, non-obvious finite-difference wrinkle**: `token_emb`'s
  gradient check initially failed (median relative error 0.108) at the
  same `epsilon=1e-3` every other parameter uses cleanly. Root-caused (not
  papered over) by comparing analytic vs. numeric values directly: unlike
  every other parameter here, one `token_emb` ROW is often read at
  *multiple positions within the same sequence* (repeated characters, e.g.
  the two `p`s in "apple"), so perturbing it by a fixed epsilon shifts
  several attention inputs in that sequence simultaneously — a genuine
  second-order/truncation effect, not a wrong gradient formula. Confirmed
  by re-running at `epsilon=1e-4`: median error dropped to 0.0078, in line
  with every other parameter. `pos_emb`, by contrast, is read at most once
  per sequence per position (no within-sequence repetition), so it never
  hit this effect at `epsilon=1e-3`.
- **The contrastive objective actually works end to end**: in-batch
  retrieval accuracy (does each query's nearest document by cosine
  similarity match its true positive, out of 6 candidates) goes from
  0.333 before training (already above the 0.167 random-chance baseline —
  the untrained encoder's raw embeddings have some incidental lexical
  signal from shared characters) to a clean 1.000 after 600 SGD steps,
  with training loss falling by >99% (2.0001 -> 0.0126). This is the
  proof-of-life for the whole stack: bidirectional attention, mean
  pooling, projection, L2 normalize, AND the two-tower gradient (a single
  shared parameter set receiving gradient contributions from both the
  query and document sides of the same batch) all compose correctly.
- Real, disclosed scope limitation (documented, not hidden): this is
  character-level, lexical-overlap-driven contrastive learning on 6
  hand-written topic pairs picked so each query shares specific words with
  its true document — it demonstrates the mechanism is correct, not that
  the encoder learns semantic (as opposed to lexical) similarity. The
  larger `indexing_pipeline`/`recall_eval` steps reuse this same encoder
  at slightly larger scale on a bigger synthetic corpus.

## Hardware notes
None — pure CPU, char-level tokenizer, single-sequence-per-call forward
(same scope note as `transformer/`), no batched-matmul kernel dependency.
