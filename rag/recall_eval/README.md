# recall_eval

**Status: code-complete AND locally run — pure CPU, no external
dependency.**

## What this measures

PLAN.md Phase 13 step 6: recall@k measured against a real
relevance-labeled query set (`rag/corpus/corpus.h`'s 8 hand-labeled
queries), over a realistically-sized index rather than a handful of
documents.

## Design

- **Why 48 documents, not 8**: at only 8 documents, `ml/knn`'s `BallTree`
  (default `leaf_size=10`) is a single leaf — its "approximate" mode is
  then *literally identical* to exact mode, since defeatist descent from
  the root immediately IS the (only) leaf. `rag/corpus/corpus.h` gained
  `sample_distractor_documents()` (40 deterministic, template-generated
  filler documents about unrelated topics, no relevance label) purely to
  give the index enough scale for exact and approximate search to
  actually differ — the same reason `ml/knn`'s own tests use n=300-2000
  rather than a handful of points.
- `recall_eval.h`'s `compute_recall_at_k()` is a thin measurement layer
  over `indexing_pipeline::query_index()` (step 4): for each labeled
  query, was its true relevant document anywhere in the top-k results?
  Reused as-is by step 8, which needs the identical measurement under
  approximate retrieval.
- The encoder is trained (`train_encoder.h`, unchanged from step 4) only
  on the 8 signal (query, document) pairs — the 40 distractors are never
  a contrastive target, only bulk. `train_corpus_encoder` gained an
  `extra_vocab_text` parameter (default empty, so steps 4/5's existing
  call sites are unaffected) purely so the tokenizer's vocabulary covers
  the distractors' characters too, since they get embedded (forward pass
  only, no gradient) when building the full index.

## Results (captured 2026-07-29, Apple clang 14 / `-std=c++2b`, this Mac)

```
  index size: 48 chunks (8 signal documents + 40 distractors)
  recall@1 = 1.000 (8/8)
  recall@3 = 1.000 (8/8)
  recall@5 = 1.000 (8/8)
PASS  recall@1 is well above chance (1/48 = 0.021) against a 48-document index
PASS  recall@3 is at least as high as recall@1 (larger k can only help)
PASS  recall@5 is at least as high as recall@3 (larger k can only help)
  recall@5 (BallTree approximate/defeatist mode) = 0.875 (7/8)
PASS  approximate retrieval's recall@5 never exceeds exact retrieval's (a real ceiling, not just usually true)
PASS
```

## Findings

- Recall@1 is a clean 1.000 against the 48-document index: the encoder,
  trained on only 8 contrastive pairs with no explicit exposure to the 40
  distractors, still separates the true match from 47 alternatives
  perfectly at this corpus's scale — the InfoNCE objective's in-batch
  negatives (the other 7 signal documents) generalize as an implicit
  margin against the unrelated distractor text too, at least here.
- **A real, measured approximate-vs-exact gap finally shows up**: exact
  search stays at 1.000 recall@5, but `BallTree`'s defeatist (approximate)
  mode drops to 0.875 (misses 1 of 8) at the same k — confirming the
  distractor bulk-up in `corpus.h` achieved its purpose (a scale where
  approximation actually costs something), and setting up exactly the
  comparison step 8 goes deeper on: does that recall drop actually harm
  the downstream generation task, or is it "free" at the system level?
- This is a small, honest sample (8 queries) — a single query's retrieval
  outcome moves recall by 12.5 points. The finding here is qualitative
  (approximate retrieval has a measurable, non-zero cost on this corpus)
  rather than a precise quantitative recall estimate, which would need a
  much larger labeled query set to pin down tightly.

## Hardware notes
None — pure CPU, same as every Phase 13 step so far.
