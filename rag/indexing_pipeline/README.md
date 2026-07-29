# indexing_pipeline

**Status: code-complete AND locally run — pure CPU, no external
dependency.**

## What this measures

PLAN.md Phase 13 step 4: real text corpus chunking (sentence-boundary
aware, with overlap) and an indexing pipeline that embeds every chunk
(step 1's encoder) and builds a searchable index over them (step 2's
`CosineBallTree`).

## Design

- **`chunking.h`**: `split_sentences()` splits at `.`/`!`/`?` boundaries
  (never mid-sentence); `chunk_text()` greedily packs sentences into
  chunks up to `max_chunk_chars`, carrying the previous chunk's trailing
  `overlap_chars` characters into the next chunk's start so a fact
  straddling a boundary still appears whole in at least one chunk.
- **`index_pipeline.h`**: `build_index()` chunks every document, embeds
  each chunk with the trained encoder, and builds a `CosineBallTree` over
  the resulting vectors — pure composition of steps 1/2/4, no new
  retrieval logic. `RagIndex` is move-only (it owns a `CosineBallTree`,
  which owns a `BallTree`, which owns `unique_ptr` tree nodes).
- **`rag/corpus/corpus.h`** (new, shared by steps 4/6/7/8): 8 short
  factual documents plus hand-labeled `QueryJudgment`s (one relevant
  document + expected answer substring per query) — one shared ground
  truth so every downstream step is comparable. Scope decision, disclosed
  in that file: every document here is short enough that `build_index`'s
  default chunking settings produce exactly one chunk per document, so
  "relevant chunk" and "relevant document" coincide throughout 6-8. This
  test file's own `test_chunking_splits_and_overlaps` is what actually
  exercises multi-chunk splitting, on a deliberately longer synthetic
  paragraph instead.
- **`rag/embedding_model/train_encoder.h`** (new): trains a Siamese
  encoder on the corpus's (query, relevant-document) pairs via the exact
  same InfoNCE mechanism `embedding_model_test.cpp` validates — reused by
  every downstream step so they all retrieve against the identical
  trained index.

## Results (captured 2026-07-29, Apple clang 14 / `-std=c++2b`, this Mac)

```
  chunk_text produced 9 chunks from a 177-char paragraph (max_chunk_chars=40)
PASS  a long paragraph with a small max_chunk_chars splits into multiple chunks
PASS  every chunk stays close to the requested max_chunk_chars bound
PASS  consecutive chunks share carried-over overlap text, not a hard cut
  no-overlap chunking of the same paragraph: 6 chunks (vs 9 with overlap=10)
PASS  disabling overlap does not produce MORE chunks than the same text with overlap enabled
  indexed 8 chunks from 8 documents (expect 8 == 8: every doc is short enough to be one chunk)
PASS  indexing_pipeline's scope decision holds: one chunk per document here
  top-1 retrieval accuracy over 8 labeled queries: 1.000
PASS  the trained encoder + cosine index retrieves the correct document for most queries
PASS
```

## Findings

- **A real, initially-wrong intuition caught by the test itself**: the
  first version of this test asserted that disabling overlap should
  never produce FEWER chunks than the same text with overlap enabled.
  The actual measurement showed the opposite (6 chunks with no overlap vs.
  9 with `overlap_chars=10`) — correct on reflection: overlap consumes
  part of each chunk's capacity re-carrying the previous chunk's tail, so
  it takes *at least as many* chunks to cover the same text as with
  overlap disabled, never fewer. Fixed by re-deriving the right direction
  of the inequality and re-verifying against the measured numbers, rather
  than adjusting the numbers to fit the original (wrong) assumption.
- **Hyperparameters had to be re-tuned for wall-clock, not just
  accuracy**: `rag/corpus/corpus.h`'s documents (~70-90 characters) are
  ~4x longer than `embedding_model_test`'s toy 6-pair corpus, and
  attention cost is `O(seq^2 * d_model)` — roughly 15x more attention
  compute per layer from sequence length alone. The first version of
  `train_corpus_encoder` (2 layers, `d_model=24`, 800 epochs, copied
  from step 1's proportions) was still running after several minutes and
  had to be killed; cut to 1 layer, `d_model=16`, 200 epochs, which
  finishes in the same ~50-second ballpark as the rest of this repo's
  slower tests while still reaching 100% top-1 retrieval accuracy on the
  labeled query set.
- The full pipeline — chunk, embed, index, retrieve — gets every one of
  the 8 labeled queries to its true relevant document as the single
  top-1 result. This is a functional check, not the rigorous recall@k
  measurement (that's step 6's job, over a larger `k` and with
  approximate-retrieval comparisons).

## Hardware notes
None — pure CPU, same as every Phase 13 step so far.
