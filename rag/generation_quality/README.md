# generation_quality

**Status: code-complete AND locally run — pure CPU, no external
dependency.**

## What this measures

PLAN.md Phase 13 step 7: does retrieval measurably help the base
transformer answer questions it can't from training data alone? A real,
honest finding either way — not assumed.

## Design

- `causal_qa_model.h` trains a small causal transformer
  (`transformer/transformer_model.h`, unmodified) on 16 examples derived
  from `rag/corpus/corpus.h`'s 8 queries: for each query, one example
  `"q: <query> c: <its true document text> a: <answer>."` and one
  `"q: <query> c: none a: unknown."` — teaching the model to condition its
  answer on whatever text actually appears after `c:`, and to abstain
  when there is none.
- **Real, disclosed scope limitation**: at 8 queries and a tiny char-level
  model, this is closer to memorizing 16 (prompt -> continuation) mappings
  than a skill that would generalize to an unseen question. What IS real
  and load-bearing: the model's output is driven by which TEXT is
  actually in the prompt, not hard-coded per query — so the "with
  retrieval" evaluation feeds it the ACTUAL top-1 chunk from the real
  embedding + `CosineBallTree` pipeline (steps 1/2/4), not injected ground
  truth. Retrieval still has to work for the demonstration to work.
- Two complementary measurements: **generation accuracy** (does the
  greedy-decoded continuation after `a:` contain the true answer?) and a
  **perplexity-style probe** (teacher-forced loss forcing the TRUE answer
  with real context vs. with context forced to `"none"` — the
  without-context version was never seen in training, since training's
  no-context examples all end in `"unknown."`, so this is a genuine
  held-out likelihood check, not re-scoring a memorized string).

## Results (captured 2026-07-29, Apple clang 14 / `-std=c++2b`, this Mac)

```
  generation accuracy WITH retrieval:    1.000
  generation accuracy WITHOUT retrieval: 0.000
PASS  with real retrieved context, the model generates the correct answer for most queries
PASS  without any context, the model does NOT fabricate the correct answer for most queries
PASS  retrieval measurably improves generation accuracy on this query set
  avg sequence loss forcing the TRUE answer, WITH context:    0.0800
  avg sequence loss forcing the TRUE answer, WITHOUT context: 1.1727
PASS  the model is more 'surprised' by the correct answer when no context is given (a real perplexity-style retrieval benefit)
PASS
```

## Findings

- Generation accuracy goes from 0.000 (no query answered correctly
  without context — the model reliably says "unknown" instead of
  guessing) to 1.000 (every query answered correctly with real retrieved
  context) — the cleanest possible version of "retrieval measurably
  helps," achievable here because retrieval itself was already shown to
  be perfect (recall@1 = 1.000) at this corpus scale in `recall_eval`.
- The perplexity-style probe tells a complementary, more graded story:
  forcing the true answer without context costs ~14.7x more loss (1.1727
  vs. 0.0800) than forcing it with real context — quantifying HOW
  confidently wrong the model would need to become to produce the right
  answer without retrieval, not just whether greedy decoding happens to
  land on it.
- **Wall-clock had to be re-tuned, same lesson as `indexing_pipeline`**:
  the QA model's first training pass (300 epochs) took 3m48s end to end;
  cut to 150 epochs (also raising the learning rate slightly, 0.1 ->
  0.15, to compensate) brought it to 2m11s with identical qualitative
  results (1.000/0.000 accuracy, same order-of-magnitude loss gap) —
  this step's overall runtime is the slowest in Phase 13 so far since it
  trains two separate models (the retrieval encoder AND the causal QA
  model) plus builds the 48-document index, all in an unoptimized debug
  build.

## Hardware notes
None — pure CPU, same as every Phase 13 step so far.
