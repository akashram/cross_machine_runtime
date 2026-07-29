# rag_generation

**Status: code-complete AND locally run — pure CPU, no external
dependency.**

## What this measures

PLAN.md Phase 13 step 5: retrieval-augmented prompt construction — embed
the query, retrieve top-k chunks (step 4's `indexing_pipeline`), build the
augmented context window, and feed it into the existing causal generation
path.

## Design

- **`prompt_construction.h`**: `construct_prompt()` is plain string
  assembly — `"Context: <chunk1> <chunk2> ... Question: <query> Answer:"`,
  degrading cleanly to a query-only prompt when no chunks are retrieved
  (the shape `generation_quality` (step 7) needs for its with-vs-without
  ablation).
- **`rag_generate.h`**: composes `indexing_pipeline::query_index()` (step
  4) with `inference_serving::make_cpu_backend()` (Phase 9 step 8's
  already-built greedy-decode loop) — no third reimplementation of the
  generation loop `transformer_test.cpp` and `serving_router.cpp` already
  have.
- **Deliberate scope boundary**: this step's test uses **untrained**
  encoder and causal-model weights. It verifies the WIRING — retrieval
  really runs, the prompt really changes shape, generation really returns
  `max_new_tokens` new characters — not generation quality (that needs a
  causal model actually trained for the answer-from-context skill, which
  is step 7's job) or retrieval quality (already measured in
  `indexing_pipeline`/`recall_eval`). Reusing an untrained model here
  keeps this test fast and keeps concerns separated: step 7 doesn't need
  to re-derive that the plumbing works, just that a properly trained model
  benefits from it.

## Results (captured 2026-07-29, Apple clang 14 / `-std=c++2b`, this Mac)

```
PASS  constructed prompt contains every retrieved chunk's text
PASS  constructed prompt contains the query text
PASS  retrieved context precedes the question in the constructed prompt
PASS  with no retrieved chunks, the prompt degrades to query-only (no leftover context text)
  with retrieval:    2 chunk(s), prompt length=214, continuation length=10
  without retrieval: 0 chunk(s), prompt length=56, continuation length=10
PASS  rag_generate produces exactly max_new_tokens new characters (char-level, 1 token = 1 char)
PASS  rag_generate retrieves k chunks when use_retrieval=true
PASS  rag_generate retrieves nothing when use_retrieval=false
PASS  retrieval measurably changes what gets fed to the causal model (longer, context-bearing prompt)
PASS  the with-retrieval and without-retrieval prompts are not identical
PASS
```

## Findings

- The with-retrieval prompt (214 characters, 2 chunks) is meaningfully
  larger than the without-retrieval prompt (56 characters, query only) —
  a directly measured confirmation that retrieval changes what the causal
  model actually sees, not just a structural assumption.
- Reusing `inference_serving::make_cpu_backend()` instead of writing a
  third greedy-decode loop meant this step needed zero new generation
  logic — the only new code is retrieval-to-prompt glue.

## Hardware notes
None — pure CPU, same as every Phase 13 step so far.
