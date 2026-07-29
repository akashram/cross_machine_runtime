# approx_retrieval_study

**Status: code-complete AND locally run — pure CPU, no external
dependency.**

## What this measures

PLAN.md Phase 13 step 8: does `ml/knn`'s already-established
approximate/exact recall-speed tradeoff (BallTree's defeatist mode)
actually degrade end-task GENERATION quality once a real causal model is
reading whatever gets retrieved, or is the imperfect recall "free" at the
system level?

## Design

Pure composition — no new retrieval or generation logic, only the
comparison itself is new:

- **Retrieval-only view**: `recall_eval::compute_recall_at_k()` (step 6)
  at `k=1` (the k actually used to pick generation context), exact vs.
  `BallTree`'s approximate/defeatist mode, over the same 48-document index
  `recall_eval` and `generation_quality` already use.
- **System-level view**: `generation_quality::measure_qa_accuracy()`
  (step 7) — both already accept the same `approximate` flag — measuring
  whether the trained causal QA model still generates the correct answer
  when the retrieved context comes from approximate (possibly wrong)
  search.
- The comparison between the two drops is **deliberately not asserted in
  a fixed direction** in the test: whether a recall drop is "free"
  (absorbed, generation still correct), "not free" (costs exactly as much
  end-task accuracy), or "compounds" (costs even more) is the real
  question this step exists to answer, not an assumed outcome. Only the
  bounds that must hold regardless (`approximate <= exact`, for both
  recall and generation accuracy) are asserted as pass/fail.

## Results (captured 2026-07-29, Apple clang 14 / `-std=c++2b`, this Mac)

```
  recall@1 exact:       1.000 (8/8)
  recall@1 approximate: 0.875 (7/8)
  generation accuracy exact:       1.000
  generation accuracy approximate: 0.875
  recall@1 drop from approximation:            0.125
  generation accuracy drop from approximation: 0.125
PASS  approximate retrieval's end-task generation accuracy never exceeds exact retrieval's
PASS  approximate retrieval's recall@1 never exceeds exact retrieval's
  finding: generation accuracy fell by exactly the recall drop -- not free, not compounded
PASS
```

## Findings

- **The answer, measured directly rather than assumed**: at this corpus's
  scale, the recall drop is exactly NOT free — it translates 1:1 into a
  generation-accuracy drop (both exactly 0.125, one query out of eight).
  It also does not compound: the causal model doesn't get MORE confused
  by wrong context than the retrieval error alone would predict. This
  makes sense given `generation_quality`'s own scope note — the QA model
  is closer to memorizing 8 (query, context) -> answer mappings than
  applying a robust "read and reason" skill, so when approximate
  retrieval hands it a context it never saw paired with that query, it
  simply doesn't produce the right answer (and, per `generation_quality`'s
  training, likely falls back toward "unknown" or an unrelated
  continuation rather than confidently hallucinating the correct fact).
- This is consistent with, and a direct extension of, `ml/knn`'s own
  defeatist-search finding (`ml/knn/README.md`): approximate search is a
  genuine tradeoff, not a free lunch, at the raw-neighbor level. This
  step's contribution is showing that tradeoff's cost survives all the
  way through the rest of the RAG pipeline to the final generated answer,
  rather than being absorbed or amplified along the way — at least at
  this toy model's scale and mechanism (memorization-driven, not
  reasoning-driven). A larger, more capable causal model might behave
  differently (e.g. partial credit from a related-but-wrong context, or
  more graceful degradation) — a real, disclosed boundary of what this
  measurement can claim.
- Single-sample caveat (shared with `recall_eval`): 8 queries means each
  one moves every reported number by 12.5 points. The *shape* of the
  finding (recall drop -> equal generation drop) is real; the *precise*
  1:1 ratio is not something a labeled set this small can pin down beyond
  "the same order of magnitude."

## Hardware notes
None — pure CPU, same as every Phase 13 step so far.
