# Retrieval-Augmented Generation — Design

Status: Phase 13 code-complete (9/9 steps, 2026-07-29), all 9 steps
**locally run** with real captured output — no hardware gate anywhere in
this phase, unlike Phases 3/4/7/8/15. See each step's own README for full
results; this document covers the decisions behind the split and the
recall/latency/quality tradeoffs measured across steps 6-8.

---

## 1. RAG as composition, not a new system

Every one of Phase 13's 9 steps reuses an already-built, already-tested
piece rather than re-deriving it:

| Step | New logic | Reused from |
|---|---|---|
| 1 embedding_model | bidirectional block + pooling + InfoNCE | `seq_parallel` LayerNorm, `tensor_parallel_attn` attention |
| 2 cosine_ann | L2-normalize + delegate | `ml/knn`'s `BallTree` (unmodified) |
| 3 hnsw | multi-layer graph | `ml/knn`'s `Features`/`NeighborResult`/`squared_distance` |
| 4 indexing_pipeline | chunking | steps 1 + 2 |
| 5 rag_generation | prompt template | step 4 + `inference_serving::make_cpu_backend()` |
| 6 recall_eval | measurement only | step 4 |
| 7 generation_quality | QA training loop | `transformer_model.h` (unmodified) + step 4 |
| 8 approx_retrieval_study | **zero new logic** | steps 6 + 7's own `approximate` flag |
| 9 serving_integration | retrieval-then-routing glue (~25 lines) | step 5 + `ServingRouter::route()` (unmodified) |

Step 8 is the clearest case: it adds no retrieval or generation code at
all, only a comparison between two measurements that already existed.
This is a deliberate test of the "no toy re-implementation" standard the
rest of the project holds — if RAG genuinely IS a composition of
embedding, ANN search, generation, and serving, then wiring it up should
mostly be glue, not new algorithms. It was.

## 2. Why a shared corpus (`rag/corpus/corpus.h`) instead of each step
   inventing its own data

Steps 4, 6, 7, and 8 all need the same ground truth to be comparable to
each other — recall@k (6), generation accuracy (7), and the
approximate-vs-exact system-level study (8) only mean something if
they're measuring the same retrieval decisions. `corpus.h` centralizes 8
hand-labeled `(query, relevant_doc_id, answer)` triples plus, added during
step 6, 40 deterministic distractor documents.

## 3. The distractor documents exist because 8 points isn't enough to
   see approximate search do anything

`ml/knn`'s `BallTree` defaults to `leaf_size=10`. At the original 8-document
corpus, the whole tree is a single leaf — defeatist ("approximate") search
and exact search are then LITERALLY the same code path, since descending
to "the nearest child" from the root immediately reaches the only leaf.
Step 6 added 40 template-generated (not gibberish — real English words,
just deterministic rather than hand-written) filler documents purely to
give the index enough scale for the two modes to diverge. This is the
same reason `ml/knn`'s own tests use n=300-2000 rather than a handful of
points — it's not a Phase-13-specific issue, just one this phase hadn't
hit yet until step 6 needed a meaningful approximate/exact comparison.

## 4. The recall/latency/quality tradeoff, measured end to end (steps 6-8)

This is the throughline PLAN.md's Phase 13 section asks steps 6-8 to
establish, and it required real code to answer, not intuition:

- **Recall** (step 6): exact retrieval is perfect at this scale
  (recall@1/3/5 = 1.000 over the 48-document index). `BallTree`'s
  approximate/defeatist mode drops to 0.875 recall@5 (misses 1 of 8
  queries) — a real, non-zero, measured cost of approximation, only
  visible because of the distractor bulk-up in §3.
- **Quality with vs. without retrieval** (step 7): a causal QA model
  trained to answer from whatever context text is actually in its prompt
  (or abstain when there is none) goes from 0.000 generation accuracy
  without retrieval to 1.000 with REAL retrieved context (the actual
  top-1 result from the trained embedding + `CosineBallTree` pipeline,
  not injected ground truth). A complementary perplexity-style probe
  shows a ~14.7x loss gap forcing the true answer with vs. without
  context — retrieval isn't just "sometimes helps," it's the entire
  difference between the model knowing an answer and correctly admitting
  it doesn't.
- **Is the recall cost free at the system level?** (step 8): No —
  measured directly rather than assumed. The 0.125 recall@1 drop from
  approximate retrieval translates into an EXACTLY EQUAL 0.125
  generation-accuracy drop. It doesn't compound (the model doesn't get
  more confused by wrong context than the retrieval error alone
  predicts), and it isn't absorbed either (no partial credit from
  related-but-wrong context at this model's scale). The disclosed
  boundary: this is a memorization-scale model (8 queries, tiny
  char-level transformer), not a reasoning one — a larger, more capable
  model might show graceful degradation or partial credit that this
  measurement can't speak to.

## 5. HNSW doesn't automatically win — measured, not assumed (step 3)

PLAN.md's own framing calls HNSW "a more scalable ANN structure," which
could read as "strictly better than BallTree." Benchmarked directly on
the same corpus (n=2000, 64 dims), `BallTree`'s EXACT search visited
FEWER distance-evaluated points (591.0 avg) than HNSW's APPROXIMATE
search (1015.4 avg) — while also being exact. HNSW's asymptotic advantage
over tree structures is real but shows up at a larger scale than this
corpus, and/or with a build tuned harder than this test's moderate
`M`/`ef_construction` settings. `indexing_pipeline` uses `CosineBallTree`
(step 2), not HNSW, for exactly this reason — the honest choice for this
corpus's actual size, not the newer-sounding algorithm by default.

## 6. Dependency direction: later phases depend on earlier ones, never
   the reverse (step 9)

`inference_serving/serving_backend/serving_router.h` (Phase 9) is not
modified anywhere in Phase 13. `rag/serving_integration/rag_serving.h`'s
`route_rag()` calls `ServingRouter::route()` unchanged for the actual
generation dispatch — Phase 9 stays buildable and testable with zero
knowledge that Phase 13 exists. The same discipline shows up in step 1
(a new `encoder_block_forward`/`_backward` pair rather than
parameterizing `transformer_model.h`'s causal-only block, which
`distributed_training`'s RLHF steps 22-25 already depend on and
gradient-check) and step 2 (`CosineBallTree` wraps `ml/knn::BallTree`
rather than editing it). Every new capability in this phase was added
by composing with earlier phases' code as a black box, never reaching
back into it.

## 7. Two real, disclosed toy-scale limitations

- **Character-level, lexical-overlap-driven contrastive learning** (step
  1): the embedding model demonstrates the InfoNCE mechanism is
  implemented correctly, not that it learns SEMANTIC (as opposed to
  lexical) similarity. The training corpus was deliberately picked so
  each query shares specific words with its true document.
- **Memorization over generalization** (step 7): 8 queries and a tiny
  model mean the QA model's "read the context and answer" behavior is
  closer to 16 memorized (prompt -> continuation) mappings than a skill
  that would transfer to an unseen question. What's real and
  load-bearing regardless: the mapping is keyed on the TEXT actually
  present in the prompt, so a wrong (or missing) retrieval genuinely
  changes the output — the causal chain retrieval -> context -> answer
  is real, even though the "understanding" behind it is not.

Both are exactly the kind of honestly-scoped limitation this repo's other
toy-scale demonstrations (`transformer/`, `distributed_training`'s RLHF
steps) already disclose rather than paper over — see PLAN.md's Phase 13
section and each step's own README for the full numbers.
