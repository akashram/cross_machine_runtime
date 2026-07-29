# serving_integration

**Status: code-complete AND locally run — pure CPU, no external
dependency.**

## What this measures

PLAN.md Phase 13 step 9: wire the retrieval + generation pipeline into
`inference_serving/serving_backend`'s `ServingRouter` (Phase 9 step 8) as
a new request path.

## Design

- **`serving_router.h`/`.cpp` are NOT modified.** `inference_serving` is a
  Phase 9 component and must stay usable with zero Phase 13 (`rag/`)
  dependency — this repo's dependency direction is always later phases
  depending on earlier ones, never the reverse (the same reason
  `rag_generation` links `ServingRouter`, not the other way around).
- `rag_serving.h`'s `route_rag()` does exactly what `rag_generate()`
  (step 5) did — embed the query, retrieve top-k chunks, construct the
  augmented prompt — EXCEPT the final generation call goes through the
  caller's existing `ServingRouter::route()` instead of calling
  `make_cpu_backend()` directly. CPU/GPU/FPGA/TPU backend selection and
  the fixed fallback priority order (GPU, TPU, FPGA, CPU) stay entirely
  `ServingRouter`'s unmodified job; this file only adds what happens
  BEFORE that dispatch.
- Same scope boundary as step 5: tested with untrained encoder/causal-
  model weights, since this step verifies the DISPATCH wiring
  (retrieval-then-routing composes, fallback still works, output is
  identical whether reached via fallback or direct preference), not
  generation quality (step 7) or retrieval quality (steps 4/6).

## Results (captured 2026-07-29, Apple clang 14 / `-std=c++2b`, this Mac)

```
  routed via GPU (unavailable): backend_used=cpu, fell_back=true, retrieved 2 chunk(s)
PASS  requesting an unavailable backend (GPU) falls back to CPU through the SAME ServingRouter fallback logic Phase 9 already tests
PASS  route_rag reports that a fallback occurred
PASS  route_rag retrieves k chunks before dispatching to the router
PASS  the retrieved chunk text is actually present in the prompt handed to the router
PASS  the router-dispatched generation produces exactly max_new_tokens new characters
PASS  requesting the available CPU backend directly needs no fallback
PASS  GPU-preferred-with-fallback and CPU-preferred-directly reach the identical CPU backend and produce identical output
PASS  use_retrieval=false skips retrieval entirely, same as rag_generate()
PASS  the no-retrieval request path produces a different (shorter, context-free) prompt
```

## Findings

- Requesting GPU (registered but unavailable, same honest
  `available=false` + reason-string convention every backend in this
  router uses) through the RAG path falls back to CPU and produces
  BYTE-IDENTICAL output to requesting CPU directly — confirms `route_rag`
  doesn't duplicate or diverge from `ServingRouter`'s fallback logic, it
  genuinely delegates to it.
- This is the last piece needed for Phase 13's own framing (see
  `rag/DESIGN.md`) that RAG is a composition of already-built
  capabilities: steps 1/2 (retrieval), 4 (indexing), 5 (prompt
  construction), and now 9 (serving) touch zero new generation or
  backend-dispatch logic — only step 9's ~25 lines of retrieval-then-
  routing glue are new here.
- Once GPU/FPGA/TPU stop being `available=false` stubs (the hardware
  validation pass other phases are already waiting on), this exact same
  `route_rag()` call starts actually dispatching RAG requests to whichever
  backend is fastest, with no code change — the fallback priority order
  already treats them as first-class citizens today, just unavailable
  ones.

## Hardware notes
None — pure CPU, same as every Phase 13 step so far. GPU/FPGA/TPU are
registered honestly-unavailable, matching `serving_router_test.cpp`'s own
convention.
