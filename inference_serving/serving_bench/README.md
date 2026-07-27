# serving_bench

**Status: this repo's own CPU-path half is code-complete AND locally run
(real `std::chrono` wall-clock timing of real compute — unlike steps 2/3's
pure scheduling simulations, there's actual transformer forward-pass work
happening here, so timing it is meaningful). The vLLM/TensorRT-LLM
comparison half is code-complete, hardware-gated, unrun.**

## What this measures

PLAN.md Phase 9 step 9: throughput (requests/sec), latency (p50/p99/p999
time-to-first-token, time-per-output-token), comparison against vLLM and
TensorRT-LLM.

## Design

- `serving_latency_bench.cpp`: instruments generation per-token (unlike
  `serving_backend`'s `make_cpu_backend`, which just returns the final
  token list) so TTFT (first token's latency) and TPOT (mean latency of
  every subsequent token) are captured separately, matching PLAN.md's
  metric definitions exactly. Runs 60 requests of 16 tokens each against
  a real trained `transformer/` model, reports p50/p99/p999 for both
  metrics plus overall req/s and tok/s throughput.
- `vllm_tensorrt_bench.py`: real vLLM (`vllm.LLM` +
  `SamplingParams`) and TensorRT-LLM (`tensorrt_llm.hlapi.LLM` — the
  newer high-level API, which deliberately mirrors vLLM's shape for easy
  side-by-side benchmarking) scripts against a real hosted checkpoint.
  Hardware-gated, unrun: no GPU, no vLLM/TensorRT-LLM install, no hosted
  checkpoint on this Mac.
- Explicitly **not** an apples-to-apples comparison once both halves have
  numbers: this repo's CPU path runs a tiny CPU-trained toy model,
  unbatched (no `continuous_batching`), uncached (no `paged_kv` reuse —
  every token's forward pass recomputes the *entire* sequence from
  scratch, since `transformer/model_forward` has no persistent KV cache;
  see Findings). vLLM/TensorRT-LLM would run a real hosted checkpoint with
  every serving optimization this repo's Phase 9 steps 1-3/6-7
  individually implement, already built in. The comparison's value is
  qualitative and directional (how much headroom does a naive path leave
  on the table), not a claim that this repo's toy stack should
  outperform production serving engines.

## Results (captured 2026-07-27, Apple clang 14 / `-std=c++2b`, this Mac)

```
requests=60  max_new_tokens=16  wall=2.620s
throughput: 22.9 req/s, 366.4 tok/s
metric          p50(ms)    p99(ms)   p999(ms)
TTFT             0.3178     0.4903     0.4903
TPOT             2.7239     2.8969     2.8969
```

## Findings

- TPOT is ~8.6x TTFT (2.72ms vs. 0.32ms) — not noise, a direct
  consequence of `transformer/model_forward` having no KV cache: the
  first token's forward pass processes a 1-token sequence, while every
  later token's forward pass **recomputes the full growing sequence from
  scratch** (token 16's pass reprocesses all 16 prior tokens' attention,
  not just the new one). This is a real, measured illustration of
  *exactly* the problem `paged_kv` (step 1) and `continuous_batching`
  (step 2) exist to solve in a production system — this repo's own
  minimal model doesn't implement incremental KV-cached decoding (out of
  its stated scope, see `transformer/README.md`), so this benchmark
  incidentally demonstrates the cost of not having one.
- p99 and p999 are close to p50 for both metrics (no long tail) — expected
  for a single-threaded, unbatched CPU loop with no contention, GC, or
  scheduler noise; a meaningful p999 tail is exactly what
  `sla_scheduler`'s violation-rate framing (step 3) is built to manage
  once real contention exists (batched, multi-tenant serving).

## Hardware notes
- vLLM/TensorRT-LLM comparison requires a GPU instance with a hosted
  checkpoint and both frameworks installed. TensorRT-LLM's Python API
  surface has moved fast across releases — confirm
  `tensorrt_llm.hlapi.LLM`'s exact shape against whatever version is
  installed before relying on `vllm_tensorrt_bench.py` unrun.
