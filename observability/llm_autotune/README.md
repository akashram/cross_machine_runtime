# llm_autotune

**Status: heuristic detection core code-complete AND locally run; the
Claude API call and the apply/re-benchmark/compare loop are real code,
never invoked/exercised — see Design.**

## What this measures

PLAN.md Phase 10 step 9 — the final observability step, explicitly
described as coming "after all other baselines exist": monitors runtime
execution traces (latency per op, device utilization, memory pressure),
identifies suboptimal placement or quantization decisions, proposes and
tests changes, compares pre/post baselines.

## Design

- **Trace shape mirrors `observability/dashboard`'s `MetricSample`**
  (backend, rank, metric, value, ts_ns) deliberately — both tools are
  meant to read the same trace files, dashboard for a human-readable
  summary, this agent for automated issue detection.
- `detect_issues`: real, portable, rule-based heuristic core — no LLM, no
  hardware. Three checks, each tied to a real finding from an earlier
  step in this repo rather than an arbitrary threshold invented for this
  step alone:
  1. **Placement**: the same metric's mean differs ≥3x across backends —
     the same shape of gap `serving_bench` (step 9 of Phase 9) actually
     measured (TPOT ~8.6x TTFT on this repo's own uncached CPU path).
  2. **Utilization**: GPU/TPU utilization averaging below a floor —
     signals a placement/batching problem, not a hardware limit; a real
     deployment would source the floor from `layout_opt`/`hbm_sram`'s
     ceiling models for the specific op shape rather than a flat 50%.
  3. **Memory pressure without quantization**: high memory pressure on a
     backend with no `kv_cache_bits`/`weight_bits` metric present —
     `kv_quant`/`gptq` (Phase 9 steps 6-7) measured a real 4x memory
     reduction at near-zero cost on this repo's own workload, so this
     flags where that win isn't being applied.
- `propose_fix`: real Anthropic Python SDK call
  (`model="claude-opus-4-8"`) that sends the detected issues plus trace
  context and asks for specific, testable change proposals. Never
  invoked — same explicit decision as `nsight_agent`/`kernel_variant_agent`.
- `run_autotuning_loop`'s apply/re-benchmark/compare phase is documented
  as a real gap, not implemented as a no-op: it would need to re-run
  whichever benchmark produced the trace under the proposed change,
  which requires a real running system this repo doesn't have locally.

## Results (captured 2026-07-27, this Mac — heuristic core only)

Synthetic trace built from this repo's own real numbers (the placement
check's 2.7239ms/0.3178ms pair is `serving_bench`'s actual captured
CPU-path TPOT/TTFT):

```
55 samples loaded, 3 issues detected
  [placement] 'op_latency_ms' on backend 'cpu' is 8.6x '0.318' on 'gpu' -- consider moving this op
  [utilization] gpu_util_pct on 'gpu' averaged 35.0% below the 50% floor over 10 samples
  [quantization] backend 'gpu' shows high memory pressure with no quantization metric present -- kv_quant/gptq measured a 4x memory reduction at near-zero cost on this repo's own workload; consider applying it here
```

The 8.6x the placement check recovers matches `serving_bench`'s real
measured TPOT/TTFT ratio (2.7239/0.3178 ≈ 8.57) almost exactly — the
heuristic correctly flags the same gap that step's own analysis already
found by hand, run here as an automated check instead.

## Hardware notes
- `detect_issues`: none — pure trace analysis.
- `propose_fix`: `ANTHROPIC_API_KEY` (or an `ant auth login` profile).
- The apply/re-benchmark/compare loop: a real running system to
  re-benchmark against (backend-specific — GPU for placement/quant
  changes tested on `gpu_engine`, etc.).
