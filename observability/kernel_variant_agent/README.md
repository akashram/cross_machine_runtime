# kernel_variant_agent

**Status: code-complete, hardware/API-gated — real Claude API client code
and real nvcc/CUDA-event compile+benchmark code, neither invoked. See
Design.**

## What this measures

PLAN.md Phase 10 step 8: given a kernel specification (op type, input
shapes, dtype), generates N variants with different tiling/vectorization
parameters via the Claude API, compiles and benchmarks each, promotes the
winner, documents the search process.

## Design

- `generate_variants`: real Anthropic Python SDK call
  (`model="claude-opus-4-8"`) with `output_config.format` structured
  outputs — each variant comes back as a typed object (tile dims, unroll
  factor, block dims, a rationale string, and complete compilable CUDA
  source), not free text to hand-parse.
- `compile_and_benchmark`: real `subprocess` calls to `nvcc -O3` per
  variant, then runs the compiled binary and parses a `LATENCY_US: <float>`
  line from its stdout — the contract each generated `cuda_source` is
  prompted to satisfy (a `main()` that warms up, times `iters` launches
  via CUDA events, and prints that line).
- `promote_winner`: picks the lowest-latency variant among the ones that
  both compiled and benchmarked successfully, returns it paired with
  every variant's result — the documented search process PLAN.md asks
  for, not just the winner in isolation.
- **Deliberately never invoked**: same treatment as `nsight_agent` (step
  7) and `llm_autotune` (step 9) — real client code, no actual API calls
  made from this repo (see project memory for the explicit decision). The
  compile/benchmark half is separately GPU-toolchain-gated regardless of
  the API question — no `nvcc`/CUDA device on this Mac.

## Results
TODO: run with an `ANTHROPIC_API_KEY` set (or an `ant auth login`
profile) and a CUDA-capable GPU with `nvcc` on `PATH`.

| Variant | Tile (M,N,K) | Unroll | Compiled? | Latency (us) |
|---|---|---|---|---|
| (generated at runtime) | TODO | TODO | TODO | TODO |

## Hardware notes
- Required: `ANTHROPIC_API_KEY` (or an `ant auth login` profile) for
  `generate_variants`; NVIDIA GPU + CUDA toolkit (`nvcc` on `PATH`) for
  `compile_and_benchmark`.
