# Phase 9: Inference Serving

**Status: CODE COMPLETE (9/9 steps, 2026-07-27). No GPU on this Mac, but
this phase turned out to have far more genuinely CPU-portable work than
Phases 3/4/7/8 did: steps 1-3 and 8 are pure scheduling/data-structure
logic with no hardware dependency at all, and steps 5-7 reuse
`transformer/`'s real, actually-trained-on-this-Mac model to get real
perplexity/acceptance-rate numbers instead of stubs. Only step 4 (a CUDA
kernel) and half of step 9 (the vLLM/TensorRT-LLM comparison) are
genuinely hardware-gated. Every portable step is not just written but
**actually compiled, run, and tested locally** with real captured output
in its own README — a stronger bar than Phases 3/4/7/8's "written but
unrun" convention, because this phase's algorithms mostly don't need the
hardware to be meaningful.**

## Overview
Continuous batching, paged KV cache, speculative decoding, GPTQ INT4 quantization,
FlashDecoding for long contexts. Benchmark vs vLLM and TensorRT-LLM.

## Steps

| # | Directory | What | Status |
|---|-----------|------|--------|
| 1 | paged_kv | PagedAttention KV cache with block table | **code-complete, run locally** (300-trial property test) |
| 2 | continuous_batching | Dynamic batch formation across sequences | **code-complete, run locally** (1.85x throughput vs static, real simulation) |
| 3 | sla_scheduler | Latency-budget preemption + priority queue | **code-complete, run locally** (EDF: 0% violations vs FIFO's 36.5%) |
| 4 | flash_decoding | Parallel KV access for long contexts | code-complete, hardware-gated (real CUDA, unrun) |
| 5 | speculative_decoding | Draft model + verifier, acceptance rate | **code-complete, run locally** (real transformer/ draft+verifier, correctness verified) |
| 6 | gptq | INT4 group quantization, perplexity tradeoff | **code-complete, run locally** (real Hessian-guided quantization) |
| 7 | kv_quant | INT8 KV cache, memory vs accuracy | **code-complete, run locally** (exact 4.000x memory reduction measured) |
| 8 | serving_backend | Backend-agnostic serving (CPU/GPU/FPGA/TPU) | **code-complete, run locally** (real CPU backend + fallback routing) |
| 9 | serving_bench | Throughput + latency vs vLLM + TensorRT-LLM | own CPU path **run locally** (real TTFT/TPOT); vLLM/TensorRT-LLM half hardware-gated |

## Hardware notes
- Step 4 (flash_decoding): NVIDIA GPU + CUDA toolchain (only built when
  `CMAKE_CUDA_COMPILER` is found, mirroring `gpu_engine`'s gate).
- Step 9's vLLM/TensorRT-LLM comparison: GPU instance with both
  frameworks installed and a hosted checkpoint.
- Everything else: no hardware dependency. `inference_serving/CMakeLists.txt`
  used to gate the *entire* directory behind `CMAKE_SYSTEM_NAME ==
  Linux`, which was wrong for steps 1-3/5-8 — fixed when this phase
  started (see step 1's commit).

## Next
Hardware validation (step 4's CUDA kernel, step 9's GPU comparison) is
deferred along with every earlier phase — see the root `CLAUDE.md`'s
execution strategy. Next local-implementation phase: Phase 10
(Observability), currently stubbed.
