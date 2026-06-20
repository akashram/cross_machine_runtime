# Phase 9: Inference Serving

**Status: STUB — requires GPU with large VRAM (p3.2xlarge or p4d).**

## Overview
Continuous batching, paged KV cache, speculative decoding, GPTQ INT4 quantization,
FlashDecoding for long contexts. Benchmark vs vLLM and TensorRT-LLM.

## Steps

| # | Directory | What |
|---|-----------|------|
| 1 | paged_kv | PagedAttention KV cache with block table |
| 2 | continuous_batching | Dynamic batch formation across sequences |
| 3 | sla_scheduler | Latency-budget preemption + priority queue |
| 4 | flash_decoding | Parallel KV access for long contexts |
| 5 | speculative_decoding | Draft model + verifier, acceptance rate |
| 6 | gptq | INT4 group quantization, perplexity tradeoff |
| 7 | kv_quant | INT8 KV cache, memory vs accuracy |
| 8 | serving_backend | Backend-agnostic serving (CPU/GPU/FPGA/TPU) |
| 9 | serving_bench | Throughput + latency vs vLLM + TensorRT-LLM |

## Hardware notes
- Required: GPU with ≥ 16GB VRAM (p3.2xlarge has V100 16GB)
- Steps 1-3, 8: can develop on CPU, GPU needed for real benchmarks
