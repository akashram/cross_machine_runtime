# kv_quant

**Status: code-complete AND locally run — pure CPU quantization math,
applied to `transformer/`'s real trained model's real K/V activations, no
GPU dependency.**

## What this measures

PLAN.md Phase 9 step 7: INT8 KV cache, memory reduction vs. accuracy
impact.

## Design

- `quantize_int8_symmetric`/`dequantize_int8`: per-tensor symmetric INT8
  (no zero point) — simpler than `gptq`'s per-group asymmetric scheme
  deliberately: a KV cache is quantized on the fly as new tokens are
  generated, so it can't afford GPTQ's per-group Hessian bookkeeping at
  serving time. Real INT8-KV-cache implementations use exactly this
  simpler symmetric scheme.
- `causal_attention_forward_int8kv`: same math as
  `transformer::causal_attention_forward`, except K and V are round-
  tripped through quantize/dequantize immediately after projection —
  simulating exactly what reading K/V back out of an INT8-quantized cache
  would produce, without needing a real persistent cache structure (that
  bookkeeping is `paged_kv`/step 1's concern; this step isolates the pure
  numerical effect of quantization). Q is never cached (it's the current
  step's query, not stored across steps), so it stays FP32.
- `model_forward_int8kv`: a forward-only replica of
  `transformer::model_forward`/`block_forward`'s control flow, using the
  INT8-KV attention for every head. Necessarily duplicates that control
  flow rather than reusing `block_forward` directly, since that function
  is tightly coupled to `BlockCache` for backward (irrelevant here — KV
  quantization is purely an inference-time technique).
- `compare_kv_memory`: computes the *exact* FP32-vs-INT8 KV cache byte
  count (elements × 4 vs. elements × 1, plus the real per-tensor scale
  float overhead) at a given seq_len/d_model/num_layers, rather than
  asserting "4x" as a given.

## Results (captured 2026-07-27, Apple clang 14 / `-std=c++2b`, this Mac)

```
  max quantization error=0.024906  theoretical bound=0.024917
PASS  INT8 round-trip error stays within the theoretical half-step bound
  KV cache @ seq=2048 d_model=4096 layers=32: fp32=2147483648 bytes, int8=536871168 bytes, reduction=4.000x
PASS  memory reduction is ~4x (scale overhead negligible at this size)
  perplexity: fp32-kv=1.0497  int8-kv=1.0497  (ratio=1.0000)
PASS  INT8 KV cache perplexity stays within 10% of FP32
```

## Findings

- Memory reduction at a realistic scale (2048-token context, d_model
  4096, 32 layers — roughly 7B-class dimensions) is a clean 4.000x: 2 GiB
  FP32 KV cache down to ~512 MiB INT8, with the per-tensor scale overhead
  (64 floats total) rounding away to nothing at this size — the "4x"
  claim isn't a rule of thumb here, it's computed from the actual
  element/byte counts.
- The perplexity impact on the real trained model is unmeasurably small
  (ratio 1.0000 to 4 decimal places) — same saturation effect
  `gptq/README.md` reports for weight quantization on this corpus: a
  small, well-trained model already predicts its own training corpus
  with near-certainty (perplexity ≈ 1.05), leaving very little precision
  headroom for INT8 rounding in either weights or KV cache to visibly
  damage. The max-quantization-error check (0.0249, right at the
  theoretical half-quantization-step bound of 0.0249) confirms the
  quantization itself is behaving correctly — the near-zero perplexity
  delta reflects this corpus's ceiling effect, not a broken or
  no-op quantization path.

## Hardware notes
None. Pure CPU quantization math and forward-pass arithmetic; a
production KV cache would apply this same quantize-on-write/dequantize-
on-read round trip inside a real GPU attention kernel, unchanged in
substance.
