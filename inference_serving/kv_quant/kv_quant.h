#pragma once
#include "../../transformer/transformer_model.h"

#include <cstdint>
#include <vector>

// PLAN.md Phase 9 step 7: INT8 KV cache — measure memory reduction vs.
// accuracy impact. A KV cache holds every past token's K and V vectors
// per layer/head for the whole generation; at FP32, that's 4 bytes per
// value, and it's the dominant memory cost of long-context serving (this
// is exactly what paged_kv/step 1 pages). Quantizing K and V to INT8
// before they're written into the cache — and dequantizing right before
// the attention matmuls that consume them — cuts that memory 4x. This
// file measures whether doing so actually costs anything in output
// quality, using transformer/'s real trained model, not a synthetic
// probe.

namespace inference_serving {

using distributed_training::Matrix;

// Per-tensor symmetric INT8 quantization (no zero point — K and V
// activations are roughly zero-centered post-LayerNorm-derived
// projections, so symmetric quantization loses little vs. GPTQ's
// asymmetric per-group scheme; a KV cache also can't afford GPTQ's
// per-group Hessian bookkeeping at serving time, so real INT8-KV-cache
// implementations use exactly this simpler scheme).
struct Int8Tensor {
  std::vector<int8_t> qdata;
  float scale = 1.0f;
  int rows = 0, cols = 0;
};

Int8Tensor quantize_int8_symmetric(const Matrix &m);
Matrix dequantize_int8(const Int8Tensor &q);

// Same math as transformer::causal_attention_forward, except K and V are
// round-tripped through INT8 quantize/dequantize right after projection
// — simulating what reading K/V back out of an INT8-quantized KV cache
// would produce, without needing a real persistent cache structure (that
// bookkeeping is paged_kv/step 1's concern; this step is purely about the
// numerical effect of the quantization itself).
transformer::Matrix causal_attention_forward_int8kv(const transformer::Matrix &q, const transformer::Matrix &k,
                                                     const transformer::Matrix &v);

// Forward-only (no backward — KV quantization is an inference-time
// technique) replica of transformer::model_forward/block_forward, using
// causal_attention_forward_int8kv for every head instead of the FP32
// version. Necessarily duplicates block/model forward's control flow
// rather than reusing transformer_model.h's block_forward directly,
// since that function is tightly coupled to BlockCache (needed for
// backward, irrelevant here).
transformer::Matrix model_forward_int8kv(const transformer::ModelParams &p, const std::vector<int> &token_ids);

// Bytes one layer's K+V cache occupies for `seq_len` tokens at
// `d_model` total width (summed across all heads), FP32 vs INT8 (+ the
// negligible per-tensor scale float overhead) -- the real memory-
// reduction number PLAN.md step 7 asks for, computed exactly rather than
// asserted as "4x".
struct KvMemoryComparison {
  std::size_t fp32_bytes;
  std::size_t int8_bytes;
  double reduction_factor;
};
KvMemoryComparison compare_kv_memory(int seq_len, int d_model, int num_layers);

}  // namespace inference_serving
