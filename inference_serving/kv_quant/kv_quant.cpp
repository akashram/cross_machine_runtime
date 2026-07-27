#include "kv_quant.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace inference_serving {

Int8Tensor quantize_int8_symmetric(const Matrix &m) {
  Int8Tensor q;
  q.rows = m.rows();
  q.cols = m.cols();
  float max_abs = 0.0f;
  for (int i = 0; i < m.rows(); ++i)
    for (int j = 0; j < m.cols(); ++j) max_abs = std::max(max_abs, std::abs(m(i, j)));
  q.scale = (max_abs > 1e-8f) ? (max_abs / 127.0f) : 1.0f;

  q.qdata.resize(static_cast<std::size_t>(m.rows()) * static_cast<std::size_t>(m.cols()));
  for (int i = 0; i < m.rows(); ++i)
    for (int j = 0; j < m.cols(); ++j) {
      int32_t v = static_cast<int32_t>(std::lround(m(i, j) / q.scale));
      v = std::clamp(v, -127, 127);
      q.qdata[static_cast<std::size_t>(i) * static_cast<std::size_t>(m.cols()) + static_cast<std::size_t>(j)] =
          static_cast<int8_t>(v);
    }
  return q;
}

Matrix dequantize_int8(const Int8Tensor &q) {
  Matrix out(q.rows, q.cols);
  for (int i = 0; i < q.rows; ++i)
    for (int j = 0; j < q.cols; ++j)
      out(i, j) = static_cast<float>(q.qdata[static_cast<std::size_t>(i) * static_cast<std::size_t>(q.cols) + static_cast<std::size_t>(j)]) * q.scale;
  return out;
}

transformer::Matrix causal_attention_forward_int8kv(const transformer::Matrix &q, const transformer::Matrix &k,
                                                     const transformer::Matrix &v) {
  // The KV-cache round-trip: quantize right after projection, dequantize
  // right before use -- exactly what reading K/V back out of an
  // INT8-quantized cache would produce. Q is never cached (it's the
  // current step's query, not stored across steps), so it stays FP32.
  Matrix k_rt = dequantize_int8(quantize_int8_symmetric(k));
  Matrix v_rt = dequantize_int8(quantize_int8_symmetric(v));

  float scale = 1.0f / std::sqrt(static_cast<float>(q.cols()));
  Matrix s = q.matmul(k_rt.transpose()) * scale;
  for (int i = 0; i < s.rows(); ++i)
    for (int j = i + 1; j < s.cols(); ++j) s(i, j) = -1e9f;
  Matrix a = transformer::softmax_rows(s);
  return a.matmul(v_rt);
}

transformer::Matrix model_forward_int8kv(const transformer::ModelParams &p, const std::vector<int> &token_ids) {
  const auto &cfg = p.config;
  int seq = static_cast<int>(token_ids.size());
  int head_dim = cfg.head_dim();

  Matrix x(seq, cfg.d_model);
  for (int i = 0; i < seq; ++i)
    for (int j = 0; j < cfg.d_model; ++j)
      x(i, j) = p.token_emb(token_ids[static_cast<std::size_t>(i)], j) + p.pos_emb(i, j);

  for (int l = 0; l < cfg.num_layers; ++l) {
    const auto &bp = p.blocks[static_cast<std::size_t>(l)];
    transformer::LayerNormCache ln1_cache;
    Matrix ln1_out = transformer::layernorm_forward(x, bp.gamma1, bp.beta1, ln1_cache);

    Matrix qm = ln1_out.matmul(bp.wq);
    Matrix km = ln1_out.matmul(bp.wk);
    Matrix vm = ln1_out.matmul(bp.wv);

    Matrix concat(seq, cfg.d_model);
    for (int h = 0; h < cfg.num_heads; ++h) {
      Matrix qh(seq, head_dim), kh(seq, head_dim), vh(seq, head_dim);
      for (int i = 0; i < seq; ++i)
        for (int j = 0; j < head_dim; ++j) {
          qh(i, j) = qm(i, h * head_dim + j);
          kh(i, j) = km(i, h * head_dim + j);
          vh(i, j) = vm(i, h * head_dim + j);
        }
      Matrix oh = causal_attention_forward_int8kv(qh, kh, vh);
      for (int i = 0; i < seq; ++i)
        for (int j = 0; j < head_dim; ++j) concat(i, h * head_dim + j) = oh(i, j);
    }
    Matrix attn_out = concat.matmul(bp.wo);
    Matrix x_after_attn = x + attn_out;

    transformer::LayerNormCache ln2_cache;
    Matrix ln2_out = transformer::layernorm_forward(x_after_attn, bp.gamma2, bp.beta2, ln2_cache);
    Matrix mlp_pre1 = ln2_out.matmul(bp.w1).add_row_broadcast(bp.b1);
    Matrix mlp_hidden = mlp_pre1.apply([](float v) { return v > 0.0f ? v : 0.0f; });
    Matrix mlp_out = mlp_hidden.matmul(bp.w2).add_row_broadcast(bp.b2);

    x = x_after_attn + mlp_out;
  }

  transformer::LayerNormCache final_ln_cache;
  Matrix final_ln_out = transformer::layernorm_forward(x, p.final_gamma, p.final_beta, final_ln_cache);
  return final_ln_out.matmul(p.w_out);
}

KvMemoryComparison compare_kv_memory(int seq_len, int d_model, int num_layers) {
  KvMemoryComparison r;
  // K and V, every layer, full d_model width (summed across heads),
  // seq_len tokens.
  std::size_t elements = static_cast<std::size_t>(seq_len) * static_cast<std::size_t>(d_model) *
                          static_cast<std::size_t>(num_layers) * 2;
  r.fp32_bytes = elements * sizeof(float);
  // INT8 payload plus one FP32 scale per (layer, K-or-V) tensor -- the
  // real overhead of per-tensor symmetric quantization, not rounded away.
  std::size_t scale_overhead_bytes = static_cast<std::size_t>(num_layers) * 2 * sizeof(float);
  r.int8_bytes = elements * sizeof(int8_t) + scale_overhead_bytes;
  r.reduction_factor = static_cast<double>(r.fp32_bytes) / static_cast<double>(r.int8_bytes);
  return r;
}

}  // namespace inference_serving
