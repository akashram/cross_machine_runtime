#pragma once

// A minimal decoder-only transformer: token + positional embedding, N
// pre-LN blocks (causal multi-head self-attention + residual, MLP +
// residual), final LayerNorm, output projection to vocab logits. Real,
// complete forward AND backward — hand-derived at the Matrix level
// (same pattern as col_row_linear/, tensor_parallel_attn/, and
// seq_parallel/: a fused custom Function per component, chain-ruled by
// hand through the residual connections here), not routed through
// autograd.h's generic Tensor tape, which would need new Tensor-level ops
// (embedding gather, causal masking) this file's narrower, one-off use
// doesn't justify adding to the shared engine.
//
// Reuses already-validated pieces rather than re-deriving them:
// seq_parallel::layernorm_forward/backward for both LayerNorms, and
// tensor_parallel_attn::single_head_attention_forward/backward for each
// attention head (with a causal mask applied to the score matrix before
// softmax — the backward formula is unchanged, since it's already
// implicit in the cached post-mask softmax output).
//
// Scope: single sequence per forward call (no batch dimension) — real
// systems batch, but batching multiplies this file's bookkeeping without
// changing what it validates (a real, gradient-checked transformer), so
// batch=1 keeps the hand-derived backward tractable to get right and
// verify.

#include <algorithm>
#include <cmath>
#include <random>
#include <vector>

#include "../distributed_training/autograd/matrix.h"
#include "../distributed_training/tensor_parallel_attn/attention.h"
#include "../distributed_training/seq_parallel/layernorm.h"

namespace transformer {

using distributed_training::AttentionCache;
using distributed_training::LayerNormCache;
using distributed_training::Matrix;
using distributed_training::layernorm_backward;
using distributed_training::layernorm_forward;
using distributed_training::single_head_attention_backward;
using distributed_training::softmax_rows;

struct TransformerConfig {
  int vocab_size;
  int d_model;
  int num_heads;
  int num_layers;
  int d_ff;
  int max_seq_len;

  int head_dim() const { return d_model / num_heads; }
};

// ---------------------------------------------------------------------
// Causal single-head attention: identical to
// tensor_parallel_attn::single_head_attention_forward except the score
// matrix is masked (upper triangle set to -inf) before softmax, so
// position i cannot attend to positions j > i. Backward is UNCHANGED from
// the unmasked version — the mask's effect is already fully captured in
// the cached post-mask softmax output `a`, which is all the backward
// pass reads.
// ---------------------------------------------------------------------
inline Matrix causal_attention_forward(const Matrix &q, const Matrix &k, const Matrix &v, AttentionCache &cache) {
  float scale = 1.0f / std::sqrt(static_cast<float>(q.cols()));
  Matrix s = q.matmul(k.transpose()) * scale;
  for (int i = 0; i < s.rows(); ++i)
    for (int j = i + 1; j < s.cols(); ++j) s(i, j) = -1e9f;
  Matrix a = softmax_rows(s);
  cache = AttentionCache{q, k, v, a};
  return a.matmul(v);
}

// ---------------------------------------------------------------------
// Parameters and gradients
// ---------------------------------------------------------------------

struct BlockParams {
  Matrix gamma1, beta1;  // LayerNorm 1, [1 x d_model]
  Matrix wq, wk, wv, wo; // [d_model x d_model]
  Matrix gamma2, beta2;  // LayerNorm 2
  Matrix w1, b1;         // [d_model x d_ff], [1 x d_ff]
  Matrix w2, b2;         // [d_ff x d_model], [1 x d_model]
};

struct ModelParams {
  Matrix token_emb; // [vocab_size x d_model]
  Matrix pos_emb;   // [max_seq_len x d_model]
  std::vector<BlockParams> blocks;
  Matrix final_gamma, final_beta;
  Matrix w_out; // [d_model x vocab_size], no bias

  TransformerConfig config;
};

inline BlockParams init_block(const TransformerConfig &cfg, std::mt19937 &rng) {
  float stddev = std::sqrt(2.0f / static_cast<float>(cfg.d_model));
  BlockParams p;
  p.gamma1 = Matrix(1, cfg.d_model); for (int j = 0; j < cfg.d_model; ++j) p.gamma1(0, j) = 1.0f;
  p.beta1 = Matrix(1, cfg.d_model);
  p.wq = Matrix::random(cfg.d_model, cfg.d_model, rng, stddev);
  p.wk = Matrix::random(cfg.d_model, cfg.d_model, rng, stddev);
  p.wv = Matrix::random(cfg.d_model, cfg.d_model, rng, stddev);
  p.wo = Matrix::random(cfg.d_model, cfg.d_model, rng, stddev);
  p.gamma2 = Matrix(1, cfg.d_model); for (int j = 0; j < cfg.d_model; ++j) p.gamma2(0, j) = 1.0f;
  p.beta2 = Matrix(1, cfg.d_model);
  float ff_stddev = std::sqrt(2.0f / static_cast<float>(cfg.d_model));
  p.w1 = Matrix::random(cfg.d_model, cfg.d_ff, rng, ff_stddev);
  p.b1 = Matrix(1, cfg.d_ff);
  float ff_stddev2 = std::sqrt(2.0f / static_cast<float>(cfg.d_ff));
  p.w2 = Matrix::random(cfg.d_ff, cfg.d_model, rng, ff_stddev2);
  p.b2 = Matrix(1, cfg.d_model);
  return p;
}

inline ModelParams init_model(const TransformerConfig &cfg, std::mt19937 &rng) {
  ModelParams m;
  m.config = cfg;
  float emb_stddev = 0.02f;
  m.token_emb = Matrix::random(cfg.vocab_size, cfg.d_model, rng, emb_stddev);
  m.pos_emb = Matrix::random(cfg.max_seq_len, cfg.d_model, rng, emb_stddev);
  for (int l = 0; l < cfg.num_layers; ++l) m.blocks.push_back(init_block(cfg, rng));
  m.final_gamma = Matrix(1, cfg.d_model); for (int j = 0; j < cfg.d_model; ++j) m.final_gamma(0, j) = 1.0f;
  m.final_beta = Matrix(1, cfg.d_model);
  float out_stddev = std::sqrt(1.0f / static_cast<float>(cfg.d_model));
  m.w_out = Matrix::random(cfg.d_model, cfg.vocab_size, rng, out_stddev);
  return m;
}

// Mirrors BlockParams/ModelParams shape, zero-initialized — accumulated
// into during backward.
inline BlockParams zero_block_grad(const TransformerConfig &cfg) {
  BlockParams g;
  g.gamma1 = Matrix(1, cfg.d_model); g.beta1 = Matrix(1, cfg.d_model);
  g.wq = Matrix(cfg.d_model, cfg.d_model); g.wk = Matrix(cfg.d_model, cfg.d_model);
  g.wv = Matrix(cfg.d_model, cfg.d_model); g.wo = Matrix(cfg.d_model, cfg.d_model);
  g.gamma2 = Matrix(1, cfg.d_model); g.beta2 = Matrix(1, cfg.d_model);
  g.w1 = Matrix(cfg.d_model, cfg.d_ff); g.b1 = Matrix(1, cfg.d_ff);
  g.w2 = Matrix(cfg.d_ff, cfg.d_model); g.b2 = Matrix(1, cfg.d_model);
  return g;
}

struct ModelGrads {
  Matrix token_emb, pos_emb;
  std::vector<BlockParams> blocks; // reuses BlockParams as the gradient container (same shapes)
  Matrix final_gamma, final_beta, w_out;
};

inline ModelGrads zero_model_grad(const TransformerConfig &cfg) {
  ModelGrads g;
  g.token_emb = Matrix(cfg.vocab_size, cfg.d_model);
  g.pos_emb = Matrix(cfg.max_seq_len, cfg.d_model);
  for (int l = 0; l < cfg.num_layers; ++l) g.blocks.push_back(zero_block_grad(cfg));
  g.final_gamma = Matrix(1, cfg.d_model);
  g.final_beta = Matrix(1, cfg.d_model);
  g.w_out = Matrix(cfg.d_model, cfg.vocab_size);
  return g;
}

// ---------------------------------------------------------------------
// Block forward/backward
// ---------------------------------------------------------------------

struct BlockCache {
  Matrix x_in;
  LayerNormCache ln1_cache;
  Matrix ln1_out, q, k, v, attn_concat, attn_out, x_after_attn;
  std::vector<AttentionCache> head_caches;
  LayerNormCache ln2_cache;
  Matrix ln2_out, mlp_pre1, mlp_hidden, mlp_out;
};

inline Matrix block_forward(const TransformerConfig &cfg, const BlockParams &p, const Matrix &x, BlockCache &cache) {
  int seq = x.rows();
  int head_dim = cfg.head_dim();
  cache.x_in = x;
  cache.ln1_out = layernorm_forward(x, p.gamma1, p.beta1, cache.ln1_cache);

  cache.q = cache.ln1_out.matmul(p.wq);
  cache.k = cache.ln1_out.matmul(p.wk);
  cache.v = cache.ln1_out.matmul(p.wv);

  Matrix concat(seq, cfg.d_model);
  cache.head_caches.resize(static_cast<size_t>(cfg.num_heads));
  for (int h = 0; h < cfg.num_heads; ++h) {
    Matrix qh(seq, head_dim), kh(seq, head_dim), vh(seq, head_dim);
    for (int i = 0; i < seq; ++i)
      for (int j = 0; j < head_dim; ++j) {
        qh(i, j) = cache.q(i, h * head_dim + j);
        kh(i, j) = cache.k(i, h * head_dim + j);
        vh(i, j) = cache.v(i, h * head_dim + j);
      }
    Matrix oh = causal_attention_forward(qh, kh, vh, cache.head_caches[static_cast<size_t>(h)]);
    for (int i = 0; i < seq; ++i)
      for (int j = 0; j < head_dim; ++j) concat(i, h * head_dim + j) = oh(i, j);
  }
  cache.attn_concat = concat;
  cache.attn_out = concat.matmul(p.wo);
  cache.x_after_attn = x + cache.attn_out;

  cache.ln2_out = layernorm_forward(cache.x_after_attn, p.gamma2, p.beta2, cache.ln2_cache);
  cache.mlp_pre1 = cache.ln2_out.matmul(p.w1).add_row_broadcast(p.b1);
  cache.mlp_hidden = cache.mlp_pre1.apply([](float v) { return v > 0.0f ? v : 0.0f; });
  cache.mlp_out = cache.mlp_hidden.matmul(p.w2).add_row_broadcast(p.b2);

  return cache.x_after_attn + cache.mlp_out;
}

inline Matrix block_backward(const TransformerConfig &cfg, const BlockParams &p, const BlockCache &cache,
                              const Matrix &dx_out, BlockParams &grad) {
  int seq = cache.x_in.rows();
  int head_dim = cfg.head_dim();

  // x_out = x_after_attn + mlp_out (residual)
  Matrix d_mlp_out = dx_out;
  grad.w2 = cache.mlp_hidden.transpose().matmul(d_mlp_out);
  grad.b2 = d_mlp_out.sum_rows();
  Matrix d_mlp_hidden = d_mlp_out.matmul(p.w2.transpose());
  Matrix relu_mask = cache.mlp_pre1.apply([](float v) { return v > 0.0f ? 1.0f : 0.0f; });
  Matrix d_mlp_pre1 = d_mlp_hidden.elementwise_mul(relu_mask);
  grad.w1 = cache.ln2_out.transpose().matmul(d_mlp_pre1);
  grad.b1 = d_mlp_pre1.sum_rows();
  Matrix d_ln2_out = d_mlp_pre1.matmul(p.w1.transpose());

  auto ln2_grads = layernorm_backward(cache.ln2_cache, d_ln2_out);
  grad.gamma2 = ln2_grads.dgamma;
  grad.beta2 = ln2_grads.dbeta;
  Matrix dx_after_attn = dx_out + ln2_grads.dx; // residual: direct path + through LN2/MLP

  // x_after_attn = x_in + attn_out (residual)
  Matrix d_attn_out = dx_after_attn;
  grad.wo = cache.attn_concat.transpose().matmul(d_attn_out);
  Matrix d_concat = d_attn_out.matmul(p.wo.transpose());

  Matrix dq(seq, cfg.d_model), dk(seq, cfg.d_model), dv(seq, cfg.d_model);
  for (int h = 0; h < cfg.num_heads; ++h) {
    Matrix d_oh(seq, head_dim);
    for (int i = 0; i < seq; ++i)
      for (int j = 0; j < head_dim; ++j) d_oh(i, j) = d_concat(i, h * head_dim + j);
    auto g = single_head_attention_backward(cache.head_caches[static_cast<size_t>(h)], d_oh);
    for (int i = 0; i < seq; ++i)
      for (int j = 0; j < head_dim; ++j) {
        dq(i, h * head_dim + j) = g.dq(i, j);
        dk(i, h * head_dim + j) = g.dk(i, j);
        dv(i, h * head_dim + j) = g.dv(i, j);
      }
  }
  grad.wq = cache.ln1_out.transpose().matmul(dq);
  grad.wk = cache.ln1_out.transpose().matmul(dk);
  grad.wv = cache.ln1_out.transpose().matmul(dv);
  Matrix d_ln1_out = dq.matmul(p.wq.transpose()) + dk.matmul(p.wk.transpose()) + dv.matmul(p.wv.transpose());

  auto ln1_grads = layernorm_backward(cache.ln1_cache, d_ln1_out);
  grad.gamma1 = ln1_grads.dgamma;
  grad.beta1 = ln1_grads.dbeta;

  return dx_after_attn + ln1_grads.dx; // residual: direct path + through LN1/attention
}

// ---------------------------------------------------------------------
// Full model forward/backward
// ---------------------------------------------------------------------

struct ModelCache {
  std::vector<int> token_ids;
  Matrix x0;
  std::vector<BlockCache> block_caches;
  Matrix x_final;
  LayerNormCache final_ln_cache;
  Matrix final_ln_out;
  Matrix logits;
};

inline Matrix model_forward(const ModelParams &p, const std::vector<int> &token_ids, ModelCache &cache) {
  const auto &cfg = p.config;
  int seq = static_cast<int>(token_ids.size());
  cache.token_ids = token_ids;
  cache.x0 = Matrix(seq, cfg.d_model);
  for (int i = 0; i < seq; ++i)
    for (int j = 0; j < cfg.d_model; ++j) cache.x0(i, j) = p.token_emb(token_ids[static_cast<size_t>(i)], j) + p.pos_emb(i, j);

  Matrix x = cache.x0;
  cache.block_caches.resize(static_cast<size_t>(cfg.num_layers));
  for (int l = 0; l < cfg.num_layers; ++l) x = block_forward(cfg, p.blocks[static_cast<size_t>(l)], x, cache.block_caches[static_cast<size_t>(l)]);
  cache.x_final = x;

  cache.final_ln_out = layernorm_forward(x, p.final_gamma, p.final_beta, cache.final_ln_cache);
  cache.logits = cache.final_ln_out.matmul(p.w_out);
  return cache.logits;
}

inline void model_backward(const ModelParams &p, const ModelCache &cache, const Matrix &dlogits, ModelGrads &grad) {
  const auto &cfg = p.config;
  grad.w_out = cache.final_ln_out.transpose().matmul(dlogits);
  Matrix d_final_ln_out = dlogits.matmul(p.w_out.transpose());

  auto final_ln_grads = layernorm_backward(cache.final_ln_cache, d_final_ln_out);
  grad.final_gamma = final_ln_grads.dgamma;
  grad.final_beta = final_ln_grads.dbeta;
  Matrix dx = final_ln_grads.dx;

  for (int l = cfg.num_layers - 1; l >= 0; --l) {
    dx = block_backward(cfg, p.blocks[static_cast<size_t>(l)], cache.block_caches[static_cast<size_t>(l)], dx, grad.blocks[static_cast<size_t>(l)]);
  }

  int seq = static_cast<int>(cache.token_ids.size());
  for (int i = 0; i < seq; ++i) {
    for (int j = 0; j < cfg.d_model; ++j) {
      grad.token_emb(cache.token_ids[static_cast<size_t>(i)], j) += dx(i, j);
      grad.pos_emb(i, j) += dx(i, j);
    }
  }
}

// Next-token cross-entropy: position i's logits predict token_ids[i+1],
// for i = 0..seq-2 (the last position has no target, standard causal LM).
struct LMLossResult {
  float loss;
  Matrix dlogits;
};

inline LMLossResult next_token_loss(const Matrix &logits, const std::vector<int> &token_ids) {
  int seq = logits.rows(), vocab = logits.cols();
  int n = seq - 1;
  Matrix dlogits(seq, vocab);
  float total = 0.0f;
  for (int i = 0; i < n; ++i) {
    float max_v = logits(i, 0);
    for (int v = 1; v < vocab; ++v) max_v = std::max(max_v, logits(i, v));
    float denom = 0.0f;
    for (int v = 0; v < vocab; ++v) denom += std::exp(logits(i, v) - max_v);
    int target = token_ids[static_cast<size_t>(i + 1)];
    float target_prob = std::exp(logits(i, target) - max_v) / denom;
    total += -std::log(std::max(target_prob, 1e-9f));
    for (int v = 0; v < vocab; ++v) {
      float prob = std::exp(logits(i, v) - max_v) / denom;
      dlogits(i, v) = (prob - (v == target ? 1.0f : 0.0f)) / static_cast<float>(n);
    }
  }
  return LMLossResult{total / static_cast<float>(n), dlogits};
}

// Per-position log-probabilities (softmax(logits), logged) — used by
// SFT/reward-model/PPO/DPO to score a specific continuation.
inline Matrix log_probs(const Matrix &logits) {
  Matrix out(logits.rows(), logits.cols());
  for (int i = 0; i < logits.rows(); ++i) {
    float max_v = logits(i, 0);
    for (int v = 1; v < logits.cols(); ++v) max_v = std::max(max_v, logits(i, v));
    float denom = 0.0f;
    for (int v = 0; v < logits.cols(); ++v) denom += std::exp(logits(i, v) - max_v);
    for (int v = 0; v < logits.cols(); ++v) out(i, v) = (logits(i, v) - max_v) - std::log(denom);
  }
  return out;
}

inline void sgd_step(ModelParams &p, const ModelGrads &g, float lr) {
  p.token_emb.add_inplace(g.token_emb, -lr);
  p.pos_emb.add_inplace(g.pos_emb, -lr);
  for (size_t l = 0; l < p.blocks.size(); ++l) {
    auto &b = p.blocks[l];
    const auto &gb = g.blocks[l];
    b.gamma1.add_inplace(gb.gamma1, -lr); b.beta1.add_inplace(gb.beta1, -lr);
    b.wq.add_inplace(gb.wq, -lr); b.wk.add_inplace(gb.wk, -lr); b.wv.add_inplace(gb.wv, -lr); b.wo.add_inplace(gb.wo, -lr);
    b.gamma2.add_inplace(gb.gamma2, -lr); b.beta2.add_inplace(gb.beta2, -lr);
    b.w1.add_inplace(gb.w1, -lr); b.b1.add_inplace(gb.b1, -lr);
    b.w2.add_inplace(gb.w2, -lr); b.b2.add_inplace(gb.b2, -lr);
  }
  p.final_gamma.add_inplace(g.final_gamma, -lr);
  p.final_beta.add_inplace(g.final_beta, -lr);
  p.w_out.add_inplace(g.w_out, -lr);
}

} // namespace transformer
