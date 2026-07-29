#pragma once

// PLAN.md Phase 13 step 1: embedding model. A bidirectional (non-causal)
// encoder built from the same validated blocks transformer/transformer_model.h
// uses -- distributed_training/seq_parallel's LayerNorm and
// distributed_training/tensor_parallel_attn's single-head attention
// primitive -- but NOT transformer/transformer_model.h itself: that file's
// block_forward hard-codes causal_attention_forward (upper-triangle
// masking), which is wrong for an embedding encoder that must see the
// whole sequence in both directions. Duplicating the block here (swapping
// in the unmasked attention primitive) avoids parameterizing/touching
// transformer_model.h, which distributed_training's RLHF steps 22-25
// already depend on and gradient-check against.
//
// Architecture: token+positional embedding -> N pre-LN blocks (bidirectional
// multi-head self-attention + residual, MLP + residual) -> final LayerNorm
// -> mean pooling over sequence positions -> linear projection (no bias,
// same convention as transformer_model.h's w_out) -> L2 normalize. Trained
// with a symmetric (query<->document) InfoNCE contrastive loss (Oord et al.
// 2018; symmetric formulation as in Radford et al. 2021's CLIP) over
// in-batch negatives, using dot product of L2-normalized vectors as cosine
// similarity.
//
// Scope: same as transformer_model.h -- single sequence per forward call,
// hand-derived backward (not autograd.h's generic tape), character-level
// tokenization.

#include <cmath>
#include <random>
#include <vector>

#include "../../distributed_training/autograd/matrix.h"
#include "../../distributed_training/seq_parallel/layernorm.h"
#include "../../distributed_training/tensor_parallel_attn/attention.h"

namespace rag {

using distributed_training::AttentionCache;
using distributed_training::LayerNormCache;
using distributed_training::Matrix;
using distributed_training::layernorm_backward;
using distributed_training::layernorm_forward;
using distributed_training::single_head_attention_backward;
using distributed_training::single_head_attention_forward;

struct EncoderConfig {
  int vocab_size;
  int d_model;
  int num_heads;
  int num_layers;
  int d_ff;
  int max_seq_len;
  int embed_dim;

  int head_dim() const { return d_model / num_heads; }
};

// ---------------------------------------------------------------------
// Parameters and gradients
// ---------------------------------------------------------------------

struct EncoderBlockParams {
  Matrix gamma1, beta1;  // LayerNorm 1
  Matrix wq, wk, wv, wo; // [d_model x d_model]
  Matrix gamma2, beta2;  // LayerNorm 2
  Matrix w1, b1;         // [d_model x d_ff], [1 x d_ff]
  Matrix w2, b2;         // [d_ff x d_model], [1 x d_model]
};

struct EncoderParams {
  Matrix token_emb; // [vocab_size x d_model]
  Matrix pos_emb;   // [max_seq_len x d_model]
  std::vector<EncoderBlockParams> blocks;
  Matrix final_gamma, final_beta;
  Matrix w_proj; // [d_model x embed_dim], no bias

  EncoderConfig config;
};

inline EncoderBlockParams init_encoder_block(const EncoderConfig &cfg, std::mt19937 &rng) {
  float stddev = std::sqrt(2.0f / static_cast<float>(cfg.d_model));
  EncoderBlockParams p;
  p.gamma1 = Matrix(1, cfg.d_model); for (int j = 0; j < cfg.d_model; ++j) p.gamma1(0, j) = 1.0f;
  p.beta1 = Matrix(1, cfg.d_model);
  p.wq = Matrix::random(cfg.d_model, cfg.d_model, rng, stddev);
  p.wk = Matrix::random(cfg.d_model, cfg.d_model, rng, stddev);
  p.wv = Matrix::random(cfg.d_model, cfg.d_model, rng, stddev);
  p.wo = Matrix::random(cfg.d_model, cfg.d_model, rng, stddev);
  p.gamma2 = Matrix(1, cfg.d_model); for (int j = 0; j < cfg.d_model; ++j) p.gamma2(0, j) = 1.0f;
  p.beta2 = Matrix(1, cfg.d_model);
  p.w1 = Matrix::random(cfg.d_model, cfg.d_ff, rng, stddev);
  p.b1 = Matrix(1, cfg.d_ff);
  float ff_stddev2 = std::sqrt(2.0f / static_cast<float>(cfg.d_ff));
  p.w2 = Matrix::random(cfg.d_ff, cfg.d_model, rng, ff_stddev2);
  p.b2 = Matrix(1, cfg.d_model);
  return p;
}

inline EncoderParams init_encoder(const EncoderConfig &cfg, std::mt19937 &rng) {
  EncoderParams m;
  m.config = cfg;
  float emb_stddev = 0.02f;
  m.token_emb = Matrix::random(cfg.vocab_size, cfg.d_model, rng, emb_stddev);
  m.pos_emb = Matrix::random(cfg.max_seq_len, cfg.d_model, rng, emb_stddev);
  for (int l = 0; l < cfg.num_layers; ++l) m.blocks.push_back(init_encoder_block(cfg, rng));
  m.final_gamma = Matrix(1, cfg.d_model); for (int j = 0; j < cfg.d_model; ++j) m.final_gamma(0, j) = 1.0f;
  m.final_beta = Matrix(1, cfg.d_model);
  float proj_stddev = std::sqrt(1.0f / static_cast<float>(cfg.d_model));
  m.w_proj = Matrix::random(cfg.d_model, cfg.embed_dim, rng, proj_stddev);
  return m;
}

inline EncoderBlockParams zero_encoder_block_grad(const EncoderConfig &cfg) {
  EncoderBlockParams g;
  g.gamma1 = Matrix(1, cfg.d_model); g.beta1 = Matrix(1, cfg.d_model);
  g.wq = Matrix(cfg.d_model, cfg.d_model); g.wk = Matrix(cfg.d_model, cfg.d_model);
  g.wv = Matrix(cfg.d_model, cfg.d_model); g.wo = Matrix(cfg.d_model, cfg.d_model);
  g.gamma2 = Matrix(1, cfg.d_model); g.beta2 = Matrix(1, cfg.d_model);
  g.w1 = Matrix(cfg.d_model, cfg.d_ff); g.b1 = Matrix(1, cfg.d_ff);
  g.w2 = Matrix(cfg.d_ff, cfg.d_model); g.b2 = Matrix(1, cfg.d_model);
  return g;
}

struct EncoderGrads {
  Matrix token_emb, pos_emb;
  std::vector<EncoderBlockParams> blocks; // reused as gradient container (same shapes)
  Matrix final_gamma, final_beta, w_proj;
};

inline EncoderGrads zero_encoder_grad(const EncoderConfig &cfg) {
  EncoderGrads g;
  g.token_emb = Matrix(cfg.vocab_size, cfg.d_model);
  g.pos_emb = Matrix(cfg.max_seq_len, cfg.d_model);
  for (int l = 0; l < cfg.num_layers; ++l) g.blocks.push_back(zero_encoder_block_grad(cfg));
  g.final_gamma = Matrix(1, cfg.d_model);
  g.final_beta = Matrix(1, cfg.d_model);
  g.w_proj = Matrix(cfg.d_model, cfg.embed_dim);
  return g;
}

inline void accumulate_encoder_grad(EncoderGrads &a, const EncoderGrads &b) {
  a.token_emb.add_inplace(b.token_emb);
  a.pos_emb.add_inplace(b.pos_emb);
  for (size_t l = 0; l < a.blocks.size(); ++l) {
    auto &ab = a.blocks[l];
    const auto &bb = b.blocks[l];
    ab.gamma1.add_inplace(bb.gamma1); ab.beta1.add_inplace(bb.beta1);
    ab.wq.add_inplace(bb.wq); ab.wk.add_inplace(bb.wk); ab.wv.add_inplace(bb.wv); ab.wo.add_inplace(bb.wo);
    ab.gamma2.add_inplace(bb.gamma2); ab.beta2.add_inplace(bb.beta2);
    ab.w1.add_inplace(bb.w1); ab.b1.add_inplace(bb.b1); ab.w2.add_inplace(bb.w2); ab.b2.add_inplace(bb.b2);
  }
  a.final_gamma.add_inplace(b.final_gamma);
  a.final_beta.add_inplace(b.final_beta);
  a.w_proj.add_inplace(b.w_proj);
}

inline void encoder_sgd_step(EncoderParams &p, const EncoderGrads &g, float lr) {
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
  p.w_proj.add_inplace(g.w_proj, -lr);
}

// ---------------------------------------------------------------------
// Block forward/backward (bidirectional -- UNMASKED attention, the only
// real difference from transformer_model.h's block_forward/backward)
// ---------------------------------------------------------------------

struct EncoderBlockCache {
  Matrix x_in;
  LayerNormCache ln1_cache;
  Matrix ln1_out, q, k, v, attn_concat, attn_out, x_after_attn;
  std::vector<AttentionCache> head_caches;
  LayerNormCache ln2_cache;
  Matrix ln2_out, mlp_pre1, mlp_hidden, mlp_out;
};

inline Matrix encoder_block_forward(const EncoderConfig &cfg, const EncoderBlockParams &p, const Matrix &x,
                                     EncoderBlockCache &cache) {
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
    Matrix oh = single_head_attention_forward(qh, kh, vh, cache.head_caches[static_cast<size_t>(h)]);
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

inline Matrix encoder_block_backward(const EncoderConfig &cfg, const EncoderBlockParams &p, const EncoderBlockCache &cache,
                                      const Matrix &dx_out, EncoderBlockParams &grad) {
  int seq = cache.x_in.rows();
  int head_dim = cfg.head_dim();

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
  Matrix dx_after_attn = dx_out + ln2_grads.dx;

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

  return dx_after_attn + ln1_grads.dx;
}

// ---------------------------------------------------------------------
// Full encoder forward/backward: blocks -> final LayerNorm -> mean pool
// -> linear projection -> L2 normalize.
// ---------------------------------------------------------------------

struct EncoderCache {
  std::vector<int> token_ids;
  Matrix x0;
  std::vector<EncoderBlockCache> block_caches;
  Matrix x_final;
  LayerNormCache final_ln_cache;
  Matrix final_ln_out; // [seq x d_model]
  Matrix pooled;       // [1 x d_model], mean over rows of final_ln_out
  Matrix projected;    // [1 x embed_dim] = pooled * w_proj
  float proj_norm;     // ||projected||_2
  Matrix embedding;    // [1 x embed_dim] = projected / proj_norm
};

// Fixed-size embedding for one token sequence. The pooling step (mean over
// sequence positions) and the L2 normalize are both parameter-free, so
// their backward is pure calculus, not another learned layer.
inline Matrix encode_forward(const EncoderParams &p, const std::vector<int> &token_ids, EncoderCache &cache) {
  const auto &cfg = p.config;
  int seq = static_cast<int>(token_ids.size());
  cache.token_ids = token_ids;
  cache.x0 = Matrix(seq, cfg.d_model);
  for (int i = 0; i < seq; ++i)
    for (int j = 0; j < cfg.d_model; ++j)
      cache.x0(i, j) = p.token_emb(token_ids[static_cast<size_t>(i)], j) + p.pos_emb(i, j);

  Matrix x = cache.x0;
  cache.block_caches.resize(static_cast<size_t>(cfg.num_layers));
  for (int l = 0; l < cfg.num_layers; ++l)
    x = encoder_block_forward(cfg, p.blocks[static_cast<size_t>(l)], x, cache.block_caches[static_cast<size_t>(l)]);
  cache.x_final = x;

  cache.final_ln_out = layernorm_forward(x, p.final_gamma, p.final_beta, cache.final_ln_cache);

  cache.pooled = Matrix(1, cfg.d_model);
  for (int i = 0; i < seq; ++i)
    for (int j = 0; j < cfg.d_model; ++j) cache.pooled(0, j) += cache.final_ln_out(i, j);
  cache.pooled = cache.pooled * (1.0f / static_cast<float>(seq));

  cache.projected = cache.pooled.matmul(p.w_proj);
  float norm_sq = 0.0f;
  for (int j = 0; j < cfg.embed_dim; ++j) norm_sq += cache.projected(0, j) * cache.projected(0, j);
  cache.proj_norm = std::sqrt(std::max(norm_sq, 1e-12f));
  cache.embedding = cache.projected * (1.0f / cache.proj_norm);
  return cache.embedding;
}

inline void encode_backward(const EncoderParams &p, const EncoderCache &cache, const Matrix &d_embedding,
                             EncoderGrads &grad) {
  const auto &cfg = p.config;
  int seq = static_cast<int>(cache.token_ids.size());

  // L2 normalize backward: embedding = projected / norm. For unit vector e
  // and upstream gradient de: d_projected = (de - e * dot(e, de)) / norm.
  float dot = 0.0f;
  for (int j = 0; j < cfg.embed_dim; ++j) dot += cache.embedding(0, j) * d_embedding(0, j);
  Matrix d_projected(1, cfg.embed_dim);
  for (int j = 0; j < cfg.embed_dim; ++j)
    d_projected(0, j) = (d_embedding(0, j) - cache.embedding(0, j) * dot) / cache.proj_norm;

  grad.w_proj = cache.pooled.transpose().matmul(d_projected);
  Matrix d_pooled = d_projected.matmul(p.w_proj.transpose());

  // Mean-pool backward: gradient spreads equally to every position.
  Matrix d_final_ln_out(seq, cfg.d_model);
  float inv_seq = 1.0f / static_cast<float>(seq);
  for (int i = 0; i < seq; ++i)
    for (int j = 0; j < cfg.d_model; ++j) d_final_ln_out(i, j) = d_pooled(0, j) * inv_seq;

  auto final_ln_grads = layernorm_backward(cache.final_ln_cache, d_final_ln_out);
  grad.final_gamma = final_ln_grads.dgamma;
  grad.final_beta = final_ln_grads.dbeta;
  Matrix dx = final_ln_grads.dx;

  for (int l = cfg.num_layers - 1; l >= 0; --l)
    dx = encoder_block_backward(cfg, p.blocks[static_cast<size_t>(l)], cache.block_caches[static_cast<size_t>(l)], dx,
                                 grad.blocks[static_cast<size_t>(l)]);

  for (int i = 0; i < seq; ++i)
    for (int j = 0; j < cfg.d_model; ++j) {
      grad.token_emb(cache.token_ids[static_cast<size_t>(i)], j) += dx(i, j);
      grad.pos_emb(i, j) += dx(i, j);
    }
}

// ---------------------------------------------------------------------
// Symmetric InfoNCE contrastive loss (Oord et al. 2018; symmetric
// query<->document formulation as in Radford et al. 2021's CLIP) over
// in-batch negatives. Embeddings are assumed already L2-normalized (true
// of encode_forward's output), so their dot product IS cosine similarity.
// ---------------------------------------------------------------------

// Cross-entropy loss + gradient of a square similarity matrix `s` against
// the diagonal target (row i's positive is column i) -- same softmax
// cross-entropy shape as transformer_model.h's next_token_loss, but the
// target here is always the row index itself, not a looked-up token id.
inline float diag_softmax_ce(const Matrix &s, Matrix &ds) {
  int n = s.rows();
  ds = Matrix(n, n);
  float total = 0.0f;
  for (int i = 0; i < n; ++i) {
    float max_v = s(i, 0);
    for (int j = 1; j < n; ++j) max_v = std::max(max_v, s(i, j));
    float denom = 0.0f;
    for (int j = 0; j < n; ++j) denom += std::exp(s(i, j) - max_v);
    float target_prob = std::exp(s(i, i) - max_v) / denom;
    total += -std::log(std::max(target_prob, 1e-9f));
    for (int j = 0; j < n; ++j) {
      float prob = std::exp(s(i, j) - max_v) / denom;
      ds(i, j) = (prob - (j == i ? 1.0f : 0.0f)) / static_cast<float>(n);
    }
  }
  return total / static_cast<float>(n);
}

struct InfoNCEResult {
  float loss;
  std::vector<Matrix> d_query_emb; // one [1 x embed_dim] per batch element
  std::vector<Matrix> d_doc_emb;
};

inline InfoNCEResult info_nce_loss(const std::vector<Matrix> &query_embs, const std::vector<Matrix> &doc_embs,
                                    float temperature) {
  int batch = static_cast<int>(query_embs.size());
  int embed_dim = query_embs[0].cols();
  Matrix eq(batch, embed_dim), ed(batch, embed_dim);
  for (int i = 0; i < batch; ++i)
    for (int j = 0; j < embed_dim; ++j) {
      eq(i, j) = query_embs[static_cast<size_t>(i)](0, j);
      ed(i, j) = doc_embs[static_cast<size_t>(i)](0, j);
    }

  Matrix s = eq.matmul(ed.transpose()) * (1.0f / temperature); // [batch x batch]
  Matrix ds_q2d;
  float loss_q2d = diag_softmax_ce(s, ds_q2d);
  Matrix st = s.transpose();
  Matrix ds_d2q_t;
  float loss_d2q = diag_softmax_ce(st, ds_d2q_t);

  Matrix ds = ds_q2d * 0.5f + ds_d2q_t.transpose() * 0.5f;
  Matrix d_eq = ds.matmul(ed) * (1.0f / temperature);
  Matrix d_ed = ds.transpose().matmul(eq) * (1.0f / temperature);

  InfoNCEResult result;
  result.loss = 0.5f * (loss_q2d + loss_d2q);
  result.d_query_emb.resize(static_cast<size_t>(batch));
  result.d_doc_emb.resize(static_cast<size_t>(batch));
  for (int i = 0; i < batch; ++i) {
    Matrix dq(1, embed_dim), dd(1, embed_dim);
    for (int j = 0; j < embed_dim; ++j) {
      dq(0, j) = d_eq(i, j);
      dd(0, j) = d_ed(i, j);
    }
    result.d_query_emb[static_cast<size_t>(i)] = dq;
    result.d_doc_emb[static_cast<size_t>(i)] = dd;
  }
  return result;
}

inline float cosine_similarity(const Matrix &a, const Matrix &b) {
  float dot = 0.0f;
  for (int j = 0; j < a.cols(); ++j) dot += a(0, j) * b(0, j);
  return dot;
}

} // namespace rag
