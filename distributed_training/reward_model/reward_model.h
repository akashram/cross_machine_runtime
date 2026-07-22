#pragma once

// Reward model training: PLAN.md step 23. Wraps the real transformer/
// model's body (embeddings, blocks, final LayerNorm — reused verbatim,
// same "reuse already-validated pieces" pattern as sft/) and replaces its
// vocab projection with a scalar reward head applied to the LAST token's
// final-LayerNorm output (the standard RLHF reward-model architecture:
// under causal masking, the last position is the only one that has
// attended to the entire prompt+response, so it's the natural pooled
// summary). Trained with the Bradley-Terry pairwise-preference objective
// on (chosen, rejected) response pairs, data sharded across simulated
// data-parallel ranks with gradient all-reduce (same pattern as sft/).

#include <cmath>
#include <string>
#include <vector>

#include "../../transformer/transformer_model.h"

namespace distributed_training {

struct PreferencePair {
  std::string prompt;
  std::string response_chosen;   // preferred response
  std::string response_rejected; // dispreferred response
};

// Body (embeddings/blocks/final LN) is a full transformer::ModelParams for
// simplicity — its vocab projection (w_out) is allocated but never used by
// the reward head, an acceptable toy-scale waste in exchange for reusing
// transformer::model_forward/init_model unmodified.
struct RewardModelParams {
  transformer::ModelParams body;
  transformer::Matrix w_reward; // [d_model x 1]
  transformer::Matrix b_reward; // [1 x 1]
};

struct RewardModelGrads {
  transformer::ModelGrads body;
  transformer::Matrix w_reward, b_reward;
};

inline RewardModelParams init_reward_model(const transformer::TransformerConfig &cfg, std::mt19937 &rng) {
  RewardModelParams m;
  m.body = transformer::init_model(cfg, rng);
  float stddev = std::sqrt(1.0f / static_cast<float>(cfg.d_model));
  m.w_reward = transformer::Matrix::random(cfg.d_model, 1, rng, stddev);
  m.b_reward = transformer::Matrix(1, 1);
  return m;
}

inline RewardModelGrads zero_reward_model_grad(const transformer::TransformerConfig &cfg) {
  RewardModelGrads g;
  g.body = transformer::zero_model_grad(cfg);
  g.w_reward = transformer::Matrix(cfg.d_model, 1);
  g.b_reward = transformer::Matrix(1, 1);
  return g;
}

struct RewardModelCache {
  transformer::ModelCache body_cache;
  transformer::Matrix pooled; // [1 x d_model] — final_ln_out's last row
};

// Runs the full transformer body forward (including its unused vocab
// logits — see RewardModelParams's comment) then reads out a scalar
// reward from the last token's LayerNorm-ed hidden state.
inline float reward_model_forward(const RewardModelParams &p, const std::vector<int> &token_ids, RewardModelCache &cache) {
  transformer::model_forward(p.body, token_ids, cache.body_cache);
  int d = p.body.config.d_model;
  int last = static_cast<int>(token_ids.size()) - 1;
  cache.pooled = transformer::Matrix(1, d);
  for (int j = 0; j < d; ++j) cache.pooled(0, j) = cache.body_cache.final_ln_out(last, j);
  float reward = p.b_reward(0, 0);
  for (int j = 0; j < d; ++j) reward += cache.pooled(0, j) * p.w_reward(j, 0);
  return reward;
}

// Backpropagates a scalar gradient (dreward) through the reward head, the
// final LayerNorm, and the transformer body's blocks/embeddings — the
// vocab projection (w_out) is intentionally skipped (never touched by the
// reward head), so grad.body.w_out stays zero. Mirrors
// transformer::model_backward's tail, reusing the same validated
// block_backward/layernorm_backward primitives, just entering the
// backward chain at final_ln_out instead of at dlogits/w_out.
inline void reward_model_backward(const RewardModelParams &p, const RewardModelCache &cache, float dreward, RewardModelGrads &grad) {
  const auto &cfg = p.body.config;
  int seq = static_cast<int>(cache.body_cache.token_ids.size());
  int last = seq - 1;

  for (int j = 0; j < cfg.d_model; ++j) grad.w_reward(j, 0) += cache.pooled(0, j) * dreward;
  grad.b_reward(0, 0) += dreward;

  transformer::Matrix d_final_ln_out(seq, cfg.d_model); // zero everywhere except the last row
  for (int j = 0; j < cfg.d_model; ++j) d_final_ln_out(last, j) = p.w_reward(j, 0) * dreward;

  auto final_ln_grads = transformer::layernorm_backward(cache.body_cache.final_ln_cache, d_final_ln_out);
  grad.body.final_gamma.add_inplace(final_ln_grads.dgamma);
  grad.body.final_beta.add_inplace(final_ln_grads.dbeta);
  transformer::Matrix dx = final_ln_grads.dx;

  for (int l = cfg.num_layers - 1; l >= 0; --l)
    dx = transformer::block_backward(cfg, p.body.blocks[static_cast<size_t>(l)], cache.body_cache.block_caches[static_cast<size_t>(l)], dx,
                                      grad.body.blocks[static_cast<size_t>(l)]);

  for (int i = 0; i < seq; ++i)
    for (int j = 0; j < cfg.d_model; ++j) {
      grad.body.token_emb(cache.body_cache.token_ids[static_cast<size_t>(i)], j) += dx(i, j);
      grad.body.pos_emb(i, j) += dx(i, j);
    }
}

inline void accumulate_reward_grad(RewardModelGrads &a, const RewardModelGrads &b) {
  transformer::accumulate_grad(a.body, b.body);
  a.w_reward.add_inplace(b.w_reward);
  a.b_reward.add_inplace(b.b_reward);
}

inline void sgd_step_reward(RewardModelParams &p, const RewardModelGrads &g, float lr) {
  transformer::sgd_step(p.body, g.body, lr);
  p.w_reward.add_inplace(g.w_reward, -lr);
  p.b_reward.add_inplace(g.b_reward, -lr);
}

// Appends w_reward/b_reward onto transformer::flatten_grad(g.body)'s
// buffer so the whole reward model's gradient (body + head) all-reduces
// in one ring_allreduce call.
inline std::vector<float> flatten_reward_grad(const RewardModelGrads &g) {
  std::vector<float> flat = transformer::flatten_grad(g.body);
  transformer::append_flat(flat, g.w_reward);
  transformer::append_flat(flat, g.b_reward);
  return flat;
}

inline void unflatten_into_reward_grad(RewardModelGrads &g, const std::vector<float> &flat) {
  size_t idx = 0;
  transformer::unflatten_into_grad(g.body, flat, idx);
  transformer::read_flat(g.w_reward, flat, idx);
  transformer::read_flat(g.b_reward, flat, idx);
}

// Bradley-Terry pairwise preference model: P(chosen > rejected) =
// sigmoid(reward_chosen - reward_rejected); loss = -log P. Numerically
// stable softplus(-d) formulation (avoids overflow in exp(-d) for very
// negative d). Gradient derivation: d/dd[-log sigmoid(d)] = sigmoid(d) - 1
// where d = reward_chosen - reward_rejected, so
// dL/dreward_chosen = sigmoid(d) - 1 and dL/dreward_rejected = 1 - sigmoid(d)
// (equal and opposite, since d is linear in both).
struct BradleyTerryLossResult {
  float loss;
  float dreward_chosen;
  float dreward_rejected;
};

inline BradleyTerryLossResult bradley_terry_loss(float reward_chosen, float reward_rejected) {
  float d = reward_chosen - reward_rejected;
  float loss = d >= 0.0f ? std::log1p(std::exp(-d)) : (-d + std::log1p(std::exp(d)));
  float sig = 1.0f / (1.0f + std::exp(-d));
  return BradleyTerryLossResult{loss, sig - 1.0f, 1.0f - sig};
}

} // namespace distributed_training
