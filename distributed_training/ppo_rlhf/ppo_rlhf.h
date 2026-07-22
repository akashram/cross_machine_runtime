#pragma once

// PPO-based RLHF: PLAN.md step 24. Policy (SFT-initialized), a frozen
// reference model (a deep copy of the SFT policy, never updated — the KL
// penalty's anchor), a frozen reward model (step 23's Bradley-Terry
// reward model, trained separately and not touched again here), and a
// critic (value function) trained alongside the policy.
//
// Scope: this domain's response is always exactly ONE token (a single
// digit, same as sft/ and reward_model/), so an "episode" is a single
// prompt -> single sampled action -> single terminal reward. That's not a
// simplification bolted on to dodge complexity — it's what the domain
// actually is — but it does mean this is a single-step (contextual
// bandit) instance of PPO: no multi-token credit assignment, no GAE
// across timesteps, no discounting. What's still fully real: the clipped
// surrogate objective (including the exact PPO subgradient at the clip
// boundary), the KL penalty against a frozen reference baked into the
// reward signal, a learned critic/value baseline for the advantage, and
// data-parallel gradient sync via ring_allreduce (in ppo_rlhf_test.cpp,
// matching sft/ and reward_model/'s pattern — this header stays free of
// networking/threading, same layering as sft_trainer.h/reward_model.h).

#include <algorithm>
#include <cmath>
#include <random>
#include <vector>

#include "../reward_model/reward_model.h"
#include "../../transformer/transformer_model.h"

namespace distributed_training {

// The critic is architecturally identical to the reward model (transformer
// body + scalar head) — same reason reward_model.h gives for reusing
// transformer::ModelParams's body: this is a value function V(prompt),
// which is exactly reward_model_forward's shape when called on prompt
// tokens alone (pooled at the LAST token of whatever sequence you pass
// it — prompt-only here, prompt+response for the reward model). Trained
// by MSE regression instead of Bradley-Terry.
using CriticParams = RewardModelParams;
using CriticGrads = RewardModelGrads;
inline CriticParams init_critic(const transformer::TransformerConfig &cfg, std::mt19937 &rng) { return init_reward_model(cfg, rng); }
inline CriticGrads zero_critic_grad(const transformer::TransformerConfig &cfg) { return zero_reward_model_grad(cfg); }

struct PolicyLossResult {
  float loss;
  float d_new_logprob; // d(loss) / d(new_logprob(action)), to backprop into the policy
};

// Clipped PPO surrogate: ratio = pi_new(a) / pi_old(a);
// loss = -min(ratio * advantage, clip(ratio, 1-eps, 1+eps) * advantage).
// The gradient follows PPO's standard subgradient: it flows through
// whichever of {unclipped, clipped} branch achieves the min, and is
// exactly zero when the clip is BOTH active AND the selected branch —
// that's what stops PPO from pushing the ratio further outside the trust
// region once it's already been clipped in the direction that would
// increase the objective.
inline PolicyLossResult ppo_policy_loss(float new_logprob, float old_logprob, float advantage, float clip_eps) {
  float ratio = std::exp(new_logprob - old_logprob);
  float unclipped = ratio * advantage;
  float clipped_ratio = std::clamp(ratio, 1.0f - clip_eps, 1.0f + clip_eps);
  float clipped = clipped_ratio * advantage;

  float loss, d_loss_d_ratio;
  if (unclipped <= clipped) {
    loss = -unclipped;
    d_loss_d_ratio = -advantage;
  } else {
    loss = -clipped;
    bool ratio_was_clamped = (ratio < 1.0f - clip_eps) || (ratio > 1.0f + clip_eps);
    d_loss_d_ratio = ratio_was_clamped ? 0.0f : -advantage;
  }
  float d_loss_d_new_logprob = d_loss_d_ratio * ratio; // d(ratio)/d(new_logprob) = ratio, since ratio = exp(new_logprob - old_logprob)
  return PolicyLossResult{loss, d_loss_d_new_logprob};
}

// Turns a scalar d(loss)/d(logprob_at_action) into a full [seq x vocab]
// dlogits matrix (nonzero only at `position`), via the log-softmax
// gradient d(log p_a)/d(logit_v) = 1[v==a] - p_v — reuses
// transformer::log_probs (already-validated log-softmax) for p_v.
inline transformer::Matrix policy_dlogits(const transformer::Matrix &logits, int position, int action, float d_loss_d_logprob) {
  transformer::Matrix dlogits(logits.rows(), logits.cols());
  transformer::Matrix lp = transformer::log_probs(logits);
  for (int v = 0; v < logits.cols(); ++v) {
    float p_v = std::exp(lp(position, v));
    float d_logprob_d_logit = (v == action ? 1.0f : 0.0f) - p_v;
    dlogits(position, v) = d_loss_d_logprob * d_logprob_d_logit;
  }
  return dlogits;
}

// One rollout sample: everything computed at ROLLOUT time using the
// current "behavior" policy/critic, held fixed across the several PPO
// update epochs that reuse this same batch (standard PPO: the advantage
// and the old_logprob used in the ratio are NOT recomputed each epoch,
// only new_logprob and the critic's value prediction are).
struct RolloutSample {
  std::vector<int> prompt_tokens;
  int action;          // sampled response token id
  float old_logprob;   // log pi_behavior(action | prompt) — fixed for all PPO epochs on this batch
  float ref_logprob;   // log pi_ref(action | prompt), frozen reference model
  float reward_score;  // reward_model(prompt + action), frozen reward model
  float shaped_reward; // reward_score - kl_coef * (old_logprob - ref_logprob) — the PPO return/regression target
  float advantage;     // shaped_reward - value_pred (value_pred from the critic AT ROLLOUT TIME)
};

// Samples one action from the current policy (temperature-1.0 softmax
// over the full vocab at the position right after the prompt — no hard
// restriction to "valid" tokens, so a policy that drifts into producing
// nonsense is visible in the reward/KL trace, not hidden by the action
// space), scores it with the frozen reward model, KL-penalizes against
// the frozen reference model, and baselines it against the critic.
inline RolloutSample rollout_one(const transformer::ModelParams &policy, const transformer::ModelParams &ref_model,
                                  const RewardModelParams &reward_model, const CriticParams &critic,
                                  const std::vector<int> &prompt_tokens, float kl_coef, std::mt19937 &rng) {
  transformer::ModelCache policy_cache;
  transformer::Matrix policy_logits = transformer::model_forward(policy, prompt_tokens, policy_cache);
  int pos = static_cast<int>(prompt_tokens.size()) - 1;

  transformer::Matrix policy_lp = transformer::log_probs(policy_logits);
  std::vector<float> probs(static_cast<size_t>(policy_logits.cols()));
  for (int v = 0; v < policy_logits.cols(); ++v) probs[static_cast<size_t>(v)] = std::exp(policy_lp(pos, v));
  std::discrete_distribution<int> dist(probs.begin(), probs.end());
  int action = dist(rng);
  float old_logprob = policy_lp(pos, action);

  transformer::ModelCache ref_cache;
  transformer::Matrix ref_logits = transformer::model_forward(ref_model, prompt_tokens, ref_cache);
  float ref_logprob = transformer::log_probs(ref_logits)(pos, action);

  std::vector<int> full_tokens = prompt_tokens;
  full_tokens.push_back(action);
  RewardModelCache reward_cache;
  float reward_score = reward_model_forward(reward_model, full_tokens, reward_cache);

  RewardModelCache critic_cache;
  float value_pred = reward_model_forward(critic, prompt_tokens, critic_cache);

  // k1 KL estimator: E_{a~pi_old}[log pi_old(a) - log pi_ref(a)] is an
  // unbiased single-sample estimate of KL(pi_old || pi_ref) exactly
  // because the sample comes from pi_old (Schulman's "k1" estimator).
  float kl_term = old_logprob - ref_logprob;
  float shaped_reward = reward_score - kl_coef * kl_term;
  float advantage = shaped_reward - value_pred;

  return RolloutSample{prompt_tokens, action, old_logprob, ref_logprob, reward_score, shaped_reward, advantage};
}

} // namespace distributed_training
