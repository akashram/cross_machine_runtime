#pragma once

// DPO (Direct Preference Optimization): PLAN.md step 25. Optimizes the
// policy DIRECTLY on preference pairs — no reward model, no critic, no
// on-policy rollouts at train time. Rafailov et al. (2023)'s
// reparameterization: the same KL-constrained reward-maximization problem
// PPO solves has a closed form optimal policy, which lets the
// Bradley-Terry preference loss be rewritten entirely in terms of the
// POLICY's own log-probabilities against a frozen reference — an
// "implicit reward" r(x,y) = beta * (log pi_policy(y|x) - log pi_ref(y|x))
// substituted directly into the same pairwise loss reward_model/ trains
// a separate network to predict. That means DPO's loss is EXACTLY
// reward_model.h's bradley_terry_loss with this implicit reward in place
// of a learned reward model's output — reused unchanged below, not
// re-derived — and its gradient is EXACTLY ppo_rlhf.h's policy_dlogits
// applied to the chosen/rejected response's log-prob position.

#include "../ppo_rlhf/ppo_rlhf.h"         // reuses policy_dlogits
#include "../reward_model/reward_model.h" // reuses PreferencePair, bradley_terry_loss

namespace distributed_training {

struct DpoGrad {
  float d_new_logprob_chosen;   // d(loss) / d(log pi_policy(chosen | prompt))
  float d_new_logprob_rejected; // d(loss) / d(log pi_policy(rejected | prompt))
};

struct DpoLossResult {
  float loss;
  float margin; // implicit reward margin: r(chosen) - r(rejected) — DPO's analogue of a reward model's score gap
  DpoGrad grad;
};

// beta plays the same role as PPO's kl_coef: both derive from the same
// KL-constrained objective (Rafailov et al., Section 4), just enforced
// differently — PPO penalizes KL explicitly in the reward signal every
// rollout; DPO bakes the KL constraint into the closed-form optimal
// policy, so beta only ever appears as this scale factor on the
// log-probability ratio. The reference's log-probs are treated as
// constants (frozen model — no gradient flows into it), so only
// new_logprob_chosen/rejected need backprop.
inline DpoLossResult dpo_loss(float new_logprob_chosen, float ref_logprob_chosen, float new_logprob_rejected,
                               float ref_logprob_rejected, float beta) {
  float implicit_reward_chosen = beta * (new_logprob_chosen - ref_logprob_chosen);
  float implicit_reward_rejected = beta * (new_logprob_rejected - ref_logprob_rejected);
  auto bt = bradley_terry_loss(implicit_reward_chosen, implicit_reward_rejected);
  DpoGrad grad{bt.dreward_chosen * beta, bt.dreward_rejected * beta}; // chain rule through beta*(new_logprob - ref_logprob)
  return DpoLossResult{bt.loss, implicit_reward_chosen - implicit_reward_rejected, grad};
}

} // namespace distributed_training
