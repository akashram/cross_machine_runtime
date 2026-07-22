// ppo_rlhf_test.cpp — PPO-based RLHF over the same single-digit addition
// domain as sft/ and reward_model/. Setup (both single-threaded, no
// networking needed — the NEW thing this step exercises is PPO's own
// data-parallel update loop, not re-proving gradient sync already proven
// by sft_test.cpp/reward_model_test.cpp):
//   1. SFT-initialize the policy (masked next-token loss, reusing
//      sft/sft_trainer.h unchanged) on (prompt, correct-sum) pairs.
//   2. Deep-copy the SFT policy into a frozen reference model.
//   3. Train a reward model (Bradley-Terry, reusing reward_model/
//      unchanged) on (prompt, chosen, rejected) triples, then freeze it.
// Then runs PPO: each outer iteration rolls out one sampled action per
// prompt from the CURRENT policy, scores it with the frozen reward model,
// KL-penalizes against the frozen reference, baselines with a trained
// critic, then runs several clipped-surrogate PPO update epochs
// data-parallel across 5 simulated ranks (ring_allreduce), same pattern
// as sft_test.cpp/reward_model_test.cpp. Validates mean reward rises
// substantially across iterations while mean KL divergence against the
// reference stays bounded (the "monitor for reward hacking" check PLAN.md
// step 24 asks for — unbounded KL growth would mean the policy drifted
// arbitrarily far from the reference in exchange for reward, the classic
// reward-hacking signature).
#include "ppo_rlhf.h"

#include "../sft/sft_trainer.h"
#include "char_tokenizer.h"
#include "ring_allreduce.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <future>
#include <random>
#include <vector>

using namespace distributed_training;
using namespace transformer;

namespace {

std::vector<std::string> make_prompts() {
  std::vector<std::string> prompts;
  for (int a = 0; a <= 4; ++a)
    for (int b = 0; b <= 4; ++b) prompts.push_back(std::to_string(a) + "+" + std::to_string(b) + "=");
  return prompts;
}

std::vector<SftExample> make_sft_dataset() {
  std::vector<SftExample> examples;
  for (int a = 0; a <= 4; ++a)
    for (int b = 0; b <= 4; ++b) examples.push_back(SftExample{std::to_string(a) + "+" + std::to_string(b) + "=", std::to_string(a + b)});
  return examples;
}

std::vector<PreferencePair> make_preference_dataset() {
  std::vector<PreferencePair> pairs;
  for (int a = 0; a <= 4; ++a) {
    for (int b = 0; b <= 4; ++b) {
      int sum = a + b;
      std::string prompt = std::to_string(a) + "+" + std::to_string(b) + "=";
      for (int k = 1; k <= 3; ++k) {
        int wrong = (sum + k) % 10;
        pairs.push_back(PreferencePair{prompt, std::to_string(sum), std::to_string(wrong)});
      }
    }
  }
  return pairs;
}

// Single-threaded local SGD step: scales a flattened gradient by 1/N and
// applies it directly — no ring_allreduce needed since this is a single
// (non-distributed) model, unlike the PPO update phase below.
void sft_init_policy(ModelParams &policy, const CharTokenizer &tok, const TransformerConfig &cfg, int epochs, float lr) {
  auto examples = make_sft_dataset();
  for (int epoch = 0; epoch < epochs; ++epoch) {
    ModelGrads acc = zero_model_grad(cfg);
    for (auto &ex : examples) {
      auto tokens = tok.encode(ex.prompt + ex.response);
      int prompt_len = static_cast<int>(tok.encode(ex.prompt).size());
      ModelCache cache;
      Matrix logits = model_forward(policy, tokens, cache);
      auto loss_result = masked_next_token_loss(logits, tokens, prompt_len);
      ModelGrads g = zero_model_grad(cfg);
      model_backward(policy, cache, loss_result.dlogits, g);
      accumulate_grad(acc, g);
    }
    auto flat = flatten_grad(acc);
    for (float &v : flat) v /= static_cast<float>(examples.size());
    unflatten_into_grad(acc, flat);
    sgd_step(policy, acc, lr);
  }
}

void train_reward_model(RewardModelParams &reward_model, const CharTokenizer &tok, const TransformerConfig &cfg, int epochs, float lr) {
  auto pairs = make_preference_dataset();
  for (int epoch = 0; epoch < epochs; ++epoch) {
    RewardModelGrads acc = zero_reward_model_grad(cfg);
    for (auto &pair : pairs) {
      auto tokens_chosen = tok.encode(pair.prompt + pair.response_chosen);
      auto tokens_rejected = tok.encode(pair.prompt + pair.response_rejected);
      RewardModelCache cache_chosen, cache_rejected;
      float r_chosen = reward_model_forward(reward_model, tokens_chosen, cache_chosen);
      float r_rejected = reward_model_forward(reward_model, tokens_rejected, cache_rejected);
      auto bt = bradley_terry_loss(r_chosen, r_rejected);

      RewardModelGrads grad_chosen = zero_reward_model_grad(cfg);
      reward_model_backward(reward_model, cache_chosen, bt.dreward_chosen, grad_chosen);
      accumulate_reward_grad(acc, grad_chosen);

      RewardModelGrads grad_rejected = zero_reward_model_grad(cfg);
      reward_model_backward(reward_model, cache_rejected, bt.dreward_rejected, grad_rejected);
      accumulate_reward_grad(acc, grad_rejected);
    }
    auto flat = flatten_reward_grad(acc);
    for (float &v : flat) v /= static_cast<float>(pairs.size());
    unflatten_into_reward_grad(acc, flat);
    sgd_step_reward(reward_model, acc, lr);
  }
}

} // namespace

int main() {
  std::string corpus = "0123456789+=";
  CharTokenizer tok(corpus);
  TransformerConfig cfg{tok.vocab_size(), /*d_model=*/16, /*num_heads=*/2, /*num_layers=*/2, /*d_ff=*/32,
                        /*max_seq_len=*/8};

  // --- Setup: SFT-init the policy, freeze a reference copy, train + freeze a reward model. ---
  std::mt19937 policy_rng(24);
  ModelParams policy = init_model(cfg, policy_rng);
  sft_init_policy(policy, tok, cfg, /*epochs=*/150, /*lr=*/0.05f);
  ModelParams ref_model = policy; // frozen from here on

  std::mt19937 reward_rng(124);
  RewardModelParams reward_model = init_reward_model(cfg, reward_rng);
  train_reward_model(reward_model, tok, cfg, /*epochs=*/150, /*lr=*/0.05f);
  // reward_model is frozen from here on

  std::mt19937 critic_rng(224);
  CriticParams critic = init_critic(cfg, critic_rng); // trained during PPO below

  auto prompt_strs = make_prompts();
  std::vector<std::vector<int>> prompt_tokens;
  for (auto &p : prompt_strs) prompt_tokens.push_back(tok.encode(p));

  // --- PPO ---
  constexpr int kWorldSize = 5;
  constexpr int kOuterIters = 10;
  constexpr int kPpoEpochs = 4;
  constexpr float kClipEps = 0.2f;
  constexpr float kKlCoef = 0.1f;
  constexpr float kLrPolicy = 0.02f;
  constexpr float kLrCritic = 0.05f;
  constexpr uint16_t kBasePort = 36501;

  std::mt19937 rollout_rng(101);
  float first_mean_reward = 0.0f, last_mean_reward = 0.0f;
  float max_abs_kl_seen = 0.0f;

  int per_rank = static_cast<int>(prompt_tokens.size()) / kWorldSize;

  for (int iter = 0; iter < kOuterIters; ++iter) {
    std::vector<RolloutSample> batch;
    batch.reserve(prompt_tokens.size());
    for (auto &pt : prompt_tokens) batch.push_back(rollout_one(policy, ref_model, reward_model, critic, pt, kKlCoef, rollout_rng));

    float mean_reward = 0.0f, mean_kl = 0.0f;
    for (auto &s : batch) { mean_reward += s.reward_score; mean_kl += (s.old_logprob - s.ref_logprob); }
    mean_reward /= static_cast<float>(batch.size());
    mean_kl /= static_cast<float>(batch.size());
    std::printf("iter %2d: mean_reward=%.3f mean_kl=%.3f\n", iter, mean_reward, mean_kl);
    if (iter == 0) first_mean_reward = mean_reward;
    last_mean_reward = mean_reward;
    max_abs_kl_seen = std::max(max_abs_kl_seen, std::fabs(mean_kl));

    auto channels = netcommon::make_tcp_loopback_mesh(kWorldSize, static_cast<uint16_t>(kBasePort + iter * kWorldSize));
    std::vector<std::future<void>> results;
    std::vector<ModelParams> final_policies(kWorldSize, policy);
    std::vector<CriticParams> final_critics(kWorldSize, critic);

    for (int r = 0; r < kWorldSize; ++r) {
      netcommon::Channel *ch = channels[static_cast<size_t>(r)].get();
      results.push_back(std::async(std::launch::async, [&, r, ch]() {
        ModelParams local_policy = policy;
        CriticParams local_critic = critic;
        std::vector<RolloutSample> local_batch(batch.begin() + r * per_rank, batch.begin() + (r + 1) * per_rank);

        for (int epoch = 0; epoch < kPpoEpochs; ++epoch) {
          ModelGrads policy_grad_acc = zero_model_grad(cfg);
          CriticGrads critic_grad_acc = zero_critic_grad(cfg);

          for (auto &sample : local_batch) {
            ModelCache pcache;
            Matrix logits = model_forward(local_policy, sample.prompt_tokens, pcache);
            int pos = static_cast<int>(sample.prompt_tokens.size()) - 1;
            float new_logprob = log_probs(logits)(pos, sample.action);
            auto pl = ppo_policy_loss(new_logprob, sample.old_logprob, sample.advantage, kClipEps);
            Matrix dlogits = policy_dlogits(logits, pos, sample.action, pl.d_new_logprob);
            ModelGrads pg = zero_model_grad(cfg);
            model_backward(local_policy, pcache, dlogits, pg);
            accumulate_grad(policy_grad_acc, pg);

            RewardModelCache ccache;
            float value_pred = reward_model_forward(local_critic, sample.prompt_tokens, ccache);
            float d_value = value_pred - sample.shaped_reward;
            CriticGrads cg = zero_critic_grad(cfg);
            reward_model_backward(local_critic, ccache, d_value, cg);
            accumulate_reward_grad(critic_grad_acc, cg);
          }

          // Both grads share ONE ring_allreduce call (concat, split by
          // known length) — same trick transformer_model.h's idx-threading
          // unflatten_into_grad overload was added for in sft/reward_model.
          auto policy_flat = flatten_grad(policy_grad_acc);
          auto critic_flat = flatten_reward_grad(critic_grad_acc);
          size_t policy_len = policy_flat.size();
          std::vector<float> combined = policy_flat;
          combined.insert(combined.end(), critic_flat.begin(), critic_flat.end());
          ring_allreduce(combined.data(), combined.size(), *ch);
          for (float &v : combined) v /= static_cast<float>(batch.size());
          std::vector<float> policy_part(combined.begin(), combined.begin() + static_cast<long>(policy_len));
          std::vector<float> critic_part(combined.begin() + static_cast<long>(policy_len), combined.end());
          unflatten_into_grad(policy_grad_acc, policy_part);
          unflatten_into_reward_grad(critic_grad_acc, critic_part);
          sgd_step(local_policy, policy_grad_acc, kLrPolicy);
          sgd_step_reward(local_critic, critic_grad_acc, kLrCritic);
        }
        final_policies[static_cast<size_t>(r)] = local_policy;
        final_critics[static_cast<size_t>(r)] = local_critic;
      }));
    }
    for (auto &f : results) f.get();
    policy = final_policies[0];
    critic = final_critics[0];
  }

  bool reward_improved = last_mean_reward > first_mean_reward + 0.3f;
  bool kl_bounded = max_abs_kl_seen < 3.0f; // no runaway divergence from the reference => not reward hacking
  bool ok = reward_improved && kl_bounded;
  std::printf("reward rose (%.3f -> %.3f) with KL bounded (max |mean KL| = %.3f): %s\n", first_mean_reward, last_mean_reward,
              max_abs_kl_seen, ok ? "PASS" : "FAIL");

  std::printf("%s\n", ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
}
