// reward_model_test.cpp — a tiny real preference-ranking task built on
// the same single-digit addition domain as sft_test.cpp: for prompt
// "a+b=", the CHOSEN response is always the correct sum; each prompt gets
// 3 REJECTED candidates ((a+b+1)/(a+b+2)/(a+b+3) mod 10, all guaranteed
// != the correct sum since a+b <= 8). That gives every prompt multiple
// preference pairs, so a pair-level train/held-out split (not a
// prompt-level split) leaves every prompt's correct sum visible during
// training while still holding out specific (chosen, rejected) pairings
// — held-out generalization over DISTRACTORS, not over never-seen
// arithmetic, which is what a reward model's held-out eval is normally
// checking (a fixed prompt/response distribution, not novel prompts).
// Trains with the Bradley-Terry pairwise objective, 5 simulated
// data-parallel ranks, gradient all-reduce each step. Validates ranking
// accuracy on the held-out preference pairs improves substantially over
// the untrained base model.
#include "reward_model.h"

#include "char_tokenizer.h"
#include "ring_allreduce.h"

#include <algorithm>
#include <cstdio>
#include <future>
#include <random>
#include <vector>

using namespace distributed_training;
using namespace transformer;

namespace {

std::vector<PreferencePair> make_dataset() {
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

float reward_of(const RewardModelParams &model, const CharTokenizer &tok, const std::string &prompt, const std::string &response) {
  auto tokens = tok.encode(prompt + response);
  RewardModelCache cache;
  return reward_model_forward(model, tokens, cache);
}

float evaluate_ranking_accuracy(const RewardModelParams &model, const CharTokenizer &tok, const std::vector<PreferencePair> &pairs) {
  int correct = 0;
  for (auto &pair : pairs) {
    float r_chosen = reward_of(model, tok, pair.prompt, pair.response_chosen);
    float r_rejected = reward_of(model, tok, pair.prompt, pair.response_rejected);
    if (r_chosen > r_rejected) ++correct;
  }
  return static_cast<float>(correct) / static_cast<float>(pairs.size());
}

} // namespace

int main() {
  constexpr int kWorldSize = 5;
  constexpr int kEpochs = 150;
  constexpr float kLr = 0.05f;
  constexpr uint16_t kBasePort = 36401;

  auto full_dataset = make_dataset(); // 25 prompts x 3 distractors = 75 pairs
  // Shuffle at the PAIR level (not prompt level) before splitting: this
  // scatters each prompt's 3 pairs across train/held-out so every prompt's
  // correct sum is seen during training (see file header comment) while
  // specific (chosen, rejected) pairings are still genuinely held out.
  std::mt19937 split_rng(7);
  std::shuffle(full_dataset.begin(), full_dataset.end(), split_rng);
  std::vector<PreferencePair> train(full_dataset.begin(), full_dataset.begin() + 60);
  std::vector<PreferencePair> held_out(full_dataset.begin() + 60, full_dataset.end());
  int per_rank = static_cast<int>(train.size()) / kWorldSize; // 12

  std::string corpus = "0123456789+=";
  CharTokenizer tok(corpus);
  TransformerConfig cfg{tok.vocab_size(), /*d_model=*/16, /*num_heads=*/2, /*num_layers=*/2, /*d_ff=*/32,
                        /*max_seq_len=*/8};

  std::mt19937 init_rng(23);
  RewardModelParams init_model_params = init_reward_model(cfg, init_rng);

  float base_accuracy = evaluate_ranking_accuracy(init_model_params, tok, held_out);
  std::printf("base model (untrained): held-out ranking accuracy = %.3f\n", base_accuracy);

  auto channels = netcommon::make_tcp_loopback_mesh(kWorldSize, kBasePort);
  std::vector<std::future<void>> results;
  std::vector<RewardModelParams> final_models(kWorldSize, init_model_params);

  for (int r = 0; r < kWorldSize; ++r) {
    netcommon::Channel *ch = channels[static_cast<size_t>(r)].get();
    results.push_back(std::async(std::launch::async, [&, r, ch]() {
      RewardModelParams model = init_model_params; // deep copy, independent memory per rank

      std::vector<PreferencePair> local_pairs(train.begin() + r * per_rank, train.begin() + (r + 1) * per_rank);

      for (int epoch = 0; epoch < kEpochs; ++epoch) {
        RewardModelGrads accumulated = zero_reward_model_grad(cfg);
        for (auto &pair : local_pairs) {
          auto tokens_chosen = tok.encode(pair.prompt + pair.response_chosen);
          auto tokens_rejected = tok.encode(pair.prompt + pair.response_rejected);
          RewardModelCache cache_chosen, cache_rejected;
          float r_chosen = reward_model_forward(model, tokens_chosen, cache_chosen);
          float r_rejected = reward_model_forward(model, tokens_rejected, cache_rejected);
          auto bt = bradley_terry_loss(r_chosen, r_rejected);

          // reward_model_backward (like transformer::model_backward) writes
          // block-level gradients by ASSIGNMENT, not +=, so each call needs
          // its own freshly-zeroed grad struct; accumulate_reward_grad does
          // the actual summing across the chosen/rejected pair and across
          // examples.
          RewardModelGrads grad_chosen = zero_reward_model_grad(cfg);
          reward_model_backward(model, cache_chosen, bt.dreward_chosen, grad_chosen);
          accumulate_reward_grad(accumulated, grad_chosen);

          RewardModelGrads grad_rejected = zero_reward_model_grad(cfg);
          reward_model_backward(model, cache_rejected, bt.dreward_rejected, grad_rejected);
          accumulate_reward_grad(accumulated, grad_rejected);
        }
        auto flat_grad = flatten_reward_grad(accumulated);
        ring_allreduce(flat_grad.data(), flat_grad.size(), *ch); // sum across ranks
        for (float &g : flat_grad) g /= static_cast<float>(train.size()); // -> mean over all 20 train pairs
        unflatten_into_reward_grad(accumulated, flat_grad);
        sgd_step_reward(model, accumulated, kLr);
      }
      final_models[static_cast<size_t>(r)] = model;
    }));
  }
  for (auto &f : results) f.get();

  float trained_accuracy = evaluate_ranking_accuracy(final_models[0], tok, held_out);
  std::printf("after reward model training (%d epochs, %d train pairs, %d held-out, %d ranks): held-out ranking accuracy = %.3f\n",
              kEpochs, static_cast<int>(train.size()), static_cast<int>(held_out.size()), kWorldSize, trained_accuracy);

  bool ok = trained_accuracy >= 0.8f && trained_accuracy > base_accuracy;
  std::printf("reward signal meaningful (held-out ranking accuracy >= 80%%): %s\n", ok ? "PASS" : "FAIL");

  std::printf("%s\n", ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
}
