#pragma once

// Supervised fine-tuning: PLAN.md step 22. Loss masking on prompt tokens
// (only the RESPONSE portion of each example contributes to the loss —
// the standard SFT technique: the model should learn to produce good
// responses, not memorize prompts) over the real transformer/ model, data
// sharded across simulated data-parallel ranks with gradient all-reduce
// (same pattern as distributed_training/data_parallel, applied to a real
// model instead of linear regression).

#include <cstddef>
#include <string>
#include <vector>

#include "../../transformer/transformer_model.h"

namespace distributed_training {

struct SftExample {
  std::string prompt;   // e.g. "2+3="
  std::string response; // e.g. "5" — only this portion is trained on
};

// Cross-entropy loss restricted to positions predicting a RESPONSE token
// (i.e. positions [prompt_len-1, prompt_len+response_len-2] of the
// concatenated "prompt+response" sequence, predicting tokens
// [prompt_len, prompt_len+response_len-1]) — prompt-predicting positions
// get zero gradient, same shape as next_token_loss but with a mask.
inline transformer::LMLossResult masked_next_token_loss(const transformer::Matrix &logits,
                                                          const std::vector<int> &token_ids, int prompt_len) {
  int seq = logits.rows(), vocab = logits.cols();
  int response_len = static_cast<int>(token_ids.size()) - prompt_len;
  transformer::Matrix dlogits(seq, vocab);
  float total = 0.0f;
  for (int i = prompt_len - 1; i < prompt_len - 1 + response_len; ++i) {
    float max_v = logits(i, 0);
    for (int v = 1; v < vocab; ++v) max_v = std::max(max_v, logits(i, v));
    float denom = 0.0f;
    for (int v = 0; v < vocab; ++v) denom += std::exp(logits(i, v) - max_v);
    int target = token_ids[static_cast<size_t>(i + 1)];
    float target_prob = std::exp(logits(i, target) - max_v) / denom;
    total += -std::log(std::max(target_prob, 1e-9f));
    for (int v = 0; v < vocab; ++v) {
      float prob = std::exp(logits(i, v) - max_v) / denom;
      dlogits(i, v) = (prob - (v == target ? 1.0f : 0.0f)) / static_cast<float>(response_len);
    }
  }
  return transformer::LMLossResult{total / static_cast<float>(response_len), dlogits};
}

// Perplexity = exp(mean per-token cross-entropy loss) — the standard LM
// evaluation metric PLAN.md asks this step to improve.
inline float perplexity(float mean_loss) { return std::exp(mean_loss); }

} // namespace distributed_training
