// sft_test.cpp — a tiny real instruction-tuning task (single-digit
// addition: "2+3=" -> "5") over the real transformer/ model. 5 simulated
// data-parallel ranks (5 examples each, 25 total), gradient all-reduce
// each step (mean over the full 25-example batch), loss masked to the
// response digit only. Validates perplexity improves substantially from
// the untrained base model.
#include "sft_trainer.h"

#include "char_tokenizer.h"
#include "ring_allreduce.h"

#include <cmath>
#include <cstdio>
#include <future>
#include <random>
#include <vector>

using namespace distributed_training;
using namespace transformer;

namespace {

std::vector<SftExample> make_dataset() {
  std::vector<SftExample> examples;
  for (int a = 0; a <= 4; ++a) {
    for (int b = 0; b <= 4; ++b) {
      examples.push_back(SftExample{std::to_string(a) + "+" + std::to_string(b) + "=", std::to_string(a + b)});
    }
  }
  return examples;
}

float evaluate_mean_loss(const ModelParams &model, const CharTokenizer &tok, const std::vector<SftExample> &examples) {
  float total = 0.0f;
  for (auto &ex : examples) {
    auto tokens = tok.encode(ex.prompt + ex.response);
    int prompt_len = static_cast<int>(tok.encode(ex.prompt).size());
    ModelCache cache;
    Matrix logits = model_forward(model, tokens, cache);
    total += masked_next_token_loss(logits, tokens, prompt_len).loss;
  }
  return total / static_cast<float>(examples.size());
}

} // namespace

int main() {
  constexpr int kWorldSize = 5;
  constexpr int kEpochs = 150;
  constexpr float kLr = 0.05f;
  constexpr uint16_t kBasePort = 36301;

  auto dataset = make_dataset(); // 25 examples
  int per_rank = static_cast<int>(dataset.size()) / kWorldSize; // 5

  std::string corpus = "0123456789+=";
  CharTokenizer tok(corpus);
  TransformerConfig cfg{tok.vocab_size(), /*d_model=*/16, /*num_heads=*/2, /*num_layers=*/2, /*d_ff=*/32,
                        /*max_seq_len=*/8};

  std::mt19937 init_rng(21);
  ModelParams init_model_params = init_model(cfg, init_rng);

  float base_perplexity = perplexity(evaluate_mean_loss(init_model_params, tok, dataset));
  std::printf("base model (untrained): perplexity = %.3f\n", base_perplexity);

  auto channels = netcommon::make_tcp_loopback_mesh(kWorldSize, kBasePort);
  std::vector<std::future<void>> results;
  std::vector<ModelParams> final_models(kWorldSize, init_model_params);

  for (int r = 0; r < kWorldSize; ++r) {
    netcommon::Channel *ch = channels[static_cast<size_t>(r)].get();
    results.push_back(std::async(std::launch::async, [&, r, ch]() {
      ModelParams model = init_model_params; // deep copy, independent memory per rank

      std::vector<SftExample> local_examples(dataset.begin() + r * per_rank, dataset.begin() + (r + 1) * per_rank);

      for (int epoch = 0; epoch < kEpochs; ++epoch) {
        ModelGrads accumulated = zero_model_grad(cfg);
        for (auto &ex : local_examples) {
          auto tokens = tok.encode(ex.prompt + ex.response);
          int prompt_len = static_cast<int>(tok.encode(ex.prompt).size());
          ModelCache cache;
          Matrix logits = model_forward(model, tokens, cache);
          auto loss_result = masked_next_token_loss(logits, tokens, prompt_len);
          ModelGrads grad = zero_model_grad(cfg);
          model_backward(model, cache, loss_result.dlogits, grad);
          accumulate_grad(accumulated, grad);
        }
        auto flat_grad = flatten_grad(accumulated);
        ring_allreduce(flat_grad.data(), flat_grad.size(), *ch); // sum across ranks
        for (float &g : flat_grad) g /= static_cast<float>(dataset.size()); // -> mean over all 25 examples
        unflatten_into_grad(accumulated, flat_grad);
        sgd_step(model, accumulated, kLr);
      }
      final_models[static_cast<size_t>(r)] = model;
    }));
  }
  for (auto &f : results) f.get();

  float sft_perplexity = perplexity(evaluate_mean_loss(final_models[0], tok, dataset));
  std::printf("after SFT (%d epochs, %d examples, %d ranks): perplexity = %.3f\n", kEpochs, static_cast<int>(dataset.size()),
              kWorldSize, sft_perplexity);

  bool ok = sft_perplexity < base_perplexity * 0.5f;
  std::printf("perplexity improved substantially (< 50%% of base): %s\n", ok ? "PASS" : "FAIL");

  std::printf("%s\n", ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
}
