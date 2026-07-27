// serving_latency_bench.cpp — real wall-clock latency/throughput
// measurement of this repo's own CPU serving path (serving_backend's
// router, dispatching to transformer/'s real model), instrumented
// per-token so time-to-first-token (TTFT) and time-per-output-token
// (TPOT) are measured separately, matching PLAN.md Phase 9 step 9's
// metric definitions exactly.
//
// Unlike continuous_batching_bench.cpp / sla_scheduler_bench.cpp (pure
// scheduling simulations with no real compute, where wall-clock timing
// would be meaningless), this file times REAL compute — an actual
// transformer forward pass per token — so std::chrono here measures a
// real, if CPU-not-GPU, quantity.
//
// The vLLM/TensorRT-LLM comparison PLAN.md asks for stays hardware-gated
// (see vllm_tensorrt_bench.py in this directory) -- no GPU, no vLLM/
// TensorRT-LLM install, and no hosted LLM checkpoint on this Mac. This
// file's numbers characterize this repo's own unbatched, uncached CPU
// path -- a real number to eventually set the GPU comparison against,
// not a substitute for it.
#include "../serving_backend/serving_router.h"
#include "../../transformer/char_tokenizer.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <random>
#include <vector>

using namespace inference_serving;
using namespace transformer;

namespace {

ModelParams train(const TransformerConfig &cfg, const std::vector<int> &tokens, int epochs, float lr,
                   std::mt19937 &init_rng) {
  ModelParams model = init_model(cfg, init_rng);
  for (int epoch = 0; epoch < epochs; ++epoch) {
    ModelCache cache;
    Matrix logits = model_forward(model, tokens, cache);
    auto lm_loss = next_token_loss(logits, tokens);
    ModelGrads grad = zero_model_grad(cfg);
    model_backward(model, cache, lm_loss.dlogits, grad);
    sgd_step(model, grad, lr);
  }
  return model;
}

struct RequestTiming {
  double ttft_ms;                    // time to first generated token
  double tpot_ms;                    // mean time per subsequent token
  std::vector<double> per_token_ms;  // full per-token breakdown
};

RequestTiming time_one_request(const ModelParams &model, int seed_token, int max_new_tokens) {
  RequestTiming t;
  t.per_token_ms.reserve(static_cast<std::size_t>(max_new_tokens));
  std::vector<int> generated{seed_token};

  for (int i = 0; i < max_new_tokens; ++i) {
    auto t0 = std::chrono::steady_clock::now();
    ModelCache cache;
    Matrix logits = model_forward(model, generated, cache);
    int last = static_cast<int>(generated.size()) - 1;
    int argmax = 0;
    float best = logits(last, 0);
    for (int v = 1; v < model.config.vocab_size; ++v)
      if (logits(last, v) > best) { best = logits(last, v); argmax = v; }
    generated.push_back(argmax);
    auto t1 = std::chrono::steady_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    t.per_token_ms.push_back(ms);
  }

  t.ttft_ms = t.per_token_ms.front();
  double sum_rest = 0.0;
  for (std::size_t i = 1; i < t.per_token_ms.size(); ++i) sum_rest += t.per_token_ms[i];
  t.tpot_ms = (t.per_token_ms.size() > 1) ? sum_rest / static_cast<double>(t.per_token_ms.size() - 1) : 0.0;
  return t;
}

double percentile(std::vector<double> v, double pct) {
  std::sort(v.begin(), v.end());
  std::size_t idx = static_cast<std::size_t>(pct / 100.0 * static_cast<double>(v.size() - 1));
  return v[idx];
}

} // namespace

int main() {
  std::string corpus = "the quick brown fox jumps over the lazy dog ";
  CharTokenizer tok(corpus);
  std::vector<int> tokens = tok.encode(corpus);

  TransformerConfig cfg{tok.vocab_size(), /*d_model=*/32, /*num_heads=*/4, /*num_layers=*/2,
                         /*d_ff=*/64, /*max_seq_len=*/64};
  std::mt19937 rng(2);
  ModelParams model = train(cfg, tokens, /*epochs=*/200, 0.05f, rng);

  constexpr int kNumRequests = 60;
  constexpr int kMaxNewTokens = 16;

  std::vector<double> ttfts, tpots;
  ttfts.reserve(kNumRequests);
  tpots.reserve(kNumRequests);

  auto wall_start = std::chrono::steady_clock::now();
  for (int req = 0; req < kNumRequests; ++req) {
    int seed = tokens[static_cast<std::size_t>(req % static_cast<int>(tokens.size()))];
    RequestTiming t = time_one_request(model, seed, kMaxNewTokens);
    ttfts.push_back(t.ttft_ms);
    tpots.push_back(t.tpot_ms);
  }
  auto wall_end = std::chrono::steady_clock::now();
  double wall_s = std::chrono::duration<double>(wall_end - wall_start).count();

  double throughput_req_s = kNumRequests / wall_s;
  double throughput_tok_s = (kNumRequests * static_cast<double>(kMaxNewTokens)) / wall_s;

  std::printf("requests=%d  max_new_tokens=%d  wall=%.3fs\n", kNumRequests, kMaxNewTokens, wall_s);
  std::printf("throughput: %.1f req/s, %.1f tok/s\n", throughput_req_s, throughput_tok_s);
  std::printf("%-12s %10s %10s %10s\n", "metric", "p50(ms)", "p99(ms)", "p999(ms)");
  std::printf("%-12s %10.4f %10.4f %10.4f\n", "TTFT", percentile(ttfts, 50), percentile(ttfts, 99), percentile(ttfts, 99.9));
  std::printf("%-12s %10.4f %10.4f %10.4f\n", "TPOT", percentile(tpots, 50), percentile(tpots, 99), percentile(tpots, 99.9));

  std::printf(
      "\nThis is this repo's own unbatched, uncached CPU path (single "
      "request at a time, full-sequence recompute per token -- no "
      "continuous_batching, no paged_kv reuse in this measurement). See "
      "vllm_tensorrt_bench.py for the hardware-gated real-GPU comparison "
      "this number should eventually be checked against.\n");
  return 0;
}
