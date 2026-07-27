// spec_decode_test.cpp — trains a real (tiny) draft + verifier transformer
// pair on CPU, then checks the property that actually matters about
// speculative decoding: its output is IDENTICAL to greedy-decoding the
// verifier alone, for any draft model (even an untrained one) — only the
// number of verifier forward passes needed changes with draft quality,
// never the output. That correctness invariant, not just "it runs and
// prints something," is what this test verifies.
#include "spec_decode.h"
#include "../../transformer/char_tokenizer.h"

#include <cstdio>
#include <random>
#include <string>

using namespace inference_serving;
using namespace transformer;

namespace {

int g_fails = 0;
void require(bool ok, const char *name) {
  std::printf("%s  %s\n", ok ? "PASS" : "FAIL", name);
  if (!ok) ++g_fails;
}

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

// Greedy-decodes `steps` tokens continuing from a 1-token seed, using
// `model` alone -- the ground truth speculative decoding must match
// exactly.
std::vector<int> greedy_generate(const ModelParams &model, int seed_token, int steps) {
  std::vector<int> generated{seed_token};
  for (int i = 0; i < steps; ++i) {
    ModelCache cache;
    Matrix logits = model_forward(model, generated, cache);
    int last = static_cast<int>(generated.size()) - 1;
    int argmax = 0;
    float best = logits(last, 0);
    for (int v = 1; v < model.config.vocab_size; ++v)
      if (logits(last, v) > best) { best = logits(last, v); argmax = v; }
    generated.push_back(argmax);
  }
  return generated;
}

// Runs speculative decoding until at least `min_len` tokens are generated,
// then truncates to exactly min_len for a fair comparison against
// greedy_generate's fixed-length output.
std::vector<int> spec_generate(SpecDecoder &dec, int seed_token, int min_len) {
  std::vector<int> context{seed_token};
  while (static_cast<int>(context.size()) < min_len) dec.generate_round(context);
  context.resize(static_cast<std::size_t>(min_len));
  return context;
}

void run_case(const char *label, const TransformerConfig &draft_cfg, int draft_epochs,
              const ModelParams &verifier, const std::vector<int> &corpus_tokens, std::mt19937 &rng) {
  ModelParams draft = train(draft_cfg, corpus_tokens, draft_epochs, 0.05f, rng);

  SpecDecoder dec(draft, verifier, /*draft_len=*/4);
  int target_len = static_cast<int>(corpus_tokens.size());
  std::vector<int> spec_result = spec_generate(dec, corpus_tokens[0], target_len);
  std::vector<int> verifier_only = greedy_generate(verifier, corpus_tokens[0], target_len - 1);

  bool matches = spec_result == verifier_only;
  require(matches, (std::string(label) + ": spec-decoded output exactly matches verifier-alone greedy output").c_str());

  long long naive_calls = target_len - 1;  // 1 verifier call per token, standard greedy
  double speedup = static_cast<double>(naive_calls) / static_cast<double>(dec.num_verifier_calls());
  std::printf("  [%s] acceptance rate=%.3f  verifier calls: spec=%lld naive=%lld  speedup=%.2fx\n",
              label, dec.acceptance_rate(), dec.num_verifier_calls(), naive_calls, speedup);
}

} // namespace

int main() {
  std::string corpus = "the quick brown fox jumps over the lazy dog ";
  CharTokenizer tok(corpus);
  std::vector<int> corpus_tokens = tok.encode(corpus);

  TransformerConfig verifier_cfg{tok.vocab_size(), /*d_model=*/24, /*num_heads=*/4, /*num_layers=*/3,
                                  /*d_ff=*/48, /*max_seq_len=*/64};
  TransformerConfig draft_cfg{tok.vocab_size(), /*d_model=*/8, /*num_heads=*/2, /*num_layers=*/1,
                               /*d_ff=*/16, /*max_seq_len=*/64};

  std::mt19937 verifier_rng(3);
  ModelParams verifier = train(verifier_cfg, corpus_tokens, /*epochs=*/600, 0.05f, verifier_rng);

  // Sanity precondition: the verifier alone must actually reproduce the
  // corpus via greedy decoding, or the correctness check below (spec vs.
  // verifier-alone) would be comparing against a meaningless target.
  std::vector<int> verifier_greedy = greedy_generate(verifier, corpus_tokens[0],
                                                       static_cast<int>(corpus_tokens.size()) - 1);
  require(verifier_greedy == corpus_tokens, "precondition: well-trained verifier greedy-reproduces the corpus");

  std::mt19937 draft_rng(11);
  // Case 1: untrained draft (0 epochs, random weights) -- correctness
  // must still hold; acceptance rate should be low (near chance).
  run_case("untrained draft", draft_cfg, /*draft_epochs=*/0, verifier, corpus_tokens, draft_rng);

  // Case 2: same draft architecture, meaningfully trained (but with far
  // less capacity than the verifier: 1 layer/d_model=8 vs 3 layers/
  // d_model=24) -- correctness must still hold; acceptance rate should
  // be markedly higher than case 1.
  std::mt19937 draft_rng2(11);
  run_case("trained draft", draft_cfg, /*draft_epochs=*/300, verifier, corpus_tokens, draft_rng2);

  std::printf("%s\n", g_fails == 0 ? "PASS" : "FAIL");
  return g_fails == 0 ? 0 : 1;
}
