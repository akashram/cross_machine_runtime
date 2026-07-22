// transformer_test.cpp — two checks:
//  1. gradient checking: analytic model_backward() gradients vs. central
//     finite differences, sampled across every distinct parameter type
//     (embedding, attention weights, MLP weights, LayerNorm affine,
//     output projection) — not exhaustive (this model has thousands of
//     parameters), but covering every DISTINCT code path in
//     block_backward/model_backward at least once. Median relative error
//     per parameter, same rationale as autograd/autograd_test.cpp and
//     seq_parallel/seq_parallel_test.cpp: this model has both ReLU and
//     LayerNorm, both of which can produce a rare finite-difference
//     outlier at a kink without indicating a wrong formula.
//  2. an actual training run: train on a tiny repetitive synthetic corpus
//     until greedy generation reproduces it — proof the whole stack
//     (embedding, causal attention, residuals, LayerNorm, output
//     projection, and every backward path) composes into something that
//     actually learns to predict text, not just passes isolated checks.
#include "char_tokenizer.h"
#include "transformer_model.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <functional>
#include <random>
#include <vector>

using namespace transformer;

namespace {

float median_rel_err_for_matrix(const std::function<float(const std::vector<int> &)> &loss_fn,
                                 const std::vector<int> &tokens, Matrix &param, const Matrix &analytic_grad,
                                 int num_samples, std::mt19937 &rng, float epsilon = 1e-3f) {
  std::uniform_int_distribution<int> row_dist(0, param.rows() - 1);
  std::uniform_int_distribution<int> col_dist(0, param.cols() - 1);
  std::vector<float> rel_errs;
  for (int s = 0; s < num_samples; ++s) {
    int r = row_dist(rng), c = col_dist(rng);
    float orig = param(r, c);
    param(r, c) = orig + epsilon;
    float loss_plus = loss_fn(tokens);
    param(r, c) = orig - epsilon;
    float loss_minus = loss_fn(tokens);
    param(r, c) = orig;
    float numeric = (loss_plus - loss_minus) / (2.0f * epsilon);
    rel_errs.push_back(std::abs(analytic_grad(r, c) - numeric) / std::max(1e-4f, std::abs(numeric)));
  }
  std::sort(rel_errs.begin(), rel_errs.end());
  return rel_errs[rel_errs.size() / 2];
}

bool test_gradient_check() {
  std::string corpus = "abcdefghij ";
  CharTokenizer tok(corpus);
  TransformerConfig cfg{tok.vocab_size(), /*d_model=*/8, /*num_heads=*/2, /*num_layers=*/2, /*d_ff=*/16,
                        /*max_seq_len=*/8};
  std::mt19937 init_rng(5);
  ModelParams model = init_model(cfg, init_rng);

  std::vector<int> tokens = tok.encode("abcdef");

  auto loss_fn = [&](const std::vector<int> &t) -> float {
    ModelCache cache;
    Matrix logits = model_forward(model, t, cache);
    return next_token_loss(logits, t).loss;
  };

  ModelCache cache;
  Matrix logits = model_forward(model, tokens, cache);
  auto lm_loss = next_token_loss(logits, tokens);
  ModelGrads grad = zero_model_grad(cfg);
  model_backward(model, cache, lm_loss.dlogits, grad);

  std::mt19937 sample_rng(77);
  struct Check { const char *name; Matrix *param; const Matrix *grad; };
  std::vector<Check> checks{
      {"token_emb", &model.token_emb, &grad.token_emb},
      {"pos_emb", &model.pos_emb, &grad.pos_emb},
      {"block0.wq", &model.blocks[0].wq, &grad.blocks[0].wq},
      {"block0.w1", &model.blocks[0].w1, &grad.blocks[0].w1},
      {"block1.wo", &model.blocks[1].wo, &grad.blocks[1].wo},
      {"block1.gamma2", &model.blocks[1].gamma2, &grad.blocks[1].gamma2},
      {"final_gamma", &model.final_gamma, &grad.final_gamma},
      {"w_out", &model.w_out, &grad.w_out},
  };

  bool ok = true;
  for (auto &chk : checks) {
    float median = median_rel_err_for_matrix(loss_fn, tokens, *chk.param, *chk.grad, 8, sample_rng);
    std::printf("  %-14s median relative error (8 samples) = %.6f\n", chk.name, median);
    if (median > 2e-2f) ok = false;
  }
  std::printf("test 1 (gradient check): %s\n", ok ? "PASS" : "FAIL");
  return ok;
}

bool test_trains_and_generates() {
  std::string corpus = "the quick fox jumps ";
  CharTokenizer tok(corpus);
  TransformerConfig cfg{tok.vocab_size(), /*d_model=*/16, /*num_heads=*/2, /*num_layers=*/2, /*d_ff=*/32,
                        /*max_seq_len=*/32};
  std::mt19937 init_rng(9);
  ModelParams model = init_model(cfg, init_rng);

  std::vector<int> tokens = tok.encode(corpus);
  constexpr int kEpochs = 400;
  constexpr float kLr = 0.05f;

  float first_loss = 0.0f, last_loss = 0.0f;
  for (int epoch = 0; epoch < kEpochs; ++epoch) {
    ModelCache cache;
    Matrix logits = model_forward(model, tokens, cache);
    auto lm_loss = next_token_loss(logits, tokens);
    ModelGrads grad = zero_model_grad(cfg);
    model_backward(model, cache, lm_loss.dlogits, grad);
    sgd_step(model, grad, kLr);
    if (epoch == 0) first_loss = lm_loss.loss;
    last_loss = lm_loss.loss;
  }
  std::printf("training: loss %.4f -> %.4f\n", first_loss, last_loss);

  // Greedy generation from the first character, teacher-forced against
  // nothing (autoregressive: feed back its own argmax each step).
  std::vector<int> generated{tokens[0]};
  int steps = static_cast<int>(tokens.size()) - 1;
  for (int i = 0; i < steps; ++i) {
    ModelCache cache;
    Matrix logits = model_forward(model, generated, cache);
    int last_pos = static_cast<int>(generated.size()) - 1;
    int argmax = 0;
    float best = logits(last_pos, 0);
    for (int v = 1; v < cfg.vocab_size; ++v)
      if (logits(last_pos, v) > best) { best = logits(last_pos, v); argmax = v; }
    generated.push_back(argmax);
  }
  std::string generated_text = tok.decode(generated);
  bool matches = generated_text == corpus;
  std::printf("generated: \"%s\"\n", generated_text.c_str());
  std::printf("expected:  \"%s\"\n", corpus.c_str());

  bool ok = last_loss < first_loss * 0.05f && matches;
  std::printf("test 2 (trains and greedy-generates the training corpus exactly): %s\n", ok ? "PASS" : "FAIL");
  return ok;
}

} // namespace

int main() {
  bool ok = true;
  ok = test_gradient_check() && ok;
  ok = test_trains_and_generates() && ok;
  std::printf("%s\n", ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
}
