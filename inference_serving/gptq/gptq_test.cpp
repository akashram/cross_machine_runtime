// gptq_test.cpp — two checks: (1) a synthetic property test that GPTQ's
// Hessian-weighted output error beats plain round-to-nearest on
// correlated calibration data (the actual objective GPTQ minimizes,
// ||X(W - Wq)^T||^2 -- not raw weight reconstruction error, which GPTQ
// does NOT directly minimize and can occasionally be worse on); (2) a
// real end-to-end perplexity measurement quantizing transformer/'s
// trained output-projection weight with real calibration activations
// from an actual forward pass.
#include "gptq.h"
#include "../../transformer/transformer_model.h"
#include "../../transformer/char_tokenizer.h"

#include <cmath>
#include <cstdio>
#include <random>

using namespace inference_serving;
using namespace transformer;

namespace {

int g_fails = 0;
void require(bool ok, const char *name) {
  std::printf("%s  %s\n", ok ? "PASS" : "FAIL", name);
  if (!ok) ++g_fails;
}

double output_error(const Matrix &x, const Matrix &w_true, const Matrix &w_approx) {
  Matrix y_true = x.matmul(w_true.transpose());
  Matrix y_approx = x.matmul(w_approx.transpose());
  double sq = 0.0;
  for (int i = 0; i < y_true.rows(); ++i)
    for (int j = 0; j < y_true.cols(); ++j) {
      double d = static_cast<double>(y_true(i, j)) - static_cast<double>(y_approx(i, j));
      sq += d * d;
    }
  return sq / (y_true.rows() * y_true.cols());
}

void test_gptq_beats_rtn_on_output_error() {
  constexpr int kCols = 16, kRows = 6, kSamples = 300;
  std::mt19937 rng(21);

  // Correlated calibration activations: project a lower-dim latent
  // through a random mixing matrix, so the Hessian has real off-diagonal
  // structure for GPTQ's error compensation to exploit -- i.i.d. columns
  // would give a near-diagonal Hessian, where GPTQ degenerates to RTN by
  // construction (nothing to compensate against), so that would be too
  // easy a test to be meaningful.
  constexpr int kLatentDim = 4;
  Matrix mix = Matrix::random(kLatentDim, kCols, rng, 1.0f);
  Matrix latent = Matrix::random(kSamples, kLatentDim, rng, 1.0f);
  Matrix x = latent.matmul(mix);

  Matrix w = Matrix::random(kRows, kCols, rng, 1.0f);

  GptqQuantizer gptq(/*group_size=*/8, /*bits=*/4);
  QuantizedWeight q_gptq = gptq.quantize(w, x);
  QuantizedWeight q_rtn = quantize_rtn(w, /*group_size=*/8, /*bits=*/4);

  Matrix w_gptq = dequantize(q_gptq);
  Matrix w_rtn = dequantize(q_rtn);

  double err_gptq = output_error(x, w, w_gptq);
  double err_rtn = output_error(x, w, w_rtn);
  std::printf("  output MSE: gptq=%.6f rtn=%.6f\n", err_gptq, err_rtn);
  require(err_gptq <= err_rtn, "GPTQ's Hessian-weighted output error is no worse than RTN's on correlated calibration data");
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

float perplexity(const ModelParams &model, const std::vector<int> &tokens) {
  ModelCache cache;
  Matrix logits = model_forward(model, tokens, cache);
  return std::exp(next_token_loss(logits, tokens).loss);
}

void test_real_model_perplexity() {
  std::string corpus = "the quick brown fox jumps over the lazy dog ";
  CharTokenizer tok(corpus);
  std::vector<int> tokens = tok.encode(corpus);

  TransformerConfig cfg{tok.vocab_size(), /*d_model=*/16, /*num_heads=*/2, /*num_layers=*/2,
                         /*d_ff=*/32, /*max_seq_len=*/64};
  std::mt19937 rng(5);
  ModelParams model = train(cfg, tokens, /*epochs=*/300, 0.05f, rng);

  // Real calibration activations: the actual final-layernorm output this
  // trained model produces on its own training corpus -- exactly what
  // w_out consumes as input, captured from a real forward pass, not
  // synthesized.
  ModelCache cache;
  Matrix fp32_logits = model_forward(model, tokens, cache);
  Matrix calib_activations = cache.final_ln_out;
  float fp32_ppl = std::exp(next_token_loss(fp32_logits, tokens).loss);

  // GptqQuantizer expects the standard [out_features x in_features]
  // layout (rows = out_features, cols = in_features, matching
  // calibration_activations' column count for the Hessian). transformer/
  // stores w_out as [d_model x vocab_size] -- [in x out], the opposite
  // convention (it's used as final_ln_out.matmul(w_out) directly, no
  // transpose, in model_forward) -- so transpose in and back out.
  Matrix w_out_out_in = model.w_out.transpose();  // [vocab_size x d_model]
  GptqQuantizer gptq(/*group_size=*/8, /*bits=*/4);
  QuantizedWeight q_gptq = gptq.quantize(w_out_out_in, calib_activations);
  QuantizedWeight q_rtn = quantize_rtn(w_out_out_in, /*group_size=*/8, /*bits=*/4);

  ModelParams model_gptq = model;
  model_gptq.w_out = dequantize(q_gptq).transpose();
  ModelParams model_rtn = model;
  model_rtn.w_out = dequantize(q_rtn).transpose();

  float gptq_ppl = perplexity(model_gptq, tokens);
  float rtn_ppl = perplexity(model_rtn, tokens);

  std::printf("  perplexity: fp32=%.4f  gptq-int4=%.4f  rtn-int4=%.4f\n",
              static_cast<double>(fp32_ppl), static_cast<double>(gptq_ppl), static_cast<double>(rtn_ppl));
  require(fp32_ppl <= gptq_ppl && gptq_ppl <= rtn_ppl + 1e-3f,
          "quantization ordering holds: fp32 <= gptq-int4 <= rtn-int4 perplexity");
}

} // namespace

int main() {
  test_gptq_beats_rtn_on_output_error();
  test_real_model_perplexity();
  std::printf("%s\n", g_fails == 0 ? "PASS" : "FAIL");
  return g_fails == 0 ? 0 : 1;
}
