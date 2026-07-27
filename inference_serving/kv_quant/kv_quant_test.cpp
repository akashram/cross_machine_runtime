// kv_quant_test.cpp — three checks: (1) INT8 quantize/dequantize
// round-trip error is small and bounded on random data; (2) the exact
// memory-reduction factor at a realistic sequence length; (3) a real
// end-to-end perplexity comparison, FP32 KV vs INT8-quantized-then-
// dequantized KV, on transformer/'s actual trained model.
#include "kv_quant.h"
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

void test_quantize_roundtrip_error_bounded() {
  std::mt19937 rng(13);
  Matrix m = Matrix::random(20, 16, rng, 2.0f);
  Int8Tensor q = quantize_int8_symmetric(m);
  Matrix dq = dequantize_int8(q);

  float max_abs = 0.0f;
  for (int i = 0; i < m.rows(); ++i)
    for (int j = 0; j < m.cols(); ++j) max_abs = std::max(max_abs, std::abs(m(i, j)));
  // Max quantization error for symmetric INT8 (127 levels) is at most
  // half a quantization step: scale/2 = max_abs/254.
  float max_err = 0.0f;
  for (int i = 0; i < m.rows(); ++i)
    for (int j = 0; j < m.cols(); ++j) max_err = std::max(max_err, std::abs(m(i, j) - dq(i, j)));
  float bound = max_abs / 254.0f + 1e-6f;
  std::printf("  max quantization error=%.6f  theoretical bound=%.6f\n",
              static_cast<double>(max_err), static_cast<double>(bound));
  require(max_err <= bound, "INT8 round-trip error stays within the theoretical half-step bound");
}

void test_memory_reduction() {
  KvMemoryComparison r = compare_kv_memory(/*seq_len=*/2048, /*d_model=*/4096, /*num_layers=*/32);
  std::printf("  KV cache @ seq=2048 d_model=4096 layers=32: fp32=%zu bytes, int8=%zu bytes, reduction=%.3fx\n",
              r.fp32_bytes, r.int8_bytes, r.reduction_factor);
  require(r.reduction_factor > 3.99 && r.reduction_factor < 4.0, "memory reduction is ~4x (scale overhead negligible at this size)");
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

void test_real_model_perplexity() {
  std::string corpus = "the quick brown fox jumps over the lazy dog ";
  CharTokenizer tok(corpus);
  std::vector<int> tokens = tok.encode(corpus);

  TransformerConfig cfg{tok.vocab_size(), /*d_model=*/16, /*num_heads=*/2, /*num_layers=*/2,
                         /*d_ff=*/32, /*max_seq_len=*/64};
  std::mt19937 rng(5);
  ModelParams model = train(cfg, tokens, /*epochs=*/300, 0.05f, rng);

  ModelCache cache;
  Matrix fp32_logits = model_forward(model, tokens, cache);
  float fp32_ppl = std::exp(next_token_loss(fp32_logits, tokens).loss);

  Matrix int8kv_logits = model_forward_int8kv(model, tokens);
  float int8kv_ppl = std::exp(next_token_loss(int8kv_logits, tokens).loss);

  std::printf("  perplexity: fp32-kv=%.4f  int8-kv=%.4f  (ratio=%.4f)\n",
              static_cast<double>(fp32_ppl), static_cast<double>(int8kv_ppl),
              static_cast<double>(int8kv_ppl / fp32_ppl));
  // A real accuracy check, not just "it runs": INT8 KV must stay close
  // to FP32 (within 10% relative perplexity) to be a viable memory/
  // accuracy tradeoff -- if it weren't, that would be the real, reportable
  // finding instead.
  require(int8kv_ppl < fp32_ppl * 1.10f, "INT8 KV cache perplexity stays within 10% of FP32");
}

} // namespace

int main() {
  test_quantize_roundtrip_error_bounded();
  test_memory_reduction();
  test_real_model_perplexity();
  std::printf("%s\n", g_fails == 0 ? "PASS" : "FAIL");
  return g_fails == 0 ? 0 : 1;
}
