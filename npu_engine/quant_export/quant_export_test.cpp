// quant_export_test.cpp — PLAN.md Phase 15 step 2. Reuses
// inference_serving::GptqQuantizer (unchanged) at bits=8 on a real
// trained transformer's w_out weight with real calibration activations
// -- exactly gptq_test.cpp's own setup, since this step's job is the
// *export* format, not a new quantization algorithm. Two things measured
// for real: (1) round-trip byte-exact serialize/deserialize through this
// step's custom binary format, (2) real FP32 vs. INT8 compression ratio
// and perplexity impact, matching gptq_test.cpp's perplexity-ordering
// check but at bits=8 instead of bits=4.
#include "quant_export.h"
#include "../../transformer/transformer_model.h"
#include "../../transformer/char_tokenizer.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>

using namespace inference_serving;
using namespace transformer;
using namespace npu_engine;

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

float perplexity(const ModelParams &model, const std::vector<int> &tokens) {
  ModelCache cache;
  Matrix logits = model_forward(model, tokens, cache);
  return std::exp(next_token_loss(logits, tokens).loss);
}

void test_roundtrip_byte_exact() {
  std::mt19937 rng(3);
  Matrix w = Matrix::random(6, 16, rng, 1.0f);
  Matrix x = Matrix::random(50, 16, rng, 1.0f);

  GptqQuantizer gptq(/*group_size=*/8, /*bits=*/8);
  QuantizedWeight q = gptq.quantize(w, x);

  std::vector<uint8_t> bytes = serialize_npu_weight(q);
  QuantizedWeight q2 = deserialize_npu_weight(bytes);

  bool exact = q.rows == q2.rows && q.cols == q2.cols && q.group_size == q2.group_size &&
               q.bits == q2.bits && q.qweight == q2.qweight && q.scales == q2.scales && q.zeros == q2.zeros;
  require(exact, "serialize -> deserialize round-trip is byte-exact");

  const std::string path = "/tmp/npu_quant_export_roundtrip_test.npuw";
  write_npu_weight_file(path, q);
  QuantizedWeight q3 = read_npu_weight_file(path);
  bool file_exact = q.qweight == q3.qweight && q.scales == q3.scales && q.zeros == q3.zeros;
  require(file_exact, "write_npu_weight_file -> read_npu_weight_file round-trip is byte-exact");
  std::remove(path.c_str());
}

void test_real_model_export() {
  std::string corpus = "the quick brown fox jumps over the lazy dog ";
  CharTokenizer tok(corpus);
  std::vector<int> tokens = tok.encode(corpus);

  TransformerConfig cfg{tok.vocab_size(), /*d_model=*/16, /*num_heads=*/2, /*num_layers=*/2,
                        /*d_ff=*/32, /*max_seq_len=*/64};
  std::mt19937 rng(5);
  ModelParams model = train(cfg, tokens, /*epochs=*/300, 0.05f, rng);

  ModelCache cache;
  Matrix fp32_logits = model_forward(model, tokens, cache);
  Matrix calib_activations = cache.final_ln_out;
  float fp32_ppl = std::exp(next_token_loss(fp32_logits, tokens).loss);

  // Same [out x in] transpose gotcha gptq_test.cpp documents: transformer/
  // stores w_out as [d_model x vocab_size] ([in x out]); GptqQuantizer
  // expects [out_features x in_features].
  Matrix w_out_out_in = model.w_out.transpose();
  GptqQuantizer gptq(/*group_size=*/8, /*bits=*/8);
  QuantizedWeight q_gptq = gptq.quantize(w_out_out_in, calib_activations);
  QuantizedWeight q_rtn = quantize_rtn(w_out_out_in, /*group_size=*/8, /*bits=*/8);

  ModelParams model_gptq = model;
  model_gptq.w_out = dequantize(q_gptq).transpose();
  ModelParams model_rtn = model;
  model_rtn.w_out = dequantize(q_rtn).transpose();

  float gptq_ppl = perplexity(model_gptq, tokens);
  float rtn_ppl = perplexity(model_rtn, tokens);

  std::printf("  perplexity: fp32=%.4f  gptq-int8=%.4f  rtn-int8=%.4f\n",
              static_cast<double>(fp32_ppl), static_cast<double>(gptq_ppl), static_cast<double>(rtn_ppl));
  require(fp32_ppl <= gptq_ppl + 1e-3f && gptq_ppl <= rtn_ppl + 1e-3f,
          "quantization ordering holds at INT8: fp32 <= gptq-int8 <= rtn-int8 perplexity (within fp tolerance)");

  const std::string path = "/tmp/npu_quant_export_wout.npuw";
  write_npu_weight_file(path, q_gptq);
  QuantizedWeight loaded = read_npu_weight_file(path);
  require(loaded.qweight == q_gptq.qweight && loaded.scales == q_gptq.scales,
          "real w_out export round-trips through the on-disk .npuw format");

  size_t fp32_bytes = fp32_byte_size(q_gptq);
  size_t npu_bytes = npu_export_byte_size(q_gptq);
  double ratio = static_cast<double>(fp32_bytes) / static_cast<double>(npu_bytes);
  std::printf("  w_out export size: fp32=%zu bytes, npu-int8=%zu bytes, ratio=%.3fx\n",
              fp32_bytes, npu_bytes, ratio);
  // INT8 payload alone is exactly 4x smaller than FP32; the file's ratio
  // is slightly under 4x because it also carries float32 scales + int32
  // zero-points per group (real per-file overhead, not hidden).
  require(ratio > 3.0 && ratio < 4.0,
          "INT8 export is a real, substantial (but not exactly 4x, due to scale/zero-point overhead) size reduction vs FP32");
  std::remove(path.c_str());
}

} // namespace

int main() {
  test_roundtrip_byte_exact();
  test_real_model_export();
  std::printf("%s\n", g_fails == 0 ? "PASS" : "FAIL");
  return g_fails == 0 ? 0 : 1;
}
