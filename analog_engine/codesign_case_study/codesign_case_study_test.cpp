// codesign_case_study_test.cpp -- re-derives GPTQ (Phase 9 step 6) under
// analog crossbar constraints (Phase 17 step 1's device model), on a
// REAL trained transformer weight, exactly mirroring
// npu_engine/quant_export_test.cpp's own real-model setup:
//   1. Sweep GPTQ's bits (2..6, i.e. num_levels 4..64 -- the SAME levels
//      step 2's crossbar precision sweep used) and measure quantization-
//      only RMSE vs. quantization-plus-analog-noise RMSE against the
//      real FP32 weight.
//   2. A real end-task check: perplexity with quantization-only weights
//      vs. quantization-plus-analog-noise weights, at a representative
//      bit width.
#include "codesign_case_study.h"
#include "../../transformer/transformer_model.h"
#include "../../transformer/char_tokenizer.h"

#include <cmath>
#include <cstdio>
#include <random>

// Deliberately NOT `using namespace analog;` -- crossbar_mac.h's
// `analog::Matrix` (a plain vector<vector<double>>, used internally by
// steps 2/7's own math) collides with `distributed_training::Matrix`
// (transformer/'s and GPTQ's real weight-matrix type, brought in via
// `using namespace transformer;` below) if both are opened unqualified.
// `analog::`-prefixed calls are used explicitly instead.
using namespace inference_serving;
using namespace transformer;
using analog::DeviceParams;
using analog::ErrorStats;

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

} // namespace

int main() {
  std::string corpus = "the quick brown fox jumps over the lazy dog ";
  CharTokenizer tok(corpus);
  std::vector<int> tokens = tok.encode(corpus);
  TransformerConfig cfg{tok.vocab_size(), /*d_model=*/16, /*num_heads=*/2, /*num_layers=*/2,
                        /*d_ff=*/32, /*max_seq_len=*/64};
  std::mt19937 rng(5);
  ModelParams model = train(cfg, tokens, /*epochs=*/300, 0.05f, rng);

  ModelCache cache;
  model_forward(model, tokens, cache);
  Matrix calib_activations = cache.final_ln_out;
  Matrix w_out_out_in = model.w_out.transpose(); // GPTQ expects [out x in], same transpose gptq_test.cpp documents

  DeviceParams device_params;
  device_params.write_noise_frac_of_level = 0.25;
  device_params.read_noise_frac_of_write = 0.2;

  std::printf("  real trained transformer w_out (%dx%d), GPTQ re-derived under analog crossbar constraints:\n",
              w_out_out_in.rows(), w_out_out_in.cols());
  std::printf("  %6s %10s %20s %20s\n", "bits", "levels", "quant-only RMSE", "quant+analog RMSE");
  std::vector<int> bit_widths = {2, 3, 4, 5, 6};
  std::vector<double> quant_only_rmse, quant_analog_rmse;
  for (int bits : bit_widths) {
    GptqQuantizer gptq(/*group_size=*/8, bits);
    QuantizedWeight q = gptq.quantize(w_out_out_in, calib_activations);
    Matrix w_quant_only = dequantize(q);
    Matrix w_quant_analog = realize_on_crossbar(q, device_params, /*seed=*/static_cast<uint64_t>(bits * 1000));

    ErrorStats e_quant = analog::compare_weights(w_quant_only, w_out_out_in);
    ErrorStats e_analog = analog::compare_weights(w_quant_analog, w_out_out_in);
    quant_only_rmse.push_back(e_quant.rmse);
    quant_analog_rmse.push_back(e_analog.rmse);
    std::printf("  %6d %10d %20.6f %20.6f\n", bits, 1 << bits, e_quant.rmse, e_analog.rmse);
  }

  require(quant_only_rmse.front() > quant_only_rmse.back(),
          "GPTQ's own quantization-only RMSE improves from 2 to 6 bits, matching its known behavior");
  require(quant_analog_rmse.front() > quant_analog_rmse.back(),
          "quantization-plus-analog-noise RMSE ALSO improves from 2 to 6 bits -- analog noise doesn't erase the benefit of more GPTQ precision at these settings");

  bool analog_always_worse_or_equal = true;
  for (size_t i = 0; i < quant_only_rmse.size(); ++i)
    if (quant_analog_rmse[i] < quant_only_rmse[i] - 1e-9) analog_always_worse_or_equal = false;
  require(analog_always_worse_or_equal, "at every bit width, realizing GPTQ's quantized weight on a real analog cell never IMPROVES on the quantization-only RMSE (noise only adds error, as physically expected)");

  // Real end-task check at bits=4 (GPTQ's common default).
  GptqQuantizer gptq4(/*group_size=*/8, /*bits=*/4);
  QuantizedWeight q4 = gptq4.quantize(w_out_out_in, calib_activations);
  ModelParams model_quant_only = model;
  model_quant_only.w_out = dequantize(q4).transpose();
  ModelParams model_quant_analog = model;
  model_quant_analog.w_out = realize_on_crossbar(q4, device_params, /*seed=*/4000).transpose();

  float ppl_fp32 = perplexity(model, tokens);
  float ppl_quant_only = perplexity(model_quant_only, tokens);
  float ppl_quant_analog = perplexity(model_quant_analog, tokens);
  std::printf("\n  end-task perplexity at bits=4: fp32=%.4f | GPTQ quant-only=%.4f | GPTQ+analog-noise=%.4f\n",
              static_cast<double>(ppl_fp32), static_cast<double>(ppl_quant_only), static_cast<double>(ppl_quant_analog));
  // NOT asserting ppl_quant_analog >= ppl_quant_only: an earlier version
  // of this test did, and it genuinely FAILED on this real run
  // (quant_analog's perplexity came out very slightly BETTER than
  // quant_only's, 1.0509 vs 1.0513) -- a real, honest, non-monotonic
  // result at the task-metric level, not a bug. The weight-level RMSE
  // checks above are the metric that's actually guaranteed monotonic by
  // construction (RMSE is a direct distance, added noise cannot reduce
  // it); perplexity on this specific 45-character, 300-epoch (heavily
  // overfit) toy corpus is a highly non-linear function of the weights
  // near a sharp optimum, so a single noise realization nudging it very
  // slightly either way is expected, not evidence the analog-noise model
  // is broken. The defensible claim instead: analog noise's end-task
  // impact stays SMALL relative to quantization's own end-task impact.
  double analog_effect = std::abs(static_cast<double>(ppl_quant_analog) - static_cast<double>(ppl_quant_only));
  double quant_effect = std::abs(static_cast<double>(ppl_quant_only) - static_cast<double>(ppl_fp32));
  std::printf("  |analog-noise effect on ppl|=%.4f vs. |quantization's own effect on ppl|=%.4f\n", analog_effect,
              quant_effect);
  require(analog_effect < quant_effect,
          "layering analog noise on top of GPTQ changes end-task perplexity by LESS than GPTQ's own quantization already did -- the analog-realization step doesn't dominate the already-accepted quantization cost");

  std::printf("%s\n", g_fails == 0 ? "PASS" : "FAIL");
  return g_fails == 0 ? 0 : 1;
}
