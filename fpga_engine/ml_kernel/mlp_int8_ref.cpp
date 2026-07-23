// mlp_int8_ref.cpp — portable INT8-vs-float32 accuracy reference for the
// MLP shape ml_kernel.cpp synthesizes.
//
// Unlike the rest of fpga_engine/, this file has no HLS/Xilinx dependency
// (plain int8_t/int32_t, not ap_int) and actually compiles and runs on
// this Mac with a stock clang++ — no F1 instance needed. It exists to
// answer a question hardware synthesis can't: given a correct static
// per-tensor INT8 quantization scheme (calibrated scales, explicit
// requantization between layers), how much accuracy does INT8 cost
// relative to float32 on the same network shape? Resource/timing numbers
// stay hardware-gated (see ml_kernel.cpp); this accuracy number does not.
//
// Note: ml_kernel.cpp itself elides the inter-layer requantization
// multiply this file performs (its own header comment flags this as a
// simplification) — it saturates the raw accumulator straight to int8
// instead of rescaling by the calibrated activation scale first. This
// file measures the accuracy of the *correct* scheme, which is what a
// real deployment (and a future revision of the hardware kernel) should
// implement; it is not a bit-exact model of ml_kernel.cpp's current
// simplified arithmetic.
//
// Build & run:
//   clang++ -O2 -std=c++17 mlp_int8_ref.cpp -o mlp_int8_ref && ./mlp_int8_ref
#include <cstdint>
#include <cstdio>
#include <cmath>
#include <random>
#include <vector>

static constexpr int kInputs = 16;
static constexpr int kHidden = 32;
static constexpr int kOutputs = 8;
static constexpr int kCalibSamples = 200;
static constexpr int kEvalSamples = 2000;

namespace {

int8_t saturate_i8(int32_t v, int32_t lo, int32_t hi) {
    if (v < lo) return static_cast<int8_t>(lo);
    if (v > hi) return static_cast<int8_t>(hi);
    return static_cast<int8_t>(v);
}

float max_abs(const std::vector<float>& x) {
    float m = 1e-9f;
    for (float v : x) m = std::max(m, std::fabs(v));
    return m;
}

// Symmetric per-tensor quantization: scale = max(|x|) / 127.
float scale_for(const std::vector<float>& x) { return max_abs(x) / 127.0f; }

int8_t quantize(float x, float scale) {
    return saturate_i8(static_cast<int32_t>(std::lround(x / scale)), -127, 127);
}

void mlp_f32_forward(const float in[kInputs],
                      const float w1[kInputs][kHidden],
                      const float w2[kHidden][kOutputs],
                      float hidden_out[kHidden], float out[kOutputs]) {
    for (int h = 0; h < kHidden; ++h) {
        float acc = 0;
        for (int i = 0; i < kInputs; ++i) acc += in[i] * w1[i][h];
        hidden_out[h] = std::max(0.0f, acc); // ReLU
    }
    for (int o = 0; o < kOutputs; ++o) {
        float acc = 0;
        for (int h = 0; h < kHidden; ++h) acc += hidden_out[h] * w2[h][o];
        out[o] = acc;
    }
}

// Correct static INT8 forward pass: each layer's INT32 accumulator is
// explicitly rescaled back to a calibrated INT8 activation scale before
// the next layer consumes it — the requantization step ml_kernel.cpp's
// header comment flags as elided from the current hardware kernel.
void mlp_int8_forward(const int8_t in[kInputs], float in_scale,
                       const int8_t w1[kInputs][kHidden], float w1_scale,
                       const int8_t w2[kHidden][kOutputs], float w2_scale,
                       float hidden_scale,
                       int32_t out_acc[kOutputs], float* out_scale) {
    int8_t hidden_q[kHidden];
    float acc1_scale = in_scale * w1_scale;
    for (int h = 0; h < kHidden; ++h) {
        int32_t acc = 0;
        for (int i = 0; i < kInputs; ++i) acc += int32_t(in[i]) * int32_t(w1[i][h]);
        float acc_f = std::max(0.0f, acc * acc1_scale); // ReLU in the rescaled domain
        hidden_q[h] = quantize(acc_f, hidden_scale);
    }

    // Final layer's int32 accumulator is left un-saturated: it can range
    // far outside int8 (up to kHidden * 127 * 127), and saturating it to
    // int8 *before* dequantizing (an earlier version of this file did
    // exactly that) would destroy the value being measured instead of
    // just quantizing it — this layer stays in wider accumulator form
    // instead, same as a real quantized model often leaves its last layer
    // higher-precision ahead of a task head.
    float acc2_scale = hidden_scale * w2_scale;
    for (int o = 0; o < kOutputs; ++o) {
        int32_t acc = 0;
        for (int h = 0; h < kHidden; ++h) acc += int32_t(hidden_q[h]) * int32_t(w2[h][o]);
        out_acc[o] = acc;
    }
    *out_scale = acc2_scale;
}

} // namespace

int main() {
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> weight_dist(-1.0f, 1.0f);
    std::uniform_real_distribution<float> input_dist(-1.0f, 1.0f);

    float w1_f[kInputs][kHidden];
    float w2_f[kHidden][kOutputs];
    for (auto& row : w1_f) for (auto& v : row) v = weight_dist(rng);
    for (auto& row : w2_f) for (auto& v : row) v = weight_dist(rng);

    std::vector<float> w1_flat, w2_flat;
    for (auto& row : w1_f) for (float v : row) w1_flat.push_back(v);
    for (auto& row : w2_f) for (float v : row) w2_flat.push_back(v);
    float w1_scale = scale_for(w1_flat);
    float w2_scale = scale_for(w2_flat);

    int8_t w1_q[kInputs][kHidden], w2_q[kHidden][kOutputs];
    for (int i = 0; i < kInputs; ++i)
        for (int h = 0; h < kHidden; ++h) w1_q[i][h] = quantize(w1_f[i][h], w1_scale);
    for (int h = 0; h < kHidden; ++h)
        for (int o = 0; o < kOutputs; ++o) w2_q[h][o] = quantize(w2_f[h][o], w2_scale);

    // Calibration pass: run float32 forward over a held-out calibration
    // set purely to find a representative hidden-activation range, then
    // fix hidden_scale for the whole evaluation — static quantization,
    // not per-sample dynamic rescaling.
    std::vector<float> hidden_samples;
    for (int s = 0; s < kCalibSamples; ++s) {
        float in_f[kInputs], hidden_f[kHidden], out_f[kOutputs];
        for (auto& v : in_f) v = input_dist(rng);
        mlp_f32_forward(in_f, w1_f, w2_f, hidden_f, out_f);
        for (float v : hidden_f) hidden_samples.push_back(v);
    }
    float hidden_scale = scale_for(hidden_samples);

    // Evaluation: independent samples, float32 vs. calibrated-INT8 forward.
    double sum_sq_err = 0.0, sum_sq_ref = 0.0;
    for (int s = 0; s < kEvalSamples; ++s) {
        float in_f[kInputs], hidden_f[kHidden], out_f[kOutputs];
        for (auto& v : in_f) v = input_dist(rng);
        mlp_f32_forward(in_f, w1_f, w2_f, hidden_f, out_f);

        std::vector<float> in_vec(in_f, in_f + kInputs);
        float in_scale = scale_for(in_vec);
        int8_t in_q[kInputs];
        for (int i = 0; i < kInputs; ++i) in_q[i] = quantize(in_f[i], in_scale);

        int32_t out_acc[kOutputs];
        float out_scale;
        mlp_int8_forward(in_q, in_scale, w1_q, w1_scale, w2_q, w2_scale,
                          hidden_scale, out_acc, &out_scale);

        for (int o = 0; o < kOutputs; ++o) {
            double err = out_acc[o] * out_scale - out_f[o];
            sum_sq_err += err * err;
            sum_sq_ref += double(out_f[o]) * out_f[o];
        }
    }

    double rms_err = std::sqrt(sum_sq_err / (kEvalSamples * kOutputs));
    double rms_ref = std::sqrt(sum_sq_ref / (kEvalSamples * kOutputs));
    std::printf("eval_samples=%d  hidden_scale=%.6f  RMS error=%.6f  "
                "RMS reference magnitude=%.6f  relative=%.4f%%\n",
                kEvalSamples, hidden_scale, rms_err, rms_ref, 100.0 * rms_err / rms_ref);
    return 0;
}
