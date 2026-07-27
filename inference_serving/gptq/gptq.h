#pragma once
#include "../../distributed_training/autograd/matrix.h"

#include <cstdint>
#include <vector>

// PLAN.md Phase 9 step 6: GPTQ INT4 group quantization (Frantar et al.
// 2022) — Hessian-guided greedy optimization: quantize one column at a
// time, and after each column, push the resulting rounding error onto
// the not-yet-quantized columns via the calibration Hessian's inverse, so
// later columns partially compensate for earlier columns' rounding
// error instead of every column rounding independently (round-to-nearest,
// "RTN" — this file's baseline for comparison).
//
// Deliberate deviation from the original stub's raw-pointer signature:
// uses distributed_training::Matrix (the same type transformer/'s real
// model is built on) so this can quantize transformer/'s actual trained
// weights and measure real perplexity, not a synthetic buffer. See
// README.md's Design section and gptq_test.cpp for the end-to-end
// measurement.

namespace inference_serving {

using distributed_training::Matrix;

// One quantized matrix: qweight holds INT values (stored as int32_t for
// simplicity — no bit-packing; packing 4-bit values two-per-byte is a
// storage-format concern orthogonal to the quantization algorithm this
// step is about) in [0, 2^bits - 1], per-(row, group) scale and zero
// point for asymmetric affine dequantization: w ≈ (q - zero) * scale.
struct QuantizedWeight {
  std::vector<int32_t> qweight;  // rows * cols, row-major
  std::vector<float> scales;     // rows * num_groups
  std::vector<int32_t> zeros;    // rows * num_groups
  int rows = 0, cols = 0, group_size = 0, bits = 4;

  int num_groups() const { return (cols + group_size - 1) / group_size; }
};

// Round-to-nearest baseline: per-(row,group) affine quantization with no
// error compensation. What every column would get under GPTQ if the
// Hessian were the identity (no cross-column correlation to exploit) —
// the reference point GPTQ's whole value proposition is measured against.
QuantizedWeight quantize_rtn(const Matrix &weight, int group_size, int bits);

Matrix dequantize(const QuantizedWeight &q);

class GptqQuantizer {
 public:
  explicit GptqQuantizer(int group_size = 128, int bits = 4) : group_size_(group_size), bits_(bits) {}

  // calibration_activations: [num_samples x cols] — real activations that
  // feed INTO this weight matrix (weight is [rows x cols], so
  // activations @ weight^T is the layer this weight belongs to). The
  // Hessian H = 2 * X^T X is computed from these, shared across every
  // output row (rows), since it depends only on the layer's input
  // distribution, not which output channel is being quantized — the same
  // Hessian-sharing the real GPTQ algorithm relies on for efficiency.
  QuantizedWeight quantize(const Matrix &weight, const Matrix &calibration_activations) const;

 private:
  int group_size_, bits_;
};

}  // namespace inference_serving
