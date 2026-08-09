//===- codesign_case_study.h - GPTQ re-derived under analog constraints -===//
//
// PLAN.md Phase 17 step 7: a hardware-algorithm co-design case study --
// re-derive one of this repo's real algorithms (Phase 9 step 6's
// `inference_serving::GptqQuantizer`, Frantar et al. 2022) under
// analog-specific constraints, using Phase 17's OWN device model
// (`analog::ConductanceCell`, step 1) rather than inventing a new,
// separate analog error model for this step.
//
// The real co-design question: GPTQ's `bits` parameter already IS a
// discrete-level count (`num_levels = 2^bits`) -- the exact same
// quantity step 2's crossbar precision sweep varied. This step asks what
// happens when GPTQ's quantized integer level isn't stored in a
// noiseless digital register, but PROGRAMMED into a real (simulated)
// analog crossbar cell: does GPTQ's own precision choice still matter
// once step 1's write/read noise is layered on top, or does the analog
// noise floor dominate at some point, making extra GPTQ precision a
// wasted bit budget?
//
// Pipeline: GPTQ's quantized integer level (0..num_levels-1) -> program
// a `ConductanceCell` to that level (step 1, write noise) -> read it back
// (step 1, read noise) -> nearest-level decode (step 2's own
// `quantize_level`-style rounding) -> GPTQ's own scale/zero dequantization
// formula. Every stage reuses an existing, already-tested component; the
// only new code here is the pipeline connecting them.
//
//===----------------------------------------------------------------------===//
#pragma once

#include "../crossbar_mac/crossbar_mac.h" // for quantize_level
#include "../device_model/device_model.h"
#include "../../inference_serving/gptq/gptq.h"

#include <cmath>

namespace analog {

using inference_serving::QuantizedWeight;
// NOTE: not aliased to plain `Matrix` -- crossbar_mac.h already defines
// `analog::Matrix` as `std::vector<std::vector<double>>` for its own
// purposes, so `distributed_training::Matrix` (transformer/'s real
// weight-matrix type, and what GPTQ operates on) is referred to by its
// fully qualified name throughout this file to avoid colliding with it.

// Simulates programming every element of `q` (GPTQ's quantized integer
// levels) into a real analog crossbar cell and reading it back, then
// dequantizes using GPTQ's OWN scale/zero formula -- so any error beyond
// GPTQ's own (already-measured) quantization error is attributable
// specifically to the analog write/read noise layered on top, not a
// second, different rounding scheme.
inline distributed_training::Matrix realize_on_crossbar(const QuantizedWeight &q, DeviceParams device_params,
                                                          uint64_t seed = 0) {
  device_params.num_levels = 1 << q.bits; // GPTQ's own level count, reused exactly -- not re-chosen here
  distributed_training::Matrix out(q.rows, q.cols);
  uint64_t cell_seed = seed;
  for (int r = 0; r < q.rows; ++r) {
    for (int c = 0; c < q.cols; ++c) {
      int level = q.qweight[static_cast<size_t>(r * q.cols + c)];
      ConductanceCell cell(device_params, cell_seed++);
      cell.write(level);
      double measured = cell.read(0.0);
      double step = cell.level_step();
      int decoded_level = static_cast<int>(std::round((measured - device_params.g_min) / step));
      decoded_level = std::clamp(decoded_level, 0, device_params.num_levels - 1);

      int group = c / q.group_size;
      float scale = q.scales[static_cast<size_t>(r * q.num_groups() + group)];
      int32_t zero = q.zeros[static_cast<size_t>(r * q.num_groups() + group)];
      out(r, c) = static_cast<float>(decoded_level - zero) * scale;
    }
  }
  return out;
}

struct ErrorStats {
  double rmse;
};

inline ErrorStats compare_weights(const distributed_training::Matrix &a, const distributed_training::Matrix &b) {
  double sq = 0.0;
  int n = 0;
  for (int r = 0; r < a.rows(); ++r)
    for (int c = 0; c < a.cols(); ++c) {
      double d = static_cast<double>(a(r, c) - b(r, c));
      sq += d * d;
      ++n;
    }
  return {std::sqrt(sq / std::max(1, n))};
}

} // namespace analog
