#pragma once

// 2:4 structured sparsity during training: PLAN.md step 21. The NVIDIA
// Ampere+ pattern: within every contiguous group of 4 weights along a
// row, exactly 2 survive (the largest-magnitude 2) and 2 are pruned to
// zero — the pattern sparse Tensor Cores can skip for a real hardware
// throughput gain (already covered, hardware-gated, by Phase 3's
// gpu_engine/sparsity/). This step's scope is different and CPU-testable:
// does the model still learn once 50% of a layer's weights are pinned to
// zero? Real throughput needs real sparse Tensor Core execution — nothing
// to measure here without that hardware (see README.md).
//
// Recipe used: train dense for a warmup period, prune ONCE to 2:4 (NVIDIA's
// documented ASP recipe — prune, then fine-tune with the mask held FIXED,
// rather than repruning every step from scratch), then continue training
// with the mask re-applied to both weight and gradient every step so
// pruned entries never drift away from exactly zero.

#include <algorithm>
#include <array>
#include <cmath>

#include "matrix.h"

namespace distributed_training {

// For weight W[in x out] used as x @ W (in is the matmul's reduction/K
// axis), groups along ROWS — for each fixed output column, every group of
// 4 consecutive input rows keeps the 2 largest-magnitude entries (mask=1)
// and zeros the other 2 (mask=0). Grouping along the K axis (not the
// output axis) matches the real hardware convention: it is what lets a
// sparse Tensor Core skip the zeroed multiply-accumulates for a fixed
// output element. `in` (w.rows()) must be a multiple of 4. Ties broken by
// row index (lower index wins) — deterministic, not that it matters for
// correctness.
inline Matrix compute_2_4_mask(const Matrix &w) {
  int rows = w.rows(), cols = w.cols();
  Matrix mask(rows, cols);
  for (int c = 0; c < cols; ++c) {
    for (int g = 0; g < rows; g += 4) {
      std::array<int, 4> idx{g, g + 1, g + 2, g + 3};
      std::sort(idx.begin(), idx.end(),
                [&](int a, int b) { return std::abs(w(a, c)) > std::abs(w(b, c)); });
      for (int k = 0; k < 4; ++k) mask(idx[static_cast<size_t>(k)], c) = (k < 2) ? 1.0f : 0.0f;
    }
  }
  return mask;
}

inline void apply_mask_inplace(Matrix &m, const Matrix &mask) {
  for (int r = 0; r < m.rows(); ++r)
    for (int c = 0; c < m.cols(); ++c) m(r, c) *= mask(r, c);
}

// Verifies the 2:4 property actually holds along the K (row) axis: every
// group of 4 consecutive rows, for every column, has exactly 2 nonzero
// entries. Used by the test as a structural sanity check, not just
// trusting compute_2_4_mask's own logic.
inline bool verify_2_4_property(const Matrix &w) {
  for (int c = 0; c < w.cols(); ++c) {
    for (int g = 0; g < w.rows(); g += 4) {
      int nonzero = 0;
      for (int k = 0; k < 4; ++k)
        if (w(g + k, c) != 0.0f) ++nonzero;
      if (nonzero != 2) return false;
    }
  }
  return true;
}

} // namespace distributed_training
