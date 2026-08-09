//===- systolic_sweep.h - systolic array design-space sweep -------------===//
//
// PLAN.md Phase 17 step 6: reuse step 5's dataflow/PPA model on
// `transformer/`'s REAL GEMM shapes -- extending `tpu_engine/mxu_opt`'s
// single-shape (one systolic array size, one workload) utilization-cliff
// finding into a genuine design-space SWEEP over PE array size, per
// workload shape, to find the utilization-maximizing configuration for
// each.
//
// GEMM shapes below are computed directly from the exact config
// `transformer/transformer_test.cpp`'s `test_trains_and_generates()`
// actually trains with: `d_model=16, num_heads=2, d_ff=32,
// max_seq_len=32`, corpus `"the quick fox jumps "` (16 unique characters
// -> `vocab_size=16`). `head_dim = d_model/num_heads = 8`. Real numbers
// from a real, already-trained model in this repo, not invented shapes.
//
//===----------------------------------------------------------------------===//
#pragma once

#include "../dataflow_model/dataflow_model.h"

#include <string>
#include <vector>

namespace analog {

struct NamedGemm {
  std::string name;
  GemmWorkload workload;
};

// Every GEMM a forward pass of transformer/'s ACTUAL trained config
// (transformer_test.cpp's test_trains_and_generates(), seq_len=32) does,
// per-layer. d_model=16, num_heads=2, head_dim=8, d_ff=32, vocab_size=16.
inline std::vector<NamedGemm> transformer_real_gemm_shapes() {
  const long seq_len = 32, d_model = 16, head_dim = 8, d_ff = 32, vocab_size = 16;
  return {
      {"QKV projection (per Q/K/V)", {seq_len, d_model, d_model}},
      {"attention scores (Q @ K^T, per head)", {seq_len, head_dim, seq_len}},
      {"attention @ V (per head)", {seq_len, seq_len, head_dim}},
      {"post-attention output projection", {seq_len, d_model, d_model}},
      {"FFN up-projection", {seq_len, d_model, d_ff}},
      {"FFN down-projection", {seq_len, d_ff, d_model}},
      {"final output projection (w_out)", {seq_len, d_model, vocab_size}},
  };
}

struct SweepPoint {
  int pe_size; // square PE array, pe_size x pe_size
  double pe_utilization;
};

// Sweep square PE array sizes and return utilization at each, for one
// GEMM workload (dataflow choice doesn't affect utilization in this
// model -- utilization depends only on M, N and the PE array's rows/cols
// -- so Weight-Stationary is used as a representative dataflow for this
// sweep; see dataflow_model.h).
inline std::vector<SweepPoint> sweep_pe_sizes(const GemmWorkload &wl, const std::vector<int> &pe_sizes,
                                               long k_tile_size) {
  std::vector<SweepPoint> out;
  for (int size : pe_sizes) {
    PEArray pe{size, size};
    auto result = analyze_dataflow(Dataflow::WeightStationary, wl, pe, k_tile_size);
    out.push_back({size, result.pe_utilization});
  }
  return out;
}

// The PE size in `pe_sizes` achieving the highest utilization for `wl`
// (ties broken toward the SMALLEST PE size -- smaller silicon area for
// the same utilization is strictly better, an explicit tie-break choice).
inline int best_pe_size(const GemmWorkload &wl, const std::vector<int> &pe_sizes, long k_tile_size) {
  auto sweep = sweep_pe_sizes(wl, pe_sizes, k_tile_size);
  int best = sweep.front().pe_size;
  double best_util = sweep.front().pe_utilization;
  for (const auto &pt : sweep) {
    if (pt.pe_utilization > best_util + 1e-12) {
      best_util = pt.pe_utilization;
      best = pt.pe_size;
    }
  }
  return best;
}

} // namespace analog
