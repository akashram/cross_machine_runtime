//===- dataflow_model.h - minimal Timeloop/Accelergy-style PPA model ----===//
//
// PLAN.md Phase 17 step 5: a from-scratch, minimal reproduction of what
// Parashar et al. (2019) "Timeloop" and Wu, Emer & Sze (2019) "Accelergy"
// compute -- given a GEMM workload and a PE-array architecture
// description, analytically count data-movement volume per operand and
// estimate energy, for each of the four dataflow strategies Sze, Chen,
// Yang & Emer (2017) survey: Weight-/Output-/Input-/Row-Stationary.
//
// Model: a 2-level memory hierarchy (DRAM <-> PE-array-local registers),
// GEMM C[M,N] = A[M,K]*B[K,N] tiled onto a Prows x Pcols PE array with the
// K dimension itself tiled into `k_tile_size`-deep passes (so a dataflow
// that does NOT keep partial sums resident across K-tile passes pays a
// real read-back-and-accumulate cost -- this is what actually
// distinguishes Output-Stationary's advantage from Weight-/Input-
// Stationary; without K-tiling that distinction wouldn't show up
// numerically, an honestly-disclosed simplification a version of this
// model without K-tiling would have).
//
// Row-Stationary is a GEMM-ADAPTED simplification of Eyeriss's actual
// convolution-specific row mapping (Chen, Emer & Sze 2016/2017): rather
// than literally reproducing Eyeriss's 1D-row-of-a-conv-filter reuse
// pattern (which needs a spatial conv dimension this GEMM-only model
// doesn't have), RS here holds a K-tile-sized ROW of BOTH weight and
// input resident simultaneously and reuses each across the FULL output
// tile space before moving to the next K-tile -- capturing RS's real
// headline property (near-ideal reuse of TWO operands simultaneously,
// where WS/IS only achieve that for ONE), at the same K-tile partial-sum
// round-trip cost WS/IS pay. A disclosed adaptation, not a literal
// Eyeriss reproduction -- same pattern as every other "simplified vs. the
// full paper" note in this repo.
//
//===----------------------------------------------------------------------===//
#pragma once

#include <algorithm>
#include <string>
#include <vector>

namespace analog {

struct GemmWorkload {
  long M, K, N;
};

struct PEArray {
  int rows, cols; // Prows x Pcols
};

enum class Dataflow { WeightStationary, OutputStationary, InputStationary, RowStationary };

inline const char *dataflow_name(Dataflow d) {
  switch (d) {
  case Dataflow::WeightStationary: return "Weight-Stationary";
  case Dataflow::OutputStationary: return "Output-Stationary";
  case Dataflow::InputStationary: return "Input-Stationary";
  case Dataflow::RowStationary: return "Row-Stationary";
  }
  return "?";
}

struct DataflowResult {
  Dataflow dataflow;
  long weight_elements_moved, input_elements_moved, output_elements_moved;
  double pe_utilization; // fraction of the allocated (tile-padded) PE array actually doing useful work
  double energy_pj;
};

inline long ceil_div(long a, long b) { return (a + b - 1) / b; }

// DRAM access energy per element moved -- an illustrative literature-
// order-of-magnitude constant (same "illustrative, not measured, no fab
// access" framing as step 4's ADC-per-bit constant), not a specific
// technology-node datasheet number.
inline constexpr double kDramPjPerElement = 20.0;

inline DataflowResult analyze_dataflow(Dataflow df, const GemmWorkload &wl, const PEArray &pe, long k_tile_size) {
  long m_tiles = ceil_div(wl.M, pe.rows);
  long n_tiles = ceil_div(wl.N, pe.cols);
  long k_tiles = ceil_div(wl.K, k_tile_size);

  long padded_M = m_tiles * pe.rows;
  long padded_N = n_tiles * pe.cols;
  long padded_K = k_tiles * k_tile_size;

  long padded_size_A = padded_M * padded_K;
  long padded_size_B = padded_K * padded_N;
  long padded_size_C = padded_M * padded_N;

  long weight_moved = 0, input_moved = 0, output_moved = 0;
  long partial_sum_round_trips = 2 * k_tiles - 1; // write-back + read-back-to-accumulate, per non-final K-tile pass

  switch (df) {
  case Dataflow::WeightStationary:
    weight_moved = padded_size_B;               // loaded once total: the ideal this dataflow targets
    input_moved = n_tiles * padded_size_A;       // re-streamed once per N-tile (not stationary)
    output_moved = partial_sum_round_trips * padded_size_C;
    break;
  case Dataflow::InputStationary:
    input_moved = padded_size_A;                 // loaded once total
    weight_moved = m_tiles * padded_size_B;       // re-streamed once per M-tile
    output_moved = partial_sum_round_trips * padded_size_C;
    break;
  case Dataflow::OutputStationary:
    weight_moved = m_tiles * padded_size_B;       // neither operand is stationary across a tile loop
    input_moved = n_tiles * padded_size_A;
    output_moved = padded_size_C;                 // but output stays resident across ALL K-tile passes: no round trips
    break;
  case Dataflow::RowStationary:
    weight_moved = padded_size_B;                 // near-ideal reuse of BOTH operands simultaneously --
    input_moved = padded_size_A;                  // RS's real headline property vs. WS/IS's single-operand focus
    output_moved = partial_sum_round_trips * padded_size_C; // still pays the K-tile partial-sum cost, unlike OS
    break;
  }

  double pe_utilization = static_cast<double>(wl.M * wl.N) / static_cast<double>(padded_M * padded_N);
  double energy = static_cast<double>(weight_moved + input_moved + output_moved) * kDramPjPerElement;

  return {df, weight_moved, input_moved, output_moved, pe_utilization, energy};
}

inline std::vector<DataflowResult> analyze_all(const GemmWorkload &wl, const PEArray &pe, long k_tile_size) {
  std::vector<DataflowResult> out;
  for (Dataflow d : {Dataflow::WeightStationary, Dataflow::OutputStationary, Dataflow::InputStationary,
                      Dataflow::RowStationary})
    out.push_back(analyze_dataflow(d, wl, pe, k_tile_size));
  return out;
}

} // namespace analog
