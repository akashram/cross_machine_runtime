// dataflow_model_test.cpp -- real, measured checks on the analytical
// loop-nest data-movement model:
//   1. Each single-operand-stationary dataflow (WS/IS) genuinely
//      minimizes ITS operand's movement among all four -- structural
//      correctness of the model, not just "it runs."
//   2. Row-Stationary's real headline property: it matches WS's weight
//      movement AND IS's input movement SIMULTANEOUSLY, something no
//      single-operand-stationary dataflow achieves.
//   3. Output-Stationary genuinely minimizes output/partial-sum movement.
//   4. PE utilization drops below 1.0 for GEMM dimensions that don't
//      divide evenly into the PE array -- connecting directly to
//      tpu_engine/mxu_opt's own 128-boundary utilization-cliff finding.
#include "dataflow_model.h"

#include <cstdio>

using namespace analog;

namespace {

int g_fails = 0;
void require(bool ok, const char *name) {
  std::printf("%s  %s\n", ok ? "PASS" : "FAIL", name);
  if (!ok) ++g_fails;
}

} // namespace

int main() {
  // 512x512x512, matching the same matmul shape npu_engine/cost_model
  // already uses; 128x128 PE array + 128-deep K-tiling, matching
  // tpu_engine/mxu_opt's own 128-boundary MXU alignment.
  GemmWorkload wl{512, 512, 512};
  PEArray pe{128, 128};
  long k_tile = 128;

  auto results = analyze_all(wl, pe, k_tile);
  std::printf("  GEMM %ldx%ldx%ld on a %dx%d PE array, k_tile=%ld:\n", wl.M, wl.K, wl.N, pe.rows, pe.cols, k_tile);
  std::printf("  %-20s %14s %14s %14s %10s %12s\n", "dataflow", "weight_moved", "input_moved", "output_moved",
              "pe_util", "energy_pJ");
  for (const auto &r : results)
    std::printf("  %-20s %14ld %14ld %14ld %10.3f %12.1f\n", dataflow_name(r.dataflow), r.weight_elements_moved,
                r.input_elements_moved, r.output_elements_moved, r.pe_utilization, r.energy_pj);

  auto find = [&](Dataflow d) -> const DataflowResult & {
    for (const auto &r : results)
      if (r.dataflow == d) return r;
    return results[0];
  };
  const auto &ws = find(Dataflow::WeightStationary);
  const auto &is = find(Dataflow::InputStationary);
  const auto &os = find(Dataflow::OutputStationary);
  const auto &rs = find(Dataflow::RowStationary);

  long min_weight = ws.weight_elements_moved;
  long min_input = is.input_elements_moved;
  long min_output = os.output_elements_moved;
  for (const auto &r : results) {
    min_weight = std::min(min_weight, r.weight_elements_moved);
    min_input = std::min(min_input, r.input_elements_moved);
    min_output = std::min(min_output, r.output_elements_moved);
  }

  require(ws.weight_elements_moved == min_weight, "Weight-Stationary achieves the minimum weight movement among all four dataflows");
  require(is.input_elements_moved == min_input, "Input-Stationary achieves the minimum input movement among all four dataflows");
  require(os.output_elements_moved == min_output, "Output-Stationary achieves the minimum output/partial-sum movement among all four dataflows");

  require(rs.weight_elements_moved == ws.weight_elements_moved,
          "Row-Stationary matches Weight-Stationary's (minimal) weight movement");
  require(rs.input_elements_moved == is.input_elements_moved,
          "Row-Stationary matches Input-Stationary's (minimal) input movement -- SIMULTANEOUSLY with the weight result above, RS's real headline property no single-operand-stationary dataflow achieves");

  // Aligned case: utilization should be exactly 1.0 (512 is an exact
  // multiple of 128, no tile padding waste).
  require(std::abs(ws.pe_utilization - 1.0) < 1e-9, "PE utilization is exactly 1.0 when GEMM dimensions divide evenly into the PE array (no padding waste)");

  // Misaligned case: M=513 doesn't divide evenly by 128 -- connects
  // directly to tpu_engine/mxu_opt's own utilization-cliff finding.
  GemmWorkload wl_misaligned{513, 512, 512};
  auto misaligned = analyze_dataflow(Dataflow::WeightStationary, wl_misaligned, pe, k_tile);
  std::printf("\n  misaligned M=513 (not a multiple of 128): pe_utilization=%.4f\n", misaligned.pe_utilization);
  require(misaligned.pe_utilization < 1.0, "PE utilization drops below 1.0 for GEMM dimensions that don't divide evenly into the PE array -- the same utilization-cliff phenomenon tpu_engine/mxu_opt measured on real TPU MXU alignment");

  std::printf("%s\n", g_fails == 0 ? "PASS" : "FAIL");
  return g_fails == 0 ? 0 : 1;
}
