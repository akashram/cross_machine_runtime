// systolic_sweep_test.cpp -- sweeps PE array size across transformer/'s
// REAL GEMM shapes and measures the utilization-maximizing configuration
// per shape, extending tpu_engine/mxu_opt's single-shape finding into a
// genuine design-space sweep.
#include "systolic_sweep.h"

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
  std::vector<int> pe_sizes = {8, 16, 32, 64, 128};
  long k_tile = 8; // small K-tile depth, matching this workload's small K dimensions (head_dim=8)
  auto shapes = transformer_real_gemm_shapes();

  std::printf("  transformer/'s real GEMM shapes (d_model=16, num_heads=2, head_dim=8, d_ff=32, seq_len=32, vocab_size=16):\n");
  double sum_util_at_128 = 0.0, sum_util_at_best = 0.0;
  for (const auto &g : shapes) {
    auto sweep = sweep_pe_sizes(g.workload, pe_sizes, k_tile);
    int best = best_pe_size(g.workload, pe_sizes, k_tile);
    std::printf("\n  %s (M=%ld,K=%ld,N=%ld):\n", g.name.c_str(), g.workload.M, g.workload.K, g.workload.N);
    double util_at_128 = 0.0, util_at_best = 0.0;
    for (const auto &pt : sweep) {
      std::printf("    PE=%4dx%-4d  utilization=%.4f%s\n", pt.pe_size, pt.pe_size, pt.pe_utilization,
                  pt.pe_size == best ? "  <- best" : "");
      if (pt.pe_size == 128) util_at_128 = pt.pe_utilization;
      if (pt.pe_size == best) util_at_best = pt.pe_utilization;
    }
    sum_util_at_128 += util_at_128;
    sum_util_at_best += util_at_best;
    require(util_at_best >= util_at_128, "the best-found PE size's utilization is never worse than the largest (128x128) size's");
  }

  double avg_128 = sum_util_at_128 / static_cast<double>(shapes.size());
  double avg_best = sum_util_at_best / static_cast<double>(shapes.size());
  std::printf("\n  averaged across all %zu shapes: utilization at PE=128x128 (a production-scale array) = %.4f | at each shape's best PE size = %.4f\n",
              shapes.size(), avg_128, avg_best);

  require(avg_best > avg_128 * 2.0, "the utilization-maximizing PE size averages more than 2x the utilization a 128x128 production-scale array gets on these small toy-transformer shapes -- a real, measured over-provisioning cost, not assumed");

  // A specific, checkable case: attention scores (32,8,32) at PE=128
  // should be catastrophically underutilized -- direct connection to
  // tpu_engine/layout_opt's own "small-batch padding is catastrophic"
  // finding (0.8% ceiling at batch=1 decode).
  GemmWorkload attn_scores{32, 8, 32};
  auto attn_sweep = sweep_pe_sizes(attn_scores, pe_sizes, k_tile);
  double attn_util_128 = 0.0, attn_util_32 = 0.0;
  for (const auto &pt : attn_sweep) {
    if (pt.pe_size == 128) attn_util_128 = pt.pe_utilization;
    if (pt.pe_size == 32) attn_util_32 = pt.pe_utilization;
  }
  std::printf("\n  attention-scores shape (32x8x32) specifically: PE=128 utilization=%.4f vs. PE=32 utilization=%.4f\n",
              attn_util_128, attn_util_32);
  require(attn_util_128 < 0.10, "attention scores' 32x32 output on a 128x128 array wastes over 90% of the array -- catastrophic underutilization from over-provisioning, echoing tpu_engine/layout_opt's small-batch-padding finding on an independently-built model");
  require(attn_util_32 > 0.99, "the SAME shape on a 32x32 array (matching its own M,N dimensions exactly) achieves ~100% utilization");

  std::printf("%s\n", g_fails == 0 ? "PASS" : "FAIL");
  return g_fails == 0 ? 0 : 1;
}
