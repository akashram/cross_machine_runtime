// ssm_layer_test.cpp -- two real, measured comparisons:
//   1. Accuracy on a long-range-dependency toy task ("copy the first
//      token's value to the output at the final position"), both models
//      trained the SAME way (finite-difference gradient descent).
//   2. MEASURED (instrumented multiply-add counting, not a cited formula)
//      operation-count scaling with sequence length L: O(L) for the SSM's
//      recurrence vs. O(L^2) for attention's pairwise score matrix.
#include "ssm_layer.h"

#include <cstdio>
#include <random>

using namespace sciml;

namespace {

int g_fails = 0;
void require(bool ok, const char *name) {
  std::printf("%s  %s\n", ok ? "PASS" : "FAIL", name);
  if (!ok) ++g_fails;
}

Seq random_seq(int L, int d, std::mt19937 &rng) {
  std::uniform_real_distribution<double> dist(-1.0, 1.0);
  Seq s(static_cast<size_t>(L), Vec(static_cast<size_t>(d)));
  for (auto &v : s)
    for (double &x : v) x = dist(rng);
  return s;
}

double copy_task_loss_ssm(const std::vector<double> &flat, int n, int d, const std::vector<Seq> &batch) {
  SSMParams p = SSMParams::from_flat(flat, n, d);
  double total = 0.0;
  for (const Seq &u : batch) {
    Seq y = ssm_forward(p, u);
    const Vec &y_final = y.back();
    const Vec &target = u.front();
    double s = 0.0;
    for (size_t i = 0; i < target.size(); ++i) {
      double diff = y_final[i] - target[i];
      s += diff * diff;
    }
    total += s;
  }
  return total / static_cast<double>(batch.size());
}

double copy_task_loss_attn(const std::vector<double> &flat, int d, const std::vector<Seq> &batch) {
  AttnParams p = AttnParams::from_flat(flat, d);
  double total = 0.0;
  for (const Seq &u : batch) {
    Seq y = attn_forward(p, u);
    const Vec &y_final = y.back();
    const Vec &target = u.front();
    double s = 0.0;
    for (size_t i = 0; i < target.size(); ++i) {
      double diff = y_final[i] - target[i];
      s += diff * diff;
    }
    total += s;
  }
  return total / static_cast<double>(batch.size());
}

void test_copy_task_accuracy() {
  const int n = 4, d = 3, L = 10, batch_size = 12, iters = 150;
  std::mt19937 data_rng(1);
  std::vector<Seq> batch;
  for (int i = 0; i < batch_size; ++i) batch.push_back(random_seq(L, d, data_rng));

  SSMParams ssm = SSMParams::random_init(n, d, /*seed=*/2);
  AttnParams attn = AttnParams::random_init(d, /*seed=*/3);
  std::vector<double> ssm_flat = ssm.flatten();
  std::vector<double> attn_flat = attn.flatten();

  double ssm_loss_before = copy_task_loss_ssm(ssm_flat, n, d, batch);
  double attn_loss_before = copy_task_loss_attn(attn_flat, d, batch);

  for (int it = 0; it < iters; ++it) {
    finite_diff_gd_step(ssm_flat, [&](const std::vector<double> &f) { return copy_task_loss_ssm(f, n, d, batch); }, 0.3);
    finite_diff_gd_step(attn_flat, [&](const std::vector<double> &f) { return copy_task_loss_attn(f, d, batch); }, 0.3);
  }

  double ssm_loss_after = copy_task_loss_ssm(ssm_flat, n, d, batch);
  double attn_loss_after = copy_task_loss_attn(attn_flat, d, batch);

  std::printf("  copy-first-to-last task, L=%d, %d params trained %d steps (finite-diff GD, same recipe for both):\n", L,
              iters, iters);
  std::printf("    SSM (generic, non-HiPPO):  loss %.4f -> %.4f\n", ssm_loss_before, ssm_loss_after);
  std::printf("    Attention (single-head):   loss %.4f -> %.4f\n", attn_loss_before, attn_loss_after);

  require(ssm_loss_after < ssm_loss_before, "SSM's loss decreases with training on the copy task");
  require(attn_loss_after < attn_loss_before, "Attention's loss decreases with training on the copy task");
  require(attn_loss_after < ssm_loss_after,
          "Attention reaches a LOWER final loss than the generic (non-HiPPO) SSM on this long-range copy task -- see README for why this is the expected, literature-consistent result, not a bug");
}

void test_op_count_scaling() {
  const int n = 4, d = 3;
  std::vector<int> lengths = {8, 16, 32, 64, 128};
  SSMParams ssm = SSMParams::random_init(n, d, /*seed=*/5);
  AttnParams attn = AttnParams::random_init(d, /*seed=*/6);

  std::printf("  op-count scaling with sequence length L (instrumented multiply-add counter, not a formula):\n");
  std::vector<long> ssm_ops, attn_ops;
  for (int L : lengths) {
    std::mt19937 rng(static_cast<unsigned>(100 + L));
    Seq u = random_seq(L, d, rng);

    reset_op_counter();
    ssm_forward(ssm, u);
    ssm_ops.push_back(op_counter());

    reset_op_counter();
    attn_forward(attn, u);
    attn_ops.push_back(op_counter());

    std::printf("    L=%4d: SSM ops=%8ld   Attention ops=%8ld\n", L, ssm_ops.back(), attn_ops.back());
  }

  // SSM: doubling L should roughly double op count (linear). Attention:
  // doubling L should roughly QUADRUPLE op count (quadratic).
  double ssm_ratio_last = static_cast<double>(ssm_ops.back()) / static_cast<double>(ssm_ops[ssm_ops.size() - 2]);
  double attn_ratio_last = static_cast<double>(attn_ops.back()) / static_cast<double>(attn_ops[attn_ops.size() - 2]);
  std::printf("    doubling L (64->128): SSM op-count ratio=%.2f (expect ~2, linear) | Attention op-count ratio=%.2f (expect ~4, quadratic)\n",
              ssm_ratio_last, attn_ratio_last);

  require(ssm_ratio_last > 1.7 && ssm_ratio_last < 2.3, "SSM's measured op count scales linearly with L (ratio ~2 when L doubles)");
  require(attn_ratio_last > 3.5 && attn_ratio_last < 4.5, "Attention's measured op count scales quadratically with L (ratio ~4 when L doubles)");
  require(attn_ops.back() > ssm_ops.back(), "at the largest L tested, attention's op count exceeds the SSM's -- the crossover the O(L) vs O(L^2) difference predicts");
}

} // namespace

int main() {
  test_copy_task_accuracy();
  test_op_count_scaling();
  std::printf("%s\n", g_fails == 0 ? "PASS" : "FAIL");
  return g_fails == 0 ? 0 : 1;
}
