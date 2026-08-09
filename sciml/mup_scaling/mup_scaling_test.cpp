// mup_scaling_test.cpp -- the direct, falsifiable muP claim: sweep a base
// learning rate at several widths, under standard parametrization (SP:
// same LR for input and output layers) and muP (output-layer LR scaled
// down by 1/width). Measure whether the ARGMIN (best-performing) base LR
// shifts across widths under SP but stays fixed under muP.
#include "mup_scaling.h"

#include <cstdio>
#include <random>

using namespace sciml;

namespace {

int g_fails = 0;
void require(bool ok, const char *name) {
  std::printf("%s  %s\n", ok ? "PASS" : "FAIL", name);
  if (!ok) ++g_fails;
}

std::vector<RegressionSample> make_dataset() {
  std::mt19937 rng(1);
  std::uniform_real_distribution<double> d(-2.0, 2.0);
  std::vector<RegressionSample> data;
  for (int i = 0; i < 20; ++i) {
    Vec x = {d(rng), d(rng)};
    double y = std::sin(x[0]) + std::cos(x[1]);
    data.push_back({x, y});
  }
  return data;
}

// Trains at width `n` with the given (lr_input, lr_output) for `iters`
// steps, returns final loss.
double train_and_eval(int d_in, int n, double lr_input, double lr_output, const std::vector<RegressionSample> &data,
                       int iters, uint64_t seed) {
  MupMlpParams p = MupMlpParams::random_init(d_in, n, seed);
  std::vector<double> flat = p.flatten();
  for (int it = 0; it < iters; ++it)
    mup_gd_step(flat, d_in, n, [&](const std::vector<double> &f) { return mse_loss(f, d_in, n, data); }, lr_input,
                lr_output);
  return mse_loss(flat, d_in, n, data);
}

} // namespace

int main() {
  auto data = make_dataset();
  std::vector<int> widths = {4, 8, 16};
  std::vector<double> lr_grid = {0.03, 0.1, 0.3, 1.0, 3.0};
  const int iters = 120;

  std::printf("  final MSE by width and base LR (SP: lr_output=lr_input | muP: lr_output=lr_input/width):\n");
  std::vector<double> sp_best_lr, mup_best_lr;
  int sp_divergences = 0, mup_divergences = 0;
  for (int n : widths) {
    std::printf("  width=%d:\n", n);
    double sp_best_loss = 1e300, mup_best_loss = 1e300;
    double sp_best = 0, mup_best = 0;
    for (double lr : lr_grid) {
      double sp_loss = train_and_eval(2, n, lr, lr, data, iters, /*seed=*/static_cast<uint64_t>(n * 100));
      double mup_loss = train_and_eval(2, n, lr, lr / n, data, iters, /*seed=*/static_cast<uint64_t>(n * 100));
      std::printf("    lr=%.3f  SP loss=%.5f  muP loss=%.5f\n", lr, sp_loss, mup_loss);
      if (sp_loss > 1e6) ++sp_divergences;
      if (mup_loss > 1e6) ++mup_divergences;
      if (sp_loss < sp_best_loss) { sp_best_loss = sp_loss; sp_best = lr; }
      if (mup_loss < mup_best_loss) { mup_best_loss = mup_loss; mup_best = lr; }
    }
    std::printf("    -> best LR: SP=%.3f (loss=%.5f)  muP=%.3f (loss=%.5f)\n", sp_best, sp_best_loss, mup_best,
                mup_best_loss);
    sp_best_lr.push_back(sp_best);
    mup_best_lr.push_back(mup_best);
  }
  int total_points = static_cast<int>(widths.size() * lr_grid.size());
  std::printf("\n  divergence count (loss > 1e6) out of %d (width,lr) points: SP=%d, muP=%d\n", total_points,
              sp_divergences, mup_divergences);
  require(sp_divergences > mup_divergences, "SP diverges at more (width, LR) points than muP does -- muP's output-layer LR scaling also gives real training stability at high LR, not just LR-transfer");

  bool sp_shifts = false;
  for (size_t i = 1; i < sp_best_lr.size(); ++i)
    if (sp_best_lr[i] != sp_best_lr[0]) sp_shifts = true;
  bool mup_stays = true;
  for (size_t i = 1; i < mup_best_lr.size(); ++i)
    if (mup_best_lr[i] != mup_best_lr[0]) mup_stays = false;

  std::printf("\n  best LR across widths %d/%d/%d: SP=[%.3f,%.3f,%.3f] | muP=[%.3f,%.3f,%.3f]\n", widths[0],
              widths[1], widths[2], sp_best_lr[0], sp_best_lr[1], sp_best_lr[2], mup_best_lr[0], mup_best_lr[1],
              mup_best_lr[2]);
  std::printf("  SP best LR %s across widths | muP best LR %s across widths\n", sp_shifts ? "SHIFTS" : "stays fixed",
              mup_stays ? "stays fixed" : "SHIFTS");
  require(sp_shifts, "standard parametrization's best LR SHIFTS across widths -- hyperparameters tuned at one width do NOT transfer under SP, measured directly");
  require(mup_stays, "muP's best LR stays EXACTLY fixed across all widths tested -- the specific, falsifiable claim muP makes, confirmed by direct measurement, not assumed");

  std::printf("%s\n", g_fails == 0 ? "PASS" : "FAIL");
  return g_fails == 0 ? 0 : 1;
}
