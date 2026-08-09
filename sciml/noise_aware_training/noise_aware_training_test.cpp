// noise_aware_training_test.cpp -- mirrors
// adversarial/adversarial_training + adversarial/robustness_tradeoff's
// structure exactly, with the adversary replaced by Phase 17 step 1's
// real device-noise model:
//   1. Train a baseline model normally (clean loss only).
//   2. Train a second model of IDENTICAL architecture with device noise
//      injected into the weights at every training step (a fresh noise
//      draw per step, mirroring how PGD regenerates its perturbation
//      each iteration).
//   3. Evaluate both on CLEAN loss and on NOISY loss (averaged over
//      several independent noise draws, evaluation noise seeds disjoint
//      from every training-time seed) -- does noise-aware training
//      measurably improve robustness, at what clean-accuracy cost?
#include "noise_aware_training.h"

#include <cstdio>

using namespace sciml;

namespace {

int g_fails = 0;
void require(bool ok, const char *name) {
  std::printf("%s  %s\n", ok ? "PASS" : "FAIL", name);
  if (!ok) ++g_fails;
}

double mean_noisy_loss(const std::vector<double> &flat, int d_in, int n, const std::vector<RegressionSample> &data,
                        double w_max, const analog::DeviceParams &device_params, int num_draws,
                        uint64_t base_eval_seed) {
  double total = 0.0;
  for (int d = 0; d < num_draws; ++d)
    total += noise_aware_loss(flat, d_in, n, data, w_max, device_params,
                               base_eval_seed + static_cast<uint64_t>(d) * 100003u);
  return total / static_cast<double>(num_draws);
}

} // namespace

int main() {
  const int d_in = 2, n = 8;
  const double w_max = 2.0;
  const int iters = 250;

  analog::DeviceParams device_params;
  device_params.num_levels = 8;
  device_params.write_noise_frac_of_level = 0.25;
  device_params.read_noise_frac_of_write = 0.2;

  std::mt19937 data_rng(1);
  std::uniform_real_distribution<double> dd(-2.0, 2.0);
  std::vector<RegressionSample> data;
  for (int i = 0; i < 20; ++i) {
    Vec x = {dd(data_rng), dd(data_rng)};
    data.push_back({x, std::sin(x[0]) + std::cos(x[1])});
  }

  MupMlpParams init = MupMlpParams::random_init(d_in, n, /*seed=*/9);
  std::vector<double> baseline_flat = init.flatten();
  std::vector<double> noise_aware_flat = init.flatten(); // SAME starting point for both

  for (int it = 0; it < iters; ++it) {
    finite_diff_gd_step(baseline_flat, [&](const std::vector<double> &f) { return mse_loss(f, d_in, n, data); }, 0.1);
    uint64_t train_noise_seed = static_cast<uint64_t>(it) * 9973u + 1; // fresh noise draw every step
    noise_aware_gd_step(noise_aware_flat, d_in, n, data, w_max, device_params, train_noise_seed, 0.1);
  }

  double baseline_clean = mse_loss(baseline_flat, d_in, n, data);
  double noise_aware_clean = mse_loss(noise_aware_flat, d_in, n, data);
  const int eval_draws = 30;
  double baseline_noisy = mean_noisy_loss(baseline_flat, d_in, n, data, w_max, device_params, eval_draws, 500000);
  double noise_aware_noisy =
      mean_noisy_loss(noise_aware_flat, d_in, n, data, w_max, device_params, eval_draws, 500000);

  std::printf("  clean loss:  baseline=%.5f  noise-aware=%.5f\n", baseline_clean, noise_aware_clean);
  std::printf("  noisy loss (mean over %d independent device-noise draws, disjoint from training seeds):  baseline=%.5f  noise-aware=%.5f\n",
              eval_draws, baseline_noisy, noise_aware_noisy);

  require(noise_aware_noisy < baseline_noisy,
          "noise-aware training measurably improves robustness: lower mean loss under real device noise than the normally-trained baseline");
  std::printf("  clean-accuracy cost of noise-aware training: %.5f (noise-aware clean loss - baseline clean loss)\n",
              noise_aware_clean - baseline_clean);

  std::printf("%s\n", g_fails == 0 ? "PASS" : "FAIL");
  return g_fails == 0 ? 0 : 1;
}
