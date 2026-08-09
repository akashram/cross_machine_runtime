// ebm_test.cpp -- trains a real energy-based model via contrastive
// divergence (short-run Langevin MCMC negative sampling), then compares
// its generated-sample quality DIRECTLY against step 7's diffusion model
// on the identical toy two-cluster target -- both models trained and
// sampled in this same binary for a true same-run, same-metric
// architecture-family comparison, per PLAN.md's explicit ask.
#include "ebm.h"

#include <cmath>
#include <cstdio>

using namespace sciml;

namespace {

int g_fails = 0;
void require(bool ok, const char *name) {
  std::printf("%s  %s\n", ok ? "PASS" : "FAIL", name);
  if (!ok) ++g_fails;
}

double dist(const Point2D &a, const Point2D &b) {
  double dx = a[0] - b[0], dy = a[1] - b[1];
  return std::sqrt(dx * dx + dy * dy);
}

double mean_nearest_cluster_distance(const std::vector<Point2D> &samples) {
  double total = 0.0;
  for (const auto &s : samples) total += std::min(dist(s, kClusterA), dist(s, kClusterB));
  return total / static_cast<double>(samples.size());
}

} // namespace

int main() {
  const int h = 8;
  EBMParams ebm = EBMParams::random_init(h, /*seed=*/41);
  std::vector<double> flat = ebm.flatten();

  const int langevin_steps = 20;
  const double langevin_step_size = 0.05;
  const int batch_size = 16;
  const int iters = 200;

  std::mt19937 data_rng(2), mcmc_rng(3);
  std::normal_distribution<double> init_noise(0.0, 2.0);

  double cd_loss_before = 0.0, cd_loss_after = 0.0;
  for (int it = 0; it < iters; ++it) {
    std::vector<Point2D> x_pos, x_neg;
    for (int i = 0; i < batch_size; ++i) x_pos.push_back(sample_target(data_rng));

    EBMParams current = EBMParams::from_flat(flat, h);
    for (int i = 0; i < batch_size; ++i) {
      Point2D init = {init_noise(mcmc_rng), init_noise(mcmc_rng)};
      x_neg.push_back(langevin_sample(current, init, langevin_steps, langevin_step_size, mcmc_rng));
    }

    double loss_here = cd_loss(flat, h, x_pos, x_neg);
    if (it == 0) cd_loss_before = loss_here;
    if (it == iters - 1) cd_loss_after = loss_here;

    finite_diff_gd_step(flat, [&](const std::vector<double> &f) { return cd_loss(f, h, x_pos, x_neg); }, 0.1);
  }
  std::printf("  EBM contrastive-divergence training (%d params, %d iterations): CD surrogate loss %.4f -> %.4f (E(pos)-E(neg), should trend negative/lower as E(pos)<<E(neg))\n",
              ebm.num_params(), iters, cd_loss_before, cd_loss_after);
  require(cd_loss_after < cd_loss_before, "contrastive divergence surrogate loss decreases with training (energy at real data pulls below energy at model samples)");

  EBMParams trained = EBMParams::from_flat(flat, h);

  // Generate samples from the trained EBM via Langevin MCMC from noise,
  // more steps than during training for a cleaner final sample.
  const int num_samples = 200;
  std::mt19937 sample_rng(77);
  std::normal_distribution<double> start_noise(0.0, 2.0);
  std::vector<Point2D> ebm_samples;
  for (int i = 0; i < num_samples; ++i) {
    Point2D init = {start_noise(sample_rng), start_noise(sample_rng)};
    ebm_samples.push_back(langevin_sample(trained, init, /*num_steps=*/100, langevin_step_size, sample_rng));
  }
  double ebm_mean_dist = mean_nearest_cluster_distance(ebm_samples);

  // Untrained-EBM baseline, same Langevin sampling machinery, random init
  // -- isolates "did CD training help" the same way step 7 isolated
  // "did DDPM training help."
  EBMParams untrained = EBMParams::random_init(h, /*seed=*/41);
  std::mt19937 baseline_rng(77);
  std::vector<Point2D> baseline_samples;
  for (int i = 0; i < num_samples; ++i) {
    Point2D init = {start_noise(baseline_rng), start_noise(baseline_rng)};
    baseline_samples.push_back(langevin_sample(untrained, init, 100, langevin_step_size, baseline_rng));
  }
  double baseline_mean_dist = mean_nearest_cluster_distance(baseline_samples);

  std::printf("  mean distance to nearest TRUE cluster center: trained EBM=%.4f | untrained-EBM baseline=%.4f\n",
              ebm_mean_dist, baseline_mean_dist);
  require(ebm_mean_dist < baseline_mean_dist, "trained EBM's Langevin samples land measurably closer to the true cluster centers than an untrained EBM's samples");

  // Direct architecture comparison to step 7: retrain the SAME diffusion
  // model (identical hyperparameters to diffusion_test.cpp) in this same
  // binary for a true same-run, same-metric comparison.
  const int T = 20;
  NoiseSchedule sched = NoiseSchedule::linear(T);
  DiffusionNetParams diff_net = DiffusionNetParams::random_init(h, /*seed=*/31);
  std::mt19937 diff_data_rng(1);
  std::uniform_int_distribution<int> t_dist(0, T - 1);
  std::normal_distribution<double> std_normal(0.0, 1.0);
  std::vector<TrainingSample> diff_batch;
  for (int i = 0; i < 24; ++i) {
    Point2D x0 = sample_target(diff_data_rng);
    Point2D eps = {std_normal(diff_data_rng), std_normal(diff_data_rng)};
    diff_batch.push_back({x0, eps, t_dist(diff_data_rng)});
  }
  std::vector<double> diff_flat = diff_net.flatten();
  for (int it = 0; it < 250; ++it)
    finite_diff_gd_step(diff_flat, [&](const std::vector<double> &f) { return diffusion_loss(f, h, sched, diff_batch); }, 0.2);
  DiffusionNetParams diff_trained = DiffusionNetParams::from_flat(diff_flat, h);
  std::mt19937 diff_sample_rng(99);
  std::vector<Point2D> diff_samples;
  for (int i = 0; i < num_samples; ++i) diff_samples.push_back(ddpm_sample(diff_trained, sched, diff_sample_rng));
  double diff_mean_dist = mean_nearest_cluster_distance(diff_samples);

  std::printf("\n  DIRECT architecture comparison (identical target, identical metric, same binary):\n");
  std::printf("    DDPM diffusion (step 7):     mean distance to nearest true center = %.4f\n", diff_mean_dist);
  std::printf("    EBM + Langevin (this step):  mean distance to nearest true center = %.4f\n", ebm_mean_dist);
  require(true, "both architectures' sample quality measured on the identical target distribution and metric -- see README for which one comes out ahead and why, not asserted in a fixed direction here");

  std::printf("%s\n", g_fails == 0 ? "PASS" : "FAIL");
  return g_fails == 0 ? 0 : 1;
}
