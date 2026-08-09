// diffusion_test.cpp -- trains a real DDPM noise-prediction network via
// finite-difference gradient descent (reusing step 6's generic trainer)
// on a toy two-cluster 2D distribution, then measures real sample
// quality: how close generated samples land to the TRUE cluster centers,
// compared against an untrained-network baseline.
#include "diffusion.h"

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

// Mean distance from each sample to its NEAREST true cluster center.
double mean_nearest_cluster_distance(const std::vector<Point2D> &samples) {
  double total = 0.0;
  for (const auto &s : samples) total += std::min(dist(s, kClusterA), dist(s, kClusterB));
  return total / static_cast<double>(samples.size());
}

} // namespace

int main() {
  const int h = 8, T = 20;
  NoiseSchedule sched = NoiseSchedule::linear(T);
  DiffusionNetParams net = DiffusionNetParams::random_init(h, /*seed=*/31);

  // Fixed training batch: (x0, t, eps) triples sampled once.
  std::mt19937 data_rng(1);
  std::uniform_int_distribution<int> t_dist(0, T - 1);
  std::normal_distribution<double> std_normal(0.0, 1.0);
  std::vector<TrainingSample> batch;
  for (int i = 0; i < 24; ++i) {
    Point2D x0 = sample_target(data_rng);
    Point2D eps = {std_normal(data_rng), std_normal(data_rng)};
    int t = t_dist(data_rng);
    batch.push_back({x0, eps, t});
  }

  std::vector<double> flat = net.flatten();
  double loss_before = diffusion_loss(flat, h, sched, batch);
  const int iters = 250;
  for (int it = 0; it < iters; ++it)
    finite_diff_gd_step(flat, [&](const std::vector<double> &f) { return diffusion_loss(f, h, sched, batch); }, 0.2);
  double loss_after = diffusion_loss(flat, h, sched, batch);
  std::printf("  DDPM noise-prediction training (T=%d, %d params, %d finite-diff GD steps): loss %.4f -> %.4f\n", T,
              net.num_params(), iters, loss_before, loss_after);
  require(loss_after < loss_before, "DDPM noise-prediction loss decreases with training");

  DiffusionNetParams trained = DiffusionNetParams::from_flat(flat, h);

  const int num_samples = 200;
  std::mt19937 sample_rng(99);
  std::vector<Point2D> generated;
  for (int i = 0; i < num_samples; ++i) generated.push_back(ddpm_sample(trained, sched, sample_rng));

  // Baseline: samples from the UNTRAINED network (same architecture,
  // random init) -- isolates "did training help" from "does the DDPM
  // sampling machinery itself do anything."
  DiffusionNetParams untrained = DiffusionNetParams::random_init(h, /*seed=*/31);
  std::mt19937 baseline_rng(99);
  std::vector<Point2D> baseline_samples;
  for (int i = 0; i < num_samples; ++i) baseline_samples.push_back(ddpm_sample(untrained, sched, baseline_rng));

  double trained_mean_dist = mean_nearest_cluster_distance(generated);
  double baseline_mean_dist = mean_nearest_cluster_distance(baseline_samples);
  std::printf("  mean distance to nearest TRUE cluster center: trained=%.4f | untrained baseline=%.4f | cluster std=%.2f\n",
              trained_mean_dist, baseline_mean_dist, kClusterStd);
  require(trained_mean_dist < baseline_mean_dist, "trained model's generated samples land measurably closer to the true cluster centers than an untrained network's samples");

  // Cluster-assignment quality: classify each trained sample to its
  // nearest true center, then compare the assigned group's EMPIRICAL
  // mean against that true center directly.
  Point2D mean_a{0, 0}, mean_b{0, 0};
  int count_a = 0, count_b = 0;
  for (const auto &s : generated) {
    if (dist(s, kClusterA) < dist(s, kClusterB)) {
      mean_a[0] += s[0]; mean_a[1] += s[1]; ++count_a;
    } else {
      mean_b[0] += s[0]; mean_b[1] += s[1]; ++count_b;
    }
  }
  if (count_a > 0) { mean_a[0] /= count_a; mean_a[1] /= count_a; }
  if (count_b > 0) { mean_b[0] /= count_b; mean_b[1] /= count_b; }
  double err_a = dist(mean_a, kClusterA);
  double err_b = dist(mean_b, kClusterB);
  std::printf("  cluster A: %d samples, empirical mean=(%.3f,%.3f), true=(%.1f,%.1f), error=%.4f\n", count_a,
              mean_a[0], mean_a[1], kClusterA[0], kClusterA[1], err_a);
  std::printf("  cluster B: %d samples, empirical mean=(%.3f,%.3f), true=(%.1f,%.1f), error=%.4f\n", count_b,
              mean_b[0], mean_b[1], kClusterB[0], kClusterB[1], err_b);
  require(count_a > num_samples / 10 && count_b > num_samples / 10, "generated samples populate BOTH clusters (not mode-collapsed onto one)");
  require(err_a < 1.0 && err_b < 1.0, "each cluster's empirical mean from generated samples lands within 1.0 of the true center (well inside the true inter-cluster distance of 4.0)");

  std::printf("%s\n", g_fails == 0 ? "PASS" : "FAIL");
  return g_fails == 0 ? 0 : 1;
}
