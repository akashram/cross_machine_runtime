// Verifies: (1) squeezing produces exact closed-form quadrature
// variances and stays a minimum-uncertainty state (Vx*Vp=1 exactly);
// (2) squeezed-vacuum mean photon number matches the closed-form
// sinh^2(r) result; (3) both gate types are genuinely symplectic
// (S*Omega*S^T=Omega, checked directly, not assumed from the formula);
// (4) a beamsplitter conserves total photon number (a real physical
// invariant of a passive linear-optical element); (5) a small
// Gaussian-Boson-Sampling-style circuit (multiple squeezed modes through
// a small interferometer of beamsplitters) also conserves total photon
// number; (6) homodyne sampling's empirical mean/variance match the
// state's own marginal statistics.
#include "cv_photonic.h"

#include <cmath>
#include <cstdio>

using namespace quantum::cv;

namespace {

int g_fails = 0;
void require(bool ok, const char *name) {
  std::printf("%s  %s\n", ok ? "PASS" : "FAIL", name);
  if (!ok) ++g_fails;
}

void test_squeezing_closed_form() {
  double r = 0.6;
  GaussianState s = vacuum_state(1);
  squeeze(s, 0, r);
  double vx = s.cov[0][0], vp = s.cov[1][1];
  double expected_vx = std::exp(-2.0 * r), expected_vp = std::exp(2.0 * r);
  std::printf("  r=%.2f: Vx=%.6f (expected %.6f)  Vp=%.6f (expected %.6f)  Vx*Vp=%.9f\n", r, vx, expected_vx, vp,
              expected_vp, vx * vp);
  bool ok = std::abs(vx - expected_vx) < 1e-9 && std::abs(vp - expected_vp) < 1e-9 && std::abs(vx * vp - 1.0) < 1e-9;
  require(ok, "single-mode squeezing produces exact closed-form Vx=e^-2r, Vp=e^2r and stays minimum-uncertainty (Vx*Vp=1 exactly)");
}

void test_squeezed_photon_number_closed_form() {
  bool all_ok = true;
  for (double r : {0.2, 0.5, 1.0, 1.5}) {
    GaussianState s = vacuum_state(1);
    squeeze(s, 0, r);
    double n_measured = mean_photon_number(s, 0);
    double n_closed_form = std::sinh(r) * std::sinh(r);
    std::printf("  r=%.1f: mean_photon_number=%.6f closed-form sinh^2(r)=%.6f\n", r, n_measured, n_closed_form);
    all_ok = all_ok && std::abs(n_measured - n_closed_form) < 1e-9;
  }
  require(all_ok, "squeezed-vacuum mean photon number matches the closed-form sinh^2(r) result exactly, for r=0.2..1.5");
}

void test_gates_are_symplectic() {
  bool all_ok = true;
  for (double r : {-0.8, -0.2, 0.3, 1.1}) all_ok = all_ok && is_symplectic(squeeze_local_matrix(r));
  for (double theta : {0.1, 0.7, 1.3, 2.5}) all_ok = all_ok && is_symplectic(beamsplitter_local_matrix(theta));
  require(all_ok, "squeeze_local_matrix(r) and beamsplitter_local_matrix(theta) satisfy S*Omega*S^T=Omega exactly, for several r/theta values");
}

void test_beamsplitter_conserves_photon_number() {
  GaussianState s = vacuum_state(2);
  squeeze(s, 0, 0.4);
  squeeze(s, 1, 0.9);
  double before = total_photon_number(s);
  beamsplitter(s, 0, 1, 0.61);
  double after = total_photon_number(s);
  std::printf("  total photon number before beamsplitter=%.9f after=%.9f\n", before, after);
  require(std::abs(before - after) < 1e-9, "a beamsplitter (a passive, lossless linear-optical element) conserves TOTAL photon number exactly");
}

void test_gbs_style_interferometer_conserves_photon_number() {
  // 3 squeezed modes through a small interferometer (2 beamsplitters) --
  // the state-preparation + linear-optics half of a Gaussian Boson
  // Sampling circuit (Hamilton et al. 2017). Full hafnian-based
  // click-pattern SAMPLING is out of scope here; see README.
  GaussianState s = vacuum_state(3);
  squeeze(s, 0, 0.3);
  squeeze(s, 1, 0.5);
  squeeze(s, 2, 0.7);
  double before = total_photon_number(s);
  double closed_form_before = std::sinh(0.3) * std::sinh(0.3) + std::sinh(0.5) * std::sinh(0.5) + std::sinh(0.7) * std::sinh(0.7);
  beamsplitter(s, 0, 1, 0.4);
  beamsplitter(s, 1, 2, 0.9);
  double after = total_photon_number(s);
  std::printf("  3-mode GBS-style circuit: total photon number before interferometer=%.9f (closed-form sum of sinh^2=%.9f) after=%.9f\n",
              before, closed_form_before, after);
  bool ok = std::abs(before - closed_form_before) < 1e-9 && std::abs(before - after) < 1e-9;
  require(ok, "a small GBS-style circuit (3 squeezed modes through a 2-beamsplitter interferometer) conserves total photon number through the interferometer, matching the closed-form pre-interferometer total exactly");
}

void test_homodyne_sampling_matches_marginal_stats() {
  GaussianState s = vacuum_state(1);
  squeeze(s, 0, 0.5);
  displace(s, 0, 1.2, -0.4);

  std::mt19937_64 rng(17);
  int trials = 200000;
  double sum = 0.0, sum_sq = 0.0;
  for (int t = 0; t < trials; ++t) {
    double sample = homodyne_sample(s, 0, 'x', rng);
    sum += sample;
    sum_sq += sample * sample;
  }
  double empirical_mean = sum / trials;
  double empirical_var = sum_sq / trials - empirical_mean * empirical_mean;
  double true_mean = s.mean[0], true_var = s.cov[0][0];
  std::printf("  homodyne x-samples (n=%d): empirical mean=%.4f (true=%.4f) empirical var=%.4f (true=%.4f)\n", trials,
              empirical_mean, true_mean, empirical_var, true_var);
  bool ok = std::abs(empirical_mean - true_mean) < 0.02 && std::abs(empirical_var - true_var) < 0.02;
  require(ok, "homodyne sampling's empirical mean/variance over 200k trials match the state's own marginal mean/variance closely");
}

}  // namespace

int main() {
  test_squeezing_closed_form();
  test_squeezed_photon_number_closed_form();
  test_gates_are_symplectic();
  test_beamsplitter_conserves_photon_number();
  test_gbs_style_interferometer_conserves_photon_number();
  test_homodyne_sampling_matches_marginal_stats();
  std::printf("%s\n", g_fails == 0 ? "PASS" : "FAIL");
  return g_fails == 0 ? 0 : 1;
}
