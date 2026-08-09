// sde_solver_test.cpp -- two real checks:
//   1. Ornstein-Uhlenbeck: WEAK convergence (Monte Carlo mean/variance vs.
//      known closed-form moments) for Euler-Maruyama, PLUS a structural
//      check that Milstein reduces to Euler-Maruyama exactly when the
//      diffusion term is additive (constant in x, so b'=0) -- OU has
//      constant sigma, so this is directly checkable, not assumed.
//   2. Geometric Brownian Motion: STRONG (pathwise) convergence, run
//      against the SAME underlying Brownian path as the closed-form exact
//      solution (GBM has one, in terms of W(t)) -- this is the only way to
//      show Milstein's real advantage over Euler-Maruyama, since GBM has
//      genuinely state-dependent (multiplicative) noise.
#include "sde_solver.h"

#include <cmath>
#include <cstdio>
#include <numeric>

using namespace sciml;

namespace {

int g_fails = 0;
void require(bool ok, const char *name) {
  std::printf("%s  %s\n", ok ? "PASS" : "FAIL", name);
  if (!ok) ++g_fails;
}

void test_ou_weak_convergence_and_milstein_reduction() {
  const double theta = 2.0, mu = 1.0, sigma = 0.5, x0 = 0.0, t1 = 1.0, dt = 0.001;
  DriftFn a = [theta, mu](double, double x) { return theta * (mu - x); };
  DiffusionFn b = [sigma](double, double) { return sigma; }; // additive noise: constant in x

  double closed_mean = mu + (x0 - mu) * std::exp(-theta * t1);
  double closed_var = (sigma * sigma / (2.0 * theta)) * (1.0 - std::exp(-2.0 * theta * t1));

  int num_steps = static_cast<int>(std::round(t1 / dt));
  const int num_paths = 3000;
  std::vector<double> finals;
  double max_em_milstein_diff = 0.0;
  for (int p = 0; p < num_paths; ++p) {
    std::vector<double> z = generate_normal_increments(num_steps, /*seed=*/static_cast<uint64_t>(10000 + p));
    SDETrajectory em = euler_maruyama(a, b, x0, 0.0, t1, dt, z);
    finals.push_back(em.states.back());
    if (p < 50) { // structural check on a subset (Milstein is more expensive; no need for all 3000 paths)
      SDETrajectory mi = milstein(a, b, x0, 0.0, t1, dt, z);
      max_em_milstein_diff = std::max(max_em_milstein_diff, std::abs(em.states.back() - mi.states.back()));
    }
  }
  double mc_mean = std::accumulate(finals.begin(), finals.end(), 0.0) / finals.size();
  double mc_var = 0.0;
  for (double v : finals) mc_var += (v - mc_mean) * (v - mc_mean);
  mc_var /= finals.size();

  std::printf("  OU closed form: mean=%.4f var=%.4f | Monte Carlo (%d paths): mean=%.4f var=%.4f\n", closed_mean,
              closed_var, num_paths, mc_mean, mc_var);
  std::printf("  max |Euler-Maruyama - Milstein| final state over 50 shared-path checks = %.2e (b'=0 for constant sigma)\n",
              max_em_milstein_diff);

  require(std::abs(mc_mean - closed_mean) < 0.02, "Euler-Maruyama's Monte Carlo mean matches OU's closed-form mean");
  require(std::abs(mc_var - closed_var) < 0.01, "Euler-Maruyama's Monte Carlo variance matches OU's closed-form variance");
  require(max_em_milstein_diff < 1e-8, "Milstein reduces to Euler-Maruyama exactly for additive (state-independent) noise, on the SAME Brownian path -- a structural fact, checked directly");
}

void gbm_exact_solution(const std::vector<double> &brownian, const std::vector<double> &times, double mu,
                         double sigma, double x0, std::vector<double> &out) {
  out.resize(times.size());
  for (size_t i = 0; i < times.size(); ++i)
    out[i] = x0 * std::exp((mu - 0.5 * sigma * sigma) * times[i] + sigma * brownian[i]);
}

double gbm_strong_error_rms(bool use_milstein, double dt, int num_paths, double mu, double sigma, double x0,
                             double t1) {
  int num_steps = static_cast<int>(std::round(t1 / dt));
  DriftFn a = [mu](double, double x) { return mu * x; };
  DiffusionFn b = [sigma](double, double x) { return sigma * x; }; // multiplicative noise: b'(x) = sigma, nonzero

  double sq_sum = 0.0;
  for (int p = 0; p < num_paths; ++p) {
    std::vector<double> z = generate_normal_increments(num_steps, /*seed=*/static_cast<uint64_t>(50000 + p));
    SDETrajectory traj = use_milstein ? milstein(a, b, x0, 0.0, t1, dt, z) : euler_maruyama(a, b, x0, 0.0, t1, dt, z);
    std::vector<double> exact;
    gbm_exact_solution(traj.brownian, traj.times, mu, sigma, x0, exact);
    double err = traj.states.back() - exact.back();
    sq_sum += err * err;
  }
  return std::sqrt(sq_sum / num_paths);
}

void test_gbm_strong_convergence() {
  const double mu = 0.05, sigma = 0.3, x0 = 1.0, t1 = 1.0;
  const int num_paths = 2000;

  double dt1 = 0.02, dt2 = 0.01;
  double em_err_dt1 = gbm_strong_error_rms(false, dt1, num_paths, mu, sigma, x0, t1);
  double em_err_dt2 = gbm_strong_error_rms(false, dt2, num_paths, mu, sigma, x0, t1);
  double mi_err_dt1 = gbm_strong_error_rms(true, dt1, num_paths, mu, sigma, x0, t1);
  double mi_err_dt2 = gbm_strong_error_rms(true, dt2, num_paths, mu, sigma, x0, t1);

  std::printf("  GBM strong (pathwise) RMS error vs. exact solution, %d Monte Carlo paths:\n", num_paths);
  std::printf("    dt=%.3f: euler-maruyama=%.4f  milstein=%.4f\n", dt1, em_err_dt1, mi_err_dt1);
  std::printf("    dt=%.3f: euler-maruyama=%.4f  milstein=%.4f\n", dt2, em_err_dt2, mi_err_dt2);
  double em_ratio = em_err_dt1 / std::max(1e-12, em_err_dt2);
  double mi_ratio = mi_err_dt1 / std::max(1e-12, mi_err_dt2);
  std::printf("    error ratio halving dt: euler-maruyama=%.2f (theory: strong order 0.5 -> ~1.41) | milstein=%.2f (theory: strong order 1.0 -> ~2.0)\n",
              em_ratio, mi_ratio);

  require(mi_err_dt1 < em_err_dt1, "at the same (coarser) dt, Milstein's strong error is smaller than Euler-Maruyama's on GBM's genuinely multiplicative noise");
  require(mi_err_dt2 < em_err_dt2, "same comparison holds at the finer dt too");
  require(mi_ratio > em_ratio, "Milstein's error shrinks faster than Euler-Maruyama's when dt halves (higher strong order), measured not assumed");
}

} // namespace

int main() {
  test_ou_weak_convergence_and_milstein_reduction();
  test_gbm_strong_convergence();
  std::printf("%s\n", g_fails == 0 ? "PASS" : "FAIL");
  return g_fails == 0 ? 0 : 1;
}
