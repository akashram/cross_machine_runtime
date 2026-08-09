//===- noise_aware_training.h - training through Phase 17's device model =//
//
// PLAN.md Phase 18 step 10 (this phase's last step): a physics-informed
// / noise-aware training bridge into Phase 17's real device-noise model
// (`analog::ConductanceCell`, step 1) -- structurally mirrors Phase 14's
// adversarial-training loop (`adversarial/adversarial_training`), with
// the adversary replaced by device write/read noise instead of a
// gradient-based attacker.
//
// Reuses step 9's `mup_scaling.h` MLP and its "sin(x0)+cos(x1)"
// regression task directly, rather than a fourth new trainable model --
// this step is about the TRAINING PROCEDURE (perturb weights with real
// device noise during training, the way adversarial training perturbs
// inputs), not about a new architecture.
//
// Weight-to-conductance-cell mapping: each flattened parameter w (in a
// bounded range [-w_max, w_max], matching this repo's small-MLP init
// scales) is mapped to a discrete level, written to a
// `analog::ConductanceCell`, read back with step 1's write/read noise,
// and decoded back to a float in the same range -- a single-cell
// (signed-range, not differential-pair) round trip, since this step
// needs a realistic NOISY weight value for training robustness, not a
// literal crossbar matrix-vector product the way step 2/7 needed.
//
//===----------------------------------------------------------------------===//
#pragma once

#include "../mup_scaling/mup_scaling.h"
#include "../../analog_engine/device_model/device_model.h"

#include <algorithm>
#include <cmath>

namespace sciml {

// Round-trips one weight value through a real (simulated) analog cell:
// quantize w in [-w_max, w_max] to a level, write (noise), read (noise),
// decode back to a float in the same range.
inline double noisy_weight(double w, double w_max, const analog::DeviceParams &device_params, uint64_t seed) {
  double clamped = std::clamp(w, -w_max, w_max);
  int level = static_cast<int>(std::round((clamped + w_max) / (2.0 * w_max) * (device_params.num_levels - 1)));
  level = std::clamp(level, 0, device_params.num_levels - 1);

  analog::ConductanceCell cell(device_params, seed);
  cell.write(level);
  double measured = cell.read(0.0);
  double step = cell.level_step();
  int decoded = static_cast<int>(std::round((measured - device_params.g_min) / step));
  decoded = std::clamp(decoded, 0, device_params.num_levels - 1);

  return (static_cast<double>(decoded) / (device_params.num_levels - 1)) * (2.0 * w_max) - w_max;
}

// Maps every flattened parameter through noisy_weight(), each with its
// OWN cell (seed derived from position, so the noise draw is
// deterministic given a base seed but different per parameter).
inline std::vector<double> noisy_flat(const std::vector<double> &flat, double w_max,
                                       const analog::DeviceParams &device_params, uint64_t base_seed) {
  std::vector<double> out(flat.size());
  for (size_t i = 0; i < flat.size(); ++i)
    out[i] = noisy_weight(flat[i], w_max, device_params, base_seed + static_cast<uint64_t>(i) * 7919u);
  return out;
}

// Noise-aware training loss: the SAME mse_loss as step 9, but evaluated
// on the NOISE-INJECTED weights (flat -> noisy_flat -> loss), so a
// finite-difference gradient step on `flat` learns weights that are good
// AFTER a real device write/read round trip, not just in their
// noiseless digital form -- exactly what adversarial training does for
// input perturbations, here for device noise instead.
inline double noise_aware_loss(const std::vector<double> &flat, int d_in, int n,
                                const std::vector<RegressionSample> &data, double w_max,
                                const analog::DeviceParams &device_params, uint64_t noise_seed) {
  std::vector<double> perturbed = noisy_flat(flat, w_max, device_params, noise_seed);
  return mse_loss(perturbed, d_in, n, data);
}

// REAL BUG, caught by running the first version of this step's test, not
// by inspection: naively finite-differencing `noise_aware_loss` w.r.t.
// the CLEAN parameters (perturb flat[k] by +-1e-4, re-quantize, re-noise,
// evaluate) gives a degenerate gradient. The quantization level step at
// num_levels=16 over [-2,2] is ~0.267 -- five orders of magnitude larger
// than the 1e-4 finite-difference epsilon -- so almost every perturbation
// lands on the exact SAME discrete level (gradient contribution exactly
// 0.0, confirmed directly: `noisy_weight(w)==noisy_weight(w+-1e-4)`
// bit-for-bit in the overwhelming majority of cases), except at the rare
// perturbation that happens to straddle a level boundary, which produces
// a spurious gradient spike of order (level_step / 2*eps) ~ 1335 -- far
// larger than any real underlying gradient. Training on that pathological
// gradient diverged catastrophically (clean loss exploded to ~7700 from
// a ~0.035 starting point).
//
// This is not a bug specific to this repo's finite-difference training
// convention -- it is EXACTLY why real quantization-aware training (QAT)
// uses a Straight-Through Estimator (STE) instead of differentiating
// through round()/quantize() directly: treat the quantization+noise
// mapping as the IDENTITY function for gradient purposes. Concretely:
// compute the loss gradient w.r.t. the ALREADY-NOISY weight values
// (smooth, well-behaved -- no discrete step in that function), then apply
// that gradient directly to the CLEAN parameters. `noise_aware_gd_step`
// below implements exactly this, fixing the divergence (see
// README.md's Results/Findings for the before/after numbers).
inline void noise_aware_gd_step(std::vector<double> &clean_flat, int d_in, int n,
                                 const std::vector<RegressionSample> &data, double w_max,
                                 const analog::DeviceParams &device_params, uint64_t step_noise_seed, double lr,
                                 double eps = 1e-4) {
  std::vector<double> noisy = noisy_flat(clean_flat, w_max, device_params, step_noise_seed);

  // Finite-difference gradient of the SMOOTH mse_loss w.r.t. the
  // already-noisy weights -- no quantization inside this function, so no
  // degenerate zero-or-spike gradient.
  std::vector<double> grad(noisy.size(), 0.0);
  for (size_t k = 0; k < noisy.size(); ++k) {
    double orig = noisy[k];
    noisy[k] = orig + eps;
    double loss_plus = mse_loss(noisy, d_in, n, data);
    noisy[k] = orig - eps;
    double loss_minus = mse_loss(noisy, d_in, n, data);
    noisy[k] = orig;
    grad[k] = (loss_plus - loss_minus) / (2.0 * eps);
  }
  // Straight-through: apply the noisy-weight gradient directly to the
  // clean parameters (d(noisy)/d(clean) treated as identity).
  for (size_t k = 0; k < clean_flat.size(); ++k) clean_flat[k] -= lr * grad[k];
}

} // namespace sciml
