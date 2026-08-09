//===- device_model.h - RRAM-like conductance-cell non-ideality model ---===//
//
// PLAN.md Phase 17 step 1: a parameterized model of the non-idealities a
// real resistive (RRAM/memristor-class) memory cell exhibits when used as
// an analog weight/multiply element -- the device physics step 2's
// crossbar simulation injects, and step 3's NVM comparison table is
// grounded in.
//
// No RRAM hardware exists to characterize directly (no fab access, same
// disclosed limitation as the rest of this repo's hardware-gated phases),
// so the noise/drift/endurance parameters below are literature-informed
// order-of-magnitude values (see Yu, S. (2018), "Neuro-Inspired Computing
// with Emerging Nonvolatile Memory", Proceedings of the IEEE -- cited
// directly in READING_LIST.md's Phase 17 step 1 entry), not measurements
// of a real device. That distinction is the whole point of this step:
// model the KNOWN qualitative behaviors (cycle-to-cycle write noise,
// smaller read noise, conductance drift, endurance-linked degradation)
// with plausible magnitudes, honestly labeled, rather than pretend a
// digital-precision abstraction is what analog compute-in-memory hardware
// actually provides.
//
// Four non-idealities modeled, each independently toggleable:
//   1. Write noise: programming a cell to a target conductance level lands
//      on that level plus Gaussian noise (cycle-to-cycle variability).
//      This is the dominant source of MAC error in a real crossbar --
//      step 2 measures its effect on matrix-vector-multiply accuracy
//      directly.
//   2. Read noise: reading a cell's conductance adds a SMALLER Gaussian
//      perturbation than write noise (thermal/measurement noise, not
//      stochastic switching) -- modeled as strictly smaller, verified by
//      the test.
//   3. Conductance drift: conductance decays as a power law in time since
//      last write, G(t) = G_written * (t / t_ref) ^ (-nu), the standard
//      empirical form used for phase-change and resistive memory drift
//      (Yu 2018, Sec. III).
//   4. Endurance-linked degradation: past a device's rated write-cycle
//      endurance, the probability a cell becomes "stuck" (unable to
//      change conductance on the next write) rises from ~0 to
//      non-negligible -- modeled as a logistic ramp centered on the rated
//      endurance, not a hard cliff (real endurance failure is
//      probabilistic across a population of cells, not a fixed cutoff).
//
//===----------------------------------------------------------------------===//
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <random>
#include <vector>

namespace analog {

struct DeviceParams {
  double g_min = 1.0;   // minimum conductance (arbitrary units, e.g. uS)
  double g_max = 100.0; // maximum conductance
  int num_levels = 16;  // discrete conductance levels (multi-level cell, MLC)

  // Cycle-to-cycle write noise, as a fraction of one level's spacing
  // (g_max - g_min) / (num_levels - 1). Literature-informed order of
  // magnitude for HfOx-class RRAM (Yu 2018): a few percent of full range
  // per level at moderate MLC density.
  double write_noise_frac_of_level = 0.25;

  // Read noise as a fraction of WRITE noise -- read disturbs the cell far
  // less than programming it (measurement/thermal noise vs. stochastic
  // filament switching), so this must stay < 1.0.
  double read_noise_frac_of_write = 0.2;

  // Conductance drift exponent nu in G(t) = G_written * (t/t_ref)^(-nu).
  // Yu 2018 cites nu ~ 0.01-0.13 for PCM-class drift; RRAM drift is
  // typically smaller but present. t_ref is the reference time (same
  // units as the `elapsed` argument to read()) at which no drift has yet
  // occurred (t < t_ref treated as t_ref, avoiding a division blowup).
  double drift_nu = 0.05;
  double drift_t_ref = 1.0;

  // Endurance: rated write-cycle count (order of magnitude for RRAM,
  // Yu 2018: 10^6-10^9 depending on material system). stuck_sharpness
  // controls how quickly the stuck-probability ramps from ~0 to ~1 around
  // rated_endurance_cycles (logistic ramp, not a hard cliff).
  double rated_endurance_cycles = 1.0e6;
  double stuck_sharpness = 4.0; // logistic steepness in decades of cycle count
};

// One simulated resistive cell. Tracks its actual conductance, write-cycle
// count, and whether it has become "stuck" (endurance failure).
class ConductanceCell {
public:
  explicit ConductanceCell(DeviceParams params, uint64_t seed = 0)
      : params_(params), rng_(seed), conductance_(params.g_min), write_count_(0), stuck_(false) {
    // One failure-threshold percentile drawn per cell at construction, not
    // re-rolled per write. stuck_probability(write_count_) is a CUMULATIVE
    // failure curve (fraction of a cell population failed BY that cycle
    // count) -- crossing this cell's fixed percentile is what "this
    // specific cell has failed" means. Rolling a fresh Bernoulli check on
    // every write instead (the first version of this code did that) is a
    // real bug: it turns a single population-level failure probability
    // into dozens of near-independent chances to fail before reaching
    // that cycle count, compounding into a wildly inflated stuck rate even
    // for cells well below their rated endurance (caught by
    // device_model_test's "far below rated endurance -> almost never
    // stuck" check reporting 50% instead of <5%).
    std::uniform_real_distribution<double> u(0.0, 1.0);
    failure_threshold_ = u(rng_);
  }

  double level_step() const { return (params_.g_max - params_.g_min) / std::max(1, params_.num_levels - 1); }

  // Map an integer level index [0, num_levels-1] to its ideal conductance.
  double ideal_conductance_for_level(int level) const {
    level = std::clamp(level, 0, params_.num_levels - 1);
    return params_.g_min + level * level_step();
  }

  // Program the cell toward `level`. Applies write noise and endurance
  // effects; a stuck cell silently ignores the write (its conductance
  // stays wherever it was when it stuck -- the real failure mode: cells
  // don't announce they're stuck, they just stop responding).
  void write(int level) {
    ++write_count_;
    if (!stuck_ && stuck_probability(write_count_) > failure_threshold_) stuck_ = true;
    if (stuck_) return;
    double ideal = ideal_conductance_for_level(level);
    double sigma = params_.write_noise_frac_of_level * level_step();
    std::normal_distribution<double> noise(0.0, std::max(1e-9, sigma));
    conductance_ = std::clamp(ideal + noise(rng_), params_.g_min, params_.g_max);
  }

  // Read the cell's conductance `elapsed` time units after its last write,
  // with drift applied first, then read noise. `rng_` is `mutable` so this
  // can stay a logically-const query (reading doesn't change the cell's
  // programmed state) while still consuming randomness for the noise draw.
  double read(double elapsed) const {
    double t_eff = std::max(elapsed, params_.drift_t_ref);
    double drifted = conductance_ * std::pow(t_eff / params_.drift_t_ref, -params_.drift_nu);
    double sigma = params_.read_noise_frac_of_write * params_.write_noise_frac_of_level * level_step();
    std::normal_distribution<double> noise(0.0, std::max(1e-9, sigma));
    return std::clamp(drifted + noise(rng_), 0.0, params_.g_max * 1.5);
  }

  bool is_stuck() const { return stuck_; }
  uint64_t write_count() const { return write_count_; }
  double raw_conductance() const { return conductance_; }

private:
  // Cumulative logistic failure curve: fraction of a cell POPULATION
  // expected to have failed by `write_count` cycles, centered (in log10
  // cycle count) on rated_endurance_cycles. Called once per write with the
  // post-increment write_count_ (always >= 1 at the call site, so no
  // write_count==0 special case is needed).
  double stuck_probability(uint64_t write_count) const {
    double log_ratio = std::log10(static_cast<double>(write_count) / params_.rated_endurance_cycles);
    double x = params_.stuck_sharpness * log_ratio;
    return 1.0 / (1.0 + std::exp(-x));
  }

  DeviceParams params_;
  mutable std::mt19937_64 rng_;
  double conductance_;
  uint64_t write_count_;
  bool stuck_;
  double failure_threshold_;
};

} // namespace analog
