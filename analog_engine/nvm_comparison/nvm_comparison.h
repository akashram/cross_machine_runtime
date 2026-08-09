//===- nvm_comparison.h - RRAM vs. PCM vs. STT-MRAM vs. SRAM-CIM --------===//
//
// PLAN.md Phase 17 step 3: a non-volatile memory device-class comparison
// for use as an analog compute-in-memory substrate. No fab access exists
// to measure any of these devices directly (same disclosed limitation as
// step 1's device model and the rest of this repo's hardware-gated
// phases), so every numeric field below is a literature-informed
// REPRESENTATIVE POINT drawn from a cited order-of-magnitude range in Yu,
// S. (2018), "Neuro-Inspired Computing with Emerging Nonvolatile Memory"
// (Proceedings of the IEEE) -- not a measurement. That distinction is the
// entire point of this step, exactly as PLAN.md asks: "literature-
// grounded, honestly labeled as such."
//
// Four device classes compared on five axes real accelerator-design
// literature uses:
//   - endurance_cycles: rated write-cycle count before wearout
//   - retention_years: how long a written state persists with power off
//     (SRAM-CIM is included specifically because it's the NON-nonvolatile
//     baseline -- its retention is ~0, by construction, which is the
//     whole reason NVM-based compute-in-memory is interesting at all)
//   - write_energy_pj_per_bit: energy to program one bit
//   - relative_density: area efficiency relative to SRAM-CIM's 6T cell
//     (higher = denser = more bits per unit area)
//   - max_analog_levels: how many distinguishable resistance/charge
//     states a single cell can reliably hold -- directly the same
//     "num_levels" axis step 2's crossbar precision sweep measured
//
//===----------------------------------------------------------------------===//
#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <string>
#include <vector>

namespace analog {

struct NvmDeviceClass {
  std::string name;
  double endurance_cycles;          // representative point, cited range in comment at each entry
  double retention_years;
  double write_energy_pj_per_bit;
  double relative_density;          // vs. SRAM-CIM's 6T cell = 1.0
  int max_analog_levels;
  std::string note;
};

inline const std::array<NvmDeviceClass, 4> &nvm_device_classes() {
  static const std::array<NvmDeviceClass, 4> table = {{
      // RRAM (HfOx/TaOx-class): Yu 2018 cites endurance 10^6-10^9 cycles,
      // multi-bit-per-cell demonstrated up to ~4-6 bits (16-64 levels),
      // crossbar-compatible 4F^2 density.
      {"RRAM", /*endurance*/ 1.0e7, /*retention_years*/ 10.0, /*write_energy_pj*/ 10.0,
       /*relative_density*/ 8.0, /*max_analog_levels*/ 32,
       "Crossbar-compatible (4F^2), the ISAAC/PRIME substrate from step 2. Best balance of the four axes for analog compute-in-memory, not best on any single axis."},
      // PCM (phase-change, Ge2Sb2Te5-class): Yu 2018 cites endurance
      // 10^6-10^8 (thermal-cycling-limited, lower than RRAM), excellent
      // retention (commercially proven in Intel Optane), higher write
      // energy than RRAM (SET/RESET require heating past a phase
      // transition), and known conductance DRIFT -- exactly the
      // phenomenon step 1's drift_nu models.
      {"PCM", /*endurance*/ 1.0e7, /*retention_years*/ 10.0, /*write_energy_pj*/ 100.0,
       /*relative_density*/ 8.0, /*max_analog_levels*/ 16,
       "Commercially proven retention (Intel Optane), but write energy ~10x RRAM's (thermal SET/RESET) and known conductance drift -- the failure mode step 1's drift_nu parameter models."},
      // STT-MRAM: Yu 2018 cites endurance far higher than RRAM/PCM
      // (10^12-10^15, effectively unlimited for practical AI-accelerator
      // workloads) and fast, efficient switching, but a hard structural
      // limitation for THIS use case: STT-MRAM's two stable magnetic
      // states make genuine multi-level analog operation far harder than
      // RRAM/PCM's continuous resistance range -- great for a digital
      // NVM/cache replacement, poor for analog compute-in-memory
      // precision (directly the axis step 2 showed matters most).
      {"STT-MRAM", /*endurance*/ 1.0e13, /*retention_years*/ 10.0, /*write_energy_pj*/ 5.0,
       /*relative_density*/ 4.0, /*max_analog_levels*/ 2,
       "Best endurance and write energy of the four by a wide margin, but effectively binary (two stable magnetic states) -- see step 2's finding that precision (analog levels), not endurance, is the accuracy lever."},
      // SRAM-based compute-in-memory: standard CMOS, included as the
      // NON-nonvolatile baseline -- essentially unlimited endurance, the
      // lowest write energy and fastest switching of the four, but
      // VOLATILE (retention ~0 without power) and the lowest density
      // (6T cell), which is exactly why NVM-based analog compute-in-
      // memory is worth the engineering cost in the first place.
      {"SRAM-CIM", /*endurance*/ 1.0e16, /*retention_years*/ 0.0, /*write_energy_pj*/ 0.5,
       /*relative_density*/ 1.0, /*max_analog_levels*/ 16,
       "No wearout, cheapest/fastest writes -- but volatile (retention=0) and lowest density of the four. The baseline NVM-based analog compute-in-memory is being compared against, not a competitor on the same axis."},
  }};
  return table;
}

struct FigureOfMerit {
  std::string name;
  double score; // 0..1, higher is better; see compute_figure_of_merit()'s doc comment for the weighting
};

// A simple, explicitly-illustrative composite score for "suitability as an
// analog compute-in-memory substrate": geometric mean of each axis
// normalized to [0,1] against the best-in-class value across all four
// devices (log-scaled for endurance/write-energy/retention since those
// span orders of magnitude; linear for density and analog levels). This
// is NOT a validated accelerator-design cost model -- it's a documented,
// reproducible way to combine five genuinely different units into one
// ranking, exactly as illustrative as PLAN.md step 3 asks for ("literature
// -grounded... honestly labeled"), not a claim that this specific
// weighting is how a real accelerator team would choose a device.
inline std::vector<FigureOfMerit> compute_figure_of_merit() {
  const auto &table = nvm_device_classes();
  double max_endurance = 0, max_retention = 0, min_write_energy = 1e18, max_density = 0;
  int max_levels = 0;
  for (const auto &d : table) {
    max_endurance = std::max(max_endurance, d.endurance_cycles);
    max_retention = std::max(max_retention, d.retention_years);
    min_write_energy = std::min(min_write_energy, d.write_energy_pj_per_bit);
    max_density = std::max(max_density, d.relative_density);
    max_levels = std::max(max_levels, d.max_analog_levels);
  }

  std::vector<FigureOfMerit> out;
  for (const auto &d : table) {
    // log-normalized endurance (avoid log(0) for anything -- all entries > 0)
    double endurance_norm = std::log10(d.endurance_cycles) / std::log10(max_endurance);
    double retention_norm = max_retention > 0 ? (d.retention_years / max_retention) : 0.0;
    double energy_norm = min_write_energy / d.write_energy_pj_per_bit; // lower energy = better = closer to 1
    double density_norm = d.relative_density / max_density;
    double levels_norm = static_cast<double>(d.max_analog_levels) / static_cast<double>(max_levels);

    // Geometric mean over the five normalized axes. A single zero
    // (SRAM-CIM's retention_norm=0.0) correctly drives the whole score to
    // zero -- volatility is a hard disqualifier for persistent analog
    // weight storage, not something the other four good axes can average
    // away, which is the honest point of including SRAM-CIM at all.
    double product = endurance_norm * retention_norm * energy_norm * density_norm * levels_norm;
    double score = std::pow(std::max(0.0, product), 1.0 / 5.0);
    out.push_back({d.name, score});
  }
  std::sort(out.begin(), out.end(), [](const FigureOfMerit &a, const FigureOfMerit &b) { return a.score > b.score; });
  return out;
}

} // namespace analog
