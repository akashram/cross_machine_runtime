//===- energy_model.h - analog crossbar MAC energy vs. digital devices --===//
//
// PLAN.md Phase 17 step 4: pJ/MAC for a resistive crossbar vs. this repo's
// EXISTING digital device numbers, extending `npu_engine/cost_model`'s and
// `fpga_engine/clock_gating`'s "portable model" pattern.
//
// Digital pJ/MAC is derived directly from `npu_engine/cost_model/
// npu_cost_model.cpp`'s own TOPS/W constants (CPU/GPU/NPU), not
// re-guessed here -- same devices, same numbers, converted from TOPS/W to
// pJ/MAC (1 MAC = 2 ops, so pJ/MAC = 2 / (TOPS/W)) for direct comparability
// with the analog side's units.
//
// The analog side is where this step's real finding lives: raw analog
// multiply-accumulate on a resistive crossbar is close to free
// (attojoule-to-femtojoule scale per the literature this repo has no way
// to independently verify -- no fab access), but that number is
// routinely cited as MISLEADING in isolation, because reading the
// crossbar's analog current output back into the digital domain needs an
// ADC (analog-to-digital converter), and ADC energy is well documented
// (Yu 2018 and the broader compute-in-memory literature) to dominate a
// real crossbar accelerator's total energy -- often the majority of it.
// ADC energy also grows with the number of bits of precision needed,
// which connects directly to step 2's own finding: step 2 showed
// crossbar precision (num_levels) is what actually drives MAC accuracy.
// This step shows the other side of that same knob: more levels also
// costs more ADC energy, a genuine accuracy/energy tradeoff, not free
// precision.
//
//===----------------------------------------------------------------------===//
#pragma once

#include <cmath>
#include <string>
#include <vector>

namespace analog {

struct DigitalDevice {
  std::string name;
  double tops_per_watt; // from npu_engine/cost_model/npu_cost_model.cpp's kDevices table
};

// Same three devices, same TOPS/W constants as
// npu_engine/cost_model/npu_cost_model.cpp's kDevices table (CPU:
// 8.0 TOPS / 150W = 0.0533; GPU (A100): 312.0/400 = 0.78; NPU (Apple ANE,
// representative): 15.8/2.0 = 7.9). Not re-derived independently -- the
// point is to compare against this repo's OWN existing digital numbers.
inline const std::vector<DigitalDevice> &digital_devices() {
  static const std::vector<DigitalDevice> table = {
      {"CPU (AVX-512 VNNI server)", 8.0 / 150.0},
      {"GPU (A100, dense INT8)", 312.0 / 400.0},
      {"NPU (Apple ANE, repr.)", 15.8 / 2.0},
  };
  return table;
}

inline double digital_pj_per_mac(double tops_per_watt) {
  // TOPS/W = ops per joule (since TOPS = 1e12 ops/sec, W = J/sec, so
  // TOPS/W = 1e12 ops/J). pJ per op = 1e12 pJ/J / (TOPS/W * 1e12 ops/J) =
  // 1 / (TOPS/W). One MAC = 2 ops (multiply + accumulate).
  return 2.0 / tops_per_watt;
}

// Analog crossbar MAC energy, split into the "pure compute" figure often
// quoted in isolation and the "ADC-inclusive" figure that's the realistic
// system-level cost -- both illustrative, literature-order-of-magnitude
// points (Yu 2018), not measurements (no fab access, same disclosed
// limitation as steps 1-3).
struct AnalogEstimate {
  double pure_compute_pj_per_mac; // the crossbar's raw Ohm's-law multiply-accumulate alone
  double adc_energy_per_bit_pj;   // illustrative per-bit-of-resolution ADC conversion cost
};

inline AnalogEstimate analog_estimate() {
  // Pure analog compute: sub-femtojoule to low-femtojoule per MAC is the
  // figure commonly quoted for the crossbar's Ohm's-law multiply-
  // accumulate alone -- 0.005 pJ (5 fJ) here as an illustrative point
  // within that range.
  // ADC energy per bit of resolution: a few hundred fJ per conversion
  // step is a representative order of magnitude for a column ADC shared
  // across a crossbar's outputs (Yu 2018's discussion of peripheral
  // circuit overhead) -- 0.3 pJ/bit here, illustrative, not a specific
  // ADC datasheet number.
  return {0.005, 0.3};
}

// Total pJ/MAC INCLUDING the ADC cost of reading out `num_levels`-worth of
// precision -- log2(num_levels) bits of resolution. This is what connects
// this step to step 2's crossbar_mac precision sweep: more levels (which
// step 2 showed measurably improves MAC accuracy) costs more ADC energy
// here, a real tradeoff rather than free precision.
inline double analog_realistic_pj_per_mac(const AnalogEstimate &a, int num_levels) {
  double bits = std::log2(static_cast<double>(std::max(2, num_levels)));
  return a.pure_compute_pj_per_mac + bits * a.adc_energy_per_bit_pj;
}

} // namespace analog
