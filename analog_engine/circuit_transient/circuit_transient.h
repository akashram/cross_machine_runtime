//===- circuit_transient.h - RC step response for a crossbar bitline ----===//
//
// PLAN.md Phase 17 step 8: an analog circuit transient surrogate --
// settling time and bandwidth vs. parasitic R/C for a simplified analog
// cell, same disclosed-simplification pattern as
// `fpga_engine/thermal_router`'s first-order RC step-response model
// (temperature instead of voltage there, same math) and Phase 14 step 8's
// Wald-vs-Clopper-Pearson disclosed approximation.
//
// The real question this step answers: step 2's crossbar MAC simulation
// treats a cell READ as instantaneous (no settling time modeled at all).
// Physically, reading a crossbar bitline's summed current requires the
// bitline's parasitic RC to settle first -- a real minimum time floor on
// top of step 2's noise-accuracy story, deliberately scoped OUT of step 2
// and added here instead, following this repo's convention of adding one
// real, disclosed simplification at a time rather than one step trying to
// model everything at once.
//
// R/C scale with crossbar size N via a distributed (Elmore-style) line
// model: each of N cells along a bitline contributes its own wire-segment
// resistance and parasitic capacitance to the shared line, so
//   R_total(N) = R_per_cell * N
//   C_total(N) = C_per_cell * N
//   tau(N)     = R_total(N) * C_total(N) = R_per_cell * C_per_cell * N^2
// -- tau grows QUADRATICALLY with crossbar size, the standard real result
// for a distributed RC line (Elmore delay), not this repo's invention.
// R_per_cell/C_per_cell below are illustrative literature-order-of-
// magnitude constants (advanced-node on-chip wire resistance/parasitic
// capacitance per cell-pitch), not measurements -- no fab access, same
// disclosed limitation as every other Phase 17 step.
//
//===----------------------------------------------------------------------===//
#pragma once

#include <cmath>

namespace analog {

struct RCLineParams {
  double r_per_cell_ohm = 50.0;    // illustrative on-chip wire resistance per cell-pitch
  double c_per_cell_farad = 1e-15; // illustrative parasitic capacitance per cell-pitch (1 fF)
};

// tau = R_total * C_total = r_per_cell * c_per_cell * N^2 (distributed
// RC line, Elmore-style).
inline double tau_seconds(const RCLineParams &p, int crossbar_size) {
  double n = static_cast<double>(crossbar_size);
  double r_total = p.r_per_cell_ohm * n;
  double c_total = p.c_per_cell_farad * n;
  return r_total * c_total;
}

// First-order RC step response: V(t) = V_final * (1 - exp(-t/tau)).
inline double step_response(double v_final, double tau_s, double t_s) {
  return v_final * (1.0 - std::exp(-t_s / tau_s));
}

// Closed-form time to reach `threshold_frac` of v_final (e.g. 0.99 for
// "99% settled"): solving V(t)/v_final = 1 - exp(-t/tau) = threshold_frac
// for t gives t = -tau * ln(1 - threshold_frac).
inline double settling_time_seconds(double tau_s, double threshold_frac = 0.99) {
  return -tau_s * std::log(1.0 - threshold_frac);
}

// Standard first-order RC low-pass -3dB bandwidth: f_3dB = 1 / (2*pi*tau).
inline double bandwidth_hz(double tau_s) { return 1.0 / (2.0 * M_PI * tau_s); }

} // namespace analog
