// Prints the full provider/device comparison table and checks two real,
// well-established structural facts directly against the table's own
// data (not just asserted in prose, same discipline as
// analog_engine/nvm_comparison_test.cpp):
//   1. Trapped-ion devices (IonQ, Quantinuum) have a LOWER mean
//      two-qubit gate error than superconducting devices (IBM, Rigetti)
//      -- the real, textbook modality tradeoff (slower gates, much
//      higher per-gate fidelity).
//   2. Trapped-ion devices have a LONGER mean coherence time than
//      superconducting devices -- the other half of that same tradeoff.
//   3. Xanadu's photonic entry is correctly EXCLUDED from the gate-error/
//      coherence comparisons (it reports squeezed modes, not qubits, and
//      has no directly comparable two-qubit-gate-error figure) rather
//      than silently averaged into a number that wouldn't mean anything.
#include "qpu_landscape.h"

#include <cstdio>

using namespace quantum;

namespace {

int g_fails = 0;
void require(bool ok, const char *name) {
  std::printf("%s  %s\n", ok ? "PASS" : "FAIL", name);
  if (!ok) ++g_fails;
}

const char *modality_name(QpuModality m) {
  switch (m) {
    case QpuModality::Superconducting: return "superconducting";
    case QpuModality::TrappedIon: return "trapped-ion";
    case QpuModality::NeutralAtom: return "neutral-atom";
    case QpuModality::Photonic: return "photonic (CV)";
  }
  return "unknown";
}

}  // namespace

int main() {
  const auto &table = qpu_landscape();
  std::printf("  %-15s %-12s %-16s %8s %10s %12s\n", "provider", "vendor", "modality", "qubits", "2Q-err%", "coherence-us");
  for (const auto &d : table)
    std::printf("  %-15s %-12s %-16s %8d %10.2f %12.1f\n", d.provider.c_str(), d.hardware_vendor.c_str(),
                modality_name(d.modality), d.approx_max_qubits_or_modes, d.approx_two_qubit_gate_error_pct,
                d.approx_coherence_time_us);

  require(table.size() >= 4, "landscape table covers at least 4 distinct hardware entries across the 4 named providers");

  double ion_err = mean_gate_error_for_modality(QpuModality::TrappedIon);
  double sc_err = mean_gate_error_for_modality(QpuModality::Superconducting);
  std::printf("\n  mean 2Q gate error: trapped-ion=%.3f%% superconducting=%.3f%%\n", ion_err, sc_err);
  require(ion_err > 0.0 && sc_err > 0.0 && ion_err < sc_err, "trapped-ion devices have a LOWER mean two-qubit gate error than superconducting devices -- the real, textbook higher-fidelity/slower-gates tradeoff");

  double ion_coh = mean_coherence_for_modality(QpuModality::TrappedIon);
  double sc_coh = mean_coherence_for_modality(QpuModality::Superconducting);
  std::printf("  mean coherence time: trapped-ion=%.1fus superconducting=%.1fus\n", ion_coh, sc_coh);
  require(ion_coh > 0.0 && sc_coh > 0.0 && ion_coh > sc_coh, "trapped-ion devices have a LONGER mean coherence time than superconducting devices -- the other half of the same tradeoff");

  double photonic_err = mean_gate_error_for_modality(QpuModality::Photonic);
  require(photonic_err == 0.0, "Xanadu's photonic entry is correctly excluded from the gate-error comparison (reports squeezed modes, not a directly comparable two-qubit-gate-error figure) rather than silently averaged in");

  std::printf("%s\n", g_fails == 0 ? "PASS" : "FAIL");
  return g_fails == 0 ? 0 : 1;
}
