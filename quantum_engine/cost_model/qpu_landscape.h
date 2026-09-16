// Phase 20 step 10: cloud QPU hardware landscape + cost/qubit/error-rate
// model -- a written, literature/vendor-doc-grounded comparison of IBM
// Quantum, AWS Braket (aggregating IonQ, Rigetti, QuEra), Azure Quantum
// (aggregating IonQ, Quantinuum, Rigetti), and Xanadu Cloud, same honest-
// labeling convention as analog_engine/nvm_comparison.
//
// DISCLOSED LIMITATION, stated up front rather than buried: the figures
// below are illustrative, order-of-magnitude representative values drawn
// from each vendor's own public technology documentation and widely-
// reported specifications, NOT live-scraped or independently verified
// against a specific dated source (no network access was used to build
// this table). Real published qubit counts, gate error rates, and
// pricing change frequently and vary by device/generation within a
// single provider's own fleet -- treat every number here as "the right
// order of magnitude and qualitative ranking," not a precise citation to
// quote elsewhere. This is the same honest caveat nvm_comparison.h gives
// its own numbers, applied to a domain that changes even faster.
#pragma once

#include <algorithm>
#include <string>
#include <vector>

namespace quantum {

enum class QpuModality {
  Superconducting,  // IBM, Rigetti -- fast gates, shorter coherence, lower per-gate fidelity
  TrappedIon,       // IonQ, Quantinuum -- slower gates, much longer coherence, higher per-gate fidelity
  NeutralAtom,      // QuEra -- analog Hamiltonian simulation, a different programming model entirely
  Photonic,         // Xanadu -- continuous-variable, reports squeezed MODES not qubits (see step 8)
};

struct QpuDevice {
  std::string provider;       // cloud access point (IBM Quantum, AWS Braket, Azure Quantum, Xanadu Cloud)
  std::string hardware_vendor;  // the actual chip/device maker, which may differ from the cloud provider
  QpuModality modality;
  int approx_max_qubits_or_modes;         // "qubits" for gate-based/analog; SQUEEZED MODES for photonic (not comparable 1:1 -- see step 8's README)
  double approx_two_qubit_gate_error_pct;  // representative order-of-magnitude figure
  double approx_coherence_time_us;         // T1/T2-class figure, representative order-of-magnitude
  std::string access_model;    // how a job actually gets submitted/queued
  std::string pricing_model;   // representative billing shape, not exact current rates
};

inline const std::vector<QpuDevice> &qpu_landscape() {
  static const std::vector<QpuDevice> table = {
      {"IBM Quantum", "IBM", QpuModality::Superconducting, 130, 0.5, 150.0,
       "Direct cloud queue (IBM Quantum Platform); free open tier plus paid Premium/Pay-as-you-go runtime access",
       "Free tier: limited queue priority and runtime-seconds; paid: per-runtime-second billing"},
      {"AWS Braket", "IonQ", QpuModality::TrappedIon, 32, 0.4, 1000000.0,
       "AWS Braket managed queue, unified SDK across multiple hardware vendors",
       "Per-task submission fee plus per-shot fee, billed through AWS"},
      {"AWS Braket", "Rigetti", QpuModality::Superconducting, 80, 2.0, 20.0,
       "AWS Braket managed queue, unified SDK across multiple hardware vendors",
       "Per-task submission fee plus per-shot fee, billed through AWS"},
      {"AWS Braket", "QuEra", QpuModality::NeutralAtom, 256, 1.5, 1.0,
       "AWS Braket managed queue; QuEra targets analog Hamiltonian simulation, a different "
       "programming model from gate-based circuits (no direct two-qubit-gate-error figure applies "
       "the same way -- reported here as a representative device-level infidelity for comparability only)",
       "Per-task submission fee plus per-shot fee, billed through AWS"},
      {"Azure Quantum", "Quantinuum", QpuModality::TrappedIon, 56, 0.1, 1000000.0,
       "Azure Quantum managed queue, unified SDK across multiple hardware vendors",
       "Per-shot / per-hour billing, billed through Azure"},
      {"Azure Quantum", "Rigetti", QpuModality::Superconducting, 80, 2.0, 20.0,
       "Azure Quantum managed queue (same underlying Rigetti hardware as the AWS Braket entry above)",
       "Per-shot / per-hour billing, billed through Azure"},
      {"Xanadu Cloud", "Xanadu", QpuModality::Photonic, 216, 0.0, 0.0,
       "Direct Xanadu Cloud API/PennyLane plugin -- the most directly relevant provider to this "
       "phase's steps 7-8 (real PennyLane framework code, real hand-rolled Gaussian-state "
       "formalism), since it's the SAME software stack reaching real hardware",
       "Per-job billing; free-tier simulator access plus paid real-hardware (Borealis-class GBS "
       "device) access"},
  };
  return table;
}

// Groups devices by modality and returns each group's mean two-qubit
// gate error -- used to check the real, well-established structural
// tradeoff (trapped-ion: slower gates, much higher fidelity;
// superconducting: faster gates, lower fidelity) directly against the
// table's own data, not just asserted in prose.
inline double mean_gate_error_for_modality(QpuModality m) {
  const auto &table = qpu_landscape();
  double sum = 0.0;
  int count = 0;
  for (const auto &d : table) {
    if (d.modality != m) continue;
    if (d.approx_two_qubit_gate_error_pct <= 0.0) continue;  // photonic entries have no comparable gate-error figure
    sum += d.approx_two_qubit_gate_error_pct;
    ++count;
  }
  return count > 0 ? sum / count : 0.0;
}

inline double mean_coherence_for_modality(QpuModality m) {
  const auto &table = qpu_landscape();
  double sum = 0.0;
  int count = 0;
  for (const auto &d : table) {
    if (d.modality != m) continue;
    if (d.approx_coherence_time_us <= 0.0) continue;
    sum += d.approx_coherence_time_us;
    ++count;
  }
  return count > 0 ? sum / count : 0.0;
}

}  // namespace quantum
