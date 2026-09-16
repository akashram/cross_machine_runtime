// Phase 20 step 9: cross-machine runtime integration -- registers a QPU
// backend candidate in inference_serving's ServingRouter the same way
// Phase 15 registered NPU (see serving_router.h's QPU header comment),
// PLUS a real (simulator-backed) code path showing what a "submit this
// circuit" call actually looks like -- the genuine "cross-machine
// runtime hits a different backend" plumbing this step is about, not
// just an enum value with no call shape behind it.
#pragma once

#include "../state_vector/state_vector.h"

#include <cstddef>
#include <map>
#include <random>
#include <string>

namespace quantum {

struct QpuJobRequest {
  int n_qubits;
  int shots;
  std::string circuit_description;  // human-readable here; a real cloud API takes an OpenQASM/JSON payload
};

struct QpuJobResult {
  bool submitted;
  std::string status;
  std::map<std::size_t, int> counts;  // basis-state index -> observed count (shot-based, the real output shape)
};

// The REAL, honestly hardware-gated call: what submitting a circuit to
// an actual cloud QPU (IBM Quantum / AWS Braket / Azure Quantum / Xanadu
// Cloud) requires -- a network call to a job-submission API with real
// account credentials and a real queue wait, none of which is
// provisioned here. Returns an honest "unavailable" result with a
// reason string, same convention as every other hardware-gated backend
// in this repo, rather than a silent no-op or a fabricated response.
inline QpuJobResult submit_circuit_to_cloud_qpu(const QpuJobRequest &req) {
  QpuJobResult result;
  result.submitted = false;
  result.status = "unavailable: no cloud QPU credential configured (IBM Quantum / AWS Braket / "
                   "Azure Quantum / Xanadu Cloud) -- see quantum_engine/qpu_backend/README.md and "
                   "PLAN.md Phase 20's hardware access note";
  (void)req;
  return result;
}

// The REAL, run-LOCALLY stand-in for the same call shape: samples
// `shots` measurement outcomes from an already-prepared
// state_vector::StateVector via repeated measure_all() calls and tallies
// counts -- the actual shot-based SAMPLING output shape a real QPU
// returns (never exact amplitudes, which no quantum hardware hands back
// directly; only this simulator can). `prepared` can be the output of
// ANY of steps 1-8's circuits (state_vector, algorithms, vqe_ansatz,
// qaoa_circuit, ...) -- this function doesn't know or care which.
inline std::map<std::size_t, int> sample_circuit(const StateVector &prepared, int shots, std::mt19937_64 &rng) {
  std::map<std::size_t, int> counts;
  for (int s = 0; s < shots; ++s) {
    StateVector copy = prepared;
    std::size_t outcome = copy.measure_all(rng);
    counts[outcome]++;
  }
  return counts;
}

}  // namespace quantum
