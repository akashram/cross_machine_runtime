// Verifies: (1) sample_circuit's shot-based sampling on a Bell state
// matches the expected exact-50/50 |00>/|11> distribution statistically,
// with zero counts on |01>/|10> (an exact structural property, not just
// approximate); (2) submit_circuit_to_cloud_qpu is honest about being
// unavailable, same convention as every other hardware-gated backend in
// this repo; (3) QPU actually integrates into inference_serving's
// ServingRouter with the documented fallback priority (FPGA > NPU > QPU
// > CPU), reusing the router unmodified rather than adding QPU-specific
// logic to it.
#include "qpu_backend.h"

#include "../../inference_serving/serving_backend/serving_router.h"

#include <cmath>
#include <cstdio>

using namespace quantum;
using namespace inference_serving;

namespace {

int g_fails = 0;
void require(bool ok, const char *name) {
  std::printf("%s  %s\n", ok ? "PASS" : "FAIL", name);
  if (!ok) ++g_fails;
}

void test_sample_circuit_matches_bell_state_distribution() {
  StateVector bell(2);
  bell.h(0);
  bell.cnot(0, 1);

  std::mt19937_64 rng(5);
  int shots = 4000;
  auto counts = sample_circuit(bell, shots, rng);

  int c00 = counts.count(0) ? counts.at(0) : 0;
  int c01 = counts.count(1) ? counts.at(1) : 0;
  int c10 = counts.count(2) ? counts.at(2) : 0;
  int c11 = counts.count(3) ? counts.at(3) : 0;
  std::printf("  Bell state, %d shots: |00>=%d |01>=%d |10>=%d |11>=%d\n", shots, c00, c01, c10, c11);

  bool zero_elsewhere = (c01 == 0) && (c10 == 0);
  bool total_ok = (c00 + c11) == shots;
  double frac00 = static_cast<double>(c00) / shots;
  bool balanced = std::abs(frac00 - 0.5) < 0.03;  // statistical tolerance at 4000 shots
  require(zero_elsewhere && total_ok, "sample_circuit on a Bell state produces EXACTLY zero counts on |01>/|10> (structural, not statistical)");
  require(balanced, "sample_circuit on a Bell state splits roughly 50/50 between |00> and |11> (within 3%% at 4000 shots)");
}

void test_cloud_qpu_submission_is_honest() {
  QpuJobRequest req{2, 1000, "H(0); CNOT(0,1)"};
  QpuJobResult result = submit_circuit_to_cloud_qpu(req);
  std::printf("  submit_circuit_to_cloud_qpu: submitted=%s status=\"%s\"\n", result.submitted ? "true" : "false",
              result.status.c_str());
  require(!result.submitted && !result.status.empty(), "submit_circuit_to_cloud_qpu honestly reports unavailable with a non-empty reason string, same convention as every other hardware-gated backend");
}

void test_qpu_integrates_into_serving_router() {
  ServingRouter router;
  router.register_backend(Backend::CPU, {true, ""}, make_cpu_backend());
  router.register_backend(Backend::QPU, {false, "no cloud QPU credential configured"});
  BackendInfo qpu_info = router.info(Backend::QPU);
  std::printf("  QPU registered: available=%s reason=\"%s\"\n", qpu_info.available ? "true" : "false",
              qpu_info.unavailable_reason.c_str());
  require(!qpu_info.available && !qpu_info.unavailable_reason.empty(), "QPU registers into ServingRouter as unavailable with an honest reason string, same as GPU/FPGA/TPU/NPU");

  // Priority ordering: register FPGA, NPU, and QPU all as AVAILABLE with
  // distinguishable marker generate-functions (mirrors
  // serving_router_test.cpp's own NPU-priority test), and confirm the
  // documented FPGA > NPU > QPU > CPU order holds without any
  // QPU-specific logic added to ServingRouter itself.
  ServingRouter router2;
  router2.register_backend(Backend::CPU, {true, ""}, make_cpu_backend());
  router2.register_backend(Backend::QPU, {true, ""}, [](const transformer::ModelParams &, const std::vector<int> &, int) {
    return std::vector<int>{-3};
  });
  router2.register_backend(Backend::NPU, {true, ""}, [](const transformer::ModelParams &, const std::vector<int> &, int) {
    return std::vector<int>{-2};
  });
  router2.register_backend(Backend::FPGA, {true, ""}, [](const transformer::ModelParams &, const std::vector<int> &, int) {
    return std::vector<int>{-4};
  });

  transformer::ModelParams dummy_model{};
  RouteResult r_qpu_vs_cpu = router2.route(Backend::TPU /*unregistered, forces fallback*/, dummy_model, {0}, 1);
  // With FPGA, NPU, and QPU all available, FPGA should win (highest
  // priority among these three).
  require(r_qpu_vs_cpu.backend_used == Backend::FPGA, "with FPGA/NPU/QPU/CPU all available, fallback prefers FPGA (highest priority) over NPU, QPU, and CPU");

  ServingRouter router3;
  router3.register_backend(Backend::CPU, {true, ""}, make_cpu_backend());
  router3.register_backend(Backend::QPU, {true, ""}, [](const transformer::ModelParams &, const std::vector<int> &, int) {
    return std::vector<int>{-3};
  });
  RouteResult r_qpu_only = router3.route(Backend::TPU, dummy_model, {0}, 1);
  require(r_qpu_only.backend_used == Backend::QPU, "with only QPU and CPU available (no GPU/TPU/FPGA/NPU), fallback prefers QPU over CPU");
  require(r_qpu_only.tokens == std::vector<int>{-3}, "routing to QPU actually dispatches to QPU's registered generate function, not CPU's");
}

}  // namespace

int main() {
  test_sample_circuit_matches_bell_state_distribution();
  test_cloud_qpu_submission_is_honest();
  test_qpu_integrates_into_serving_router();
  std::printf("%s\n", g_fails == 0 ? "PASS" : "FAIL");
  return g_fails == 0 ? 0 : 1;
}
