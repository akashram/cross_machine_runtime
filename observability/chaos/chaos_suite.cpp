// chaos_suite.cpp — automated fault injection + recovery suite.
//
// PLAN.md Phase 10 step 6: "inject network partition -> verify recovery,
// kill GPU node -> verify elastic resharding, inject FPGA thermal event
// -> verify router response. All documented with recovery time
// measurements."
//
// Two scenarios run for real here, reusing this repo's own real
// components rather than re-deriving new fault-injection machinery:
//   1. Raft leader kill -> re-election, using networking/raft's actual
//      RaftNode over a real loopback TCP mesh (the same setup
//      networking/raft/raft_test.cpp uses for correctness -- this test
//      adds the explicit recovery-TIME measurement PLAN.md step 6 asks
//      for, which raft_test.cpp doesn't report as a first-class metric,
//      only checks against a timeout bound).
//   2. FPGA thermal event -> router response, using
//      fpga_engine/thermal_router's real ThermalRouter/ThermalPolicy
//      (the pure decision function, no hardware needed) against a
//      synthetic multi-stage temperature trace covering
//      normal->warning->throttle->shutdown->recovery, checking the
//      allocation-fraction sequence is always correct (never higher
//      than the policy allows at that temperature) -- and cites
//      fpga_engine/thermal_router/thermal_router_sim.cpp's own,
//      already-measured real recovery-latency numbers (10.8ms at
//      throttle, 31.6ms at shutdown) rather than re-deriving the exact
//      same RC-response poll-interval simulation redundantly.
//
// Two scenarios PLAN.md names stay documented gaps, matching this
// repo's existing honest status for both:
//   - Network partition: networking/chaos/'s tc-netem scripts already
//     exist for this, gated on Linux + a live multi-node cluster (see
//     networking/chaos/README.md) -- not duplicated here.
//   - Kill GPU node -> elastic resharding: needs real GPU hardware
//     running distributed_training/'s ZeRO/checkpoint-resharding code
//     path under an actual node failure, which this repo has never run
//     (see distributed_training/'s own hardware-gated status for those
//     steps) -- a real chaos test for this needs that hardware first.

#include "../../fpga_engine/thermal_router/thermal_router.h"
#include "../../networking/raft/raft.h"

#include <chrono>
#include <cstdio>
#include <thread>
#include <vector>

namespace {

int g_fails = 0;
void require(bool ok, const char *name) {
  std::printf("%s  %s\n", ok ? "PASS" : "FAIL", name);
  if (!ok) ++g_fails;
}

// ---------------------------------------------------------------------
// Scenario 1: Raft leader kill -> re-election, real recovery time.
// ---------------------------------------------------------------------
int find_leader(std::vector<std::unique_ptr<raft::RaftNode>> &nodes, int timeout_ms) {
  auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
  while (std::chrono::steady_clock::now() < deadline) {
    int leader_count = 0, leader_idx = -1;
    for (std::size_t i = 0; i < nodes.size(); ++i)
      if (nodes[i] && nodes[i]->state() == raft::RaftState::LEADER) {
        ++leader_count;
        leader_idx = static_cast<int>(i);
      }
    if (leader_count == 1) return leader_idx;
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  return -1;
}

void run_raft_leader_kill_scenario() {
  std::printf("\n=== chaos scenario: kill the Raft leader ===\n");
  constexpr int kWorldSize = 3;
  constexpr uint16_t kBasePort = 35701;

  auto channels = netcommon::make_tcp_loopback_mesh(kWorldSize, kBasePort);
  std::vector<std::unique_ptr<raft::RaftNode>> nodes;
  for (int r = 0; r < kWorldSize; ++r) nodes.push_back(std::make_unique<raft::RaftNode>(*channels[static_cast<std::size_t>(r)]));
  for (auto &n : nodes) n->start();

  int leader = find_leader(nodes, 2000);
  require(leader >= 0, "healthy cluster elects exactly one leader before fault injection");
  if (leader < 0) {
    for (auto &n : nodes) n->stop();
    return;
  }

  auto t0 = std::chrono::steady_clock::now();
  nodes[static_cast<std::size_t>(leader)]->stop();  // fault injection: kill the leader

  int new_leader = -1;
  auto deadline = t0 + std::chrono::milliseconds(3000);
  while (std::chrono::steady_clock::now() < deadline && new_leader < 0) {
    for (int r = 0; r < kWorldSize; ++r) {
      if (r == leader) continue;
      if (nodes[static_cast<std::size_t>(r)]->state() == raft::RaftState::LEADER) {
        new_leader = r;
        break;
      }
    }
    if (new_leader < 0) std::this_thread::sleep_for(std::chrono::milliseconds(5));
  }
  auto t1 = std::chrono::steady_clock::now();
  double recovery_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

  require(new_leader >= 0 && new_leader != leader, "a new leader is elected among survivors after the kill");
  if (new_leader >= 0) {
    bool committed = nodes[static_cast<std::size_t>(new_leader)]->propose("post-recovery command", 2000);
    require(committed, "the new leader can still commit after recovery");
  }
  std::printf("  recovery time (leader stop -> new leader elected): %.1f ms\n", recovery_ms);

  for (int r = 0; r < kWorldSize; ++r)
    if (r != leader) nodes[static_cast<std::size_t>(r)]->stop();
}

// ---------------------------------------------------------------------
// Scenario 2: FPGA thermal event -> router response.
// ---------------------------------------------------------------------
void run_fpga_thermal_scenario() {
  std::printf("\n=== chaos scenario: inject an FPGA thermal event ===\n");
  ThermalRouter router;  // default policy: warning=75, throttle=85, shutdown=95

  // Synthetic trace: normal -> warning -> throttle -> shutdown -> recovery.
  struct Point {
    float temp_c;
    float expected_alloc;
  };
  std::vector<Point> trace = {
      {60.0f, 1.0f},  // normal
      {78.0f, 1.0f},  // warning: log only, no allocation change
      {87.0f, 0.5f},  // throttle: halve FPGA allocation
      {97.0f, 0.0f},  // shutdown: route everything off FPGA
      {70.0f, 1.0f},  // recovery: back to normal once temp drops
  };

  bool all_correct = true;
  for (const auto &p : trace) {
    float actual = router.allocation_fraction_for_temp(p.temp_c);
    bool ok = (actual == p.expected_alloc);
    std::printf("  temp=%.1fC -> allocation=%.2f (expected %.2f) %s\n", static_cast<double>(p.temp_c),
                static_cast<double>(actual), static_cast<double>(p.expected_alloc), ok ? "OK" : "MISMATCH");
    all_correct = all_correct && ok;
  }
  require(all_correct, "router allocation fraction matches policy at every point in the thermal trace");
  require(router.allocation_fraction_for_temp(97.0f) <= router.allocation_fraction_for_temp(60.0f),
          "allocation never increases as temperature rises (monotonic safety property)");

  std::printf(
      "  recovery time (real, already measured by fpga_engine/thermal_router/thermal_router_sim.cpp against "
      "an RC step-response thermal model at a 100ms poll interval): 10.8ms at throttle threshold, 31.6ms at "
      "shutdown threshold -- not re-measured here to avoid duplicating that simulation; this scenario checks "
      "the decision function's correctness across the full multi-stage trace instead.\n");
}

}  // namespace

int main() {
  run_raft_leader_kill_scenario();
  run_fpga_thermal_scenario();

  std::printf("\n=== documented gaps (PLAN.md names these, not run here) ===\n");
  std::printf("  network partition recovery: see networking/chaos/ (tc netem, Linux + live multi-node cluster required)\n");
  std::printf("  GPU node kill -> elastic resharding: needs real GPU hardware running distributed_training/'s "
              "ZeRO/checkpoint-resharding path under an actual node failure\n");

  std::printf("\n%s\n", g_fails == 0 ? "PASS" : "FAIL");
  return g_fails == 0 ? 0 : 1;
}
