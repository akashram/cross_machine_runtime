// offload_scheduler_test.cpp — two checks:
//  1. run_double_buffered's SCHEDULING LOGIC is correct: every shard is
//     computed exactly once, in order, and shard i+1's transfer is always
//     issued before shard i's compute (the actual prefetch-ahead property
//     that makes overlap possible on real hardware).
//  2. simulate_offload_schedule's cost-model formula, evaluated across
//     three illustrative regimes (compute-bound, balanced, transfer-bound)
//     — real bandwidth/compute numbers are Results-table TODOs (see
//     README.md), but the arithmetic relating them is exercised and
//     printed here.
#include "offload_scheduler.h"

#include <cstdio>
#include <vector>

using namespace distributed_training;

namespace {

bool test_schedule_correctness() {
  constexpr int kNumShards = 6;
  std::vector<std::pair<char, int>> log; // ('T', shard) or ('C', shard), in call order

  auto transfer_fn = [&](int i) { log.push_back({'T', i}); };
  auto compute_fn = [&](int i) { log.push_back({'C', i}); };

  auto order = run_double_buffered(kNumShards, transfer_fn, compute_fn);

  bool ok = true;
  if (static_cast<int>(order.size()) != kNumShards) ok = false;
  for (int i = 0; i < kNumShards; ++i) {
    if (order[static_cast<size_t>(i)] != i) ok = false; // computed in order, none skipped/duplicated
  }

  // Find each 'C', i log position; verify 'T', i+1 appears strictly earlier
  // in the log (prefetch-ahead), for every i with a next shard.
  auto find_pos = [&](char kind, int shard) -> int {
    for (size_t p = 0; p < log.size(); ++p)
      if (log[p].first == kind && log[p].second == shard) return static_cast<int>(p);
    return -1;
  };
  for (int i = 0; i + 1 < kNumShards; ++i) {
    int transfer_next_pos = find_pos('T', i + 1);
    int compute_i_pos = find_pos('C', i);
    if (transfer_next_pos < 0 || compute_i_pos < 0 || transfer_next_pos >= compute_i_pos) ok = false;
  }
  // Very first call overall must be transfer(0) — the pipeline fill.
  if (log.empty() || log.front() != std::make_pair('T', 0)) ok = false;

  std::printf("test 1 (double-buffered schedule correctness): %s\n", ok ? "PASS" : "FAIL");
  return ok;
}

void print_scenario(const char *name, int num_shards, double compute_ms, double transfer_ms) {
  auto r = simulate_offload_schedule(num_shards, compute_ms, transfer_ms);
  std::printf("  %-32s compute=%.1fms transfer=%.1fms -> naive=%.1fms overlapped=%.1fms speedup=%.2fx\n", name,
              compute_ms, transfer_ms, r.naive_total_ms, r.overlapped_total_ms, r.speedup);
}

} // namespace

int main() {
  bool ok = test_schedule_correctness();

  std::printf("\ntest 2 (cost model across regimes, %d shards, illustrative inputs — see README):\n", 8);
  print_scenario("compute-bound (fast NVMe/RAM)", 8, 10.0, 2.0);
  print_scenario("balanced (compute ~= transfer)", 8, 5.0, 5.0);
  print_scenario("transfer-bound (slow NVMe)", 8, 2.0, 10.0);
  std::printf("(no PASS/FAIL — these are illustrative model outputs, not assertions; see README.md Results for\n"
              " what needs real hardware numbers)\n");

  std::printf("\n%s\n", ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
}
