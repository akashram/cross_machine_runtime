// sla_scheduler_bench.cpp — measures SLA-violation rate under load: FIFO
// admission (no priority, no preemption — what continuous_batching's
// ContinuousBatcher does) vs. the real SlaScheduler's EDF-with-preemption
// policy, on the same synthetic mixed-urgency request trace.
//
// PLAN.md Phase 9 step 3 doesn't name a specific metric the way step 2
// does ("measure throughput improvement"), but "preemption when budget
// exceeded" only matters if it changes an outcome — this quantifies that
// outcome as SLA-violation rate (fraction of requests that finish after
// their deadline).
//
// Simulated in discrete ticks, same rationale as
// continuous_batching_bench.cpp: no real GPU compute here, so wall-clock
// timing this loop would measure the CPU's scheduling-loop speed, not the
// policy's effect on deadlines.
#include "sla_scheduler.h"

#include <algorithm>
#include <cstdio>
#include <deque>
#include <random>
#include <unordered_set>
#include <vector>

using namespace inference_serving;

namespace {

struct BenchRequest {
  int id;
  double arrival_time_s;
  double latency_budget_s;
  int service_ticks;  // decode steps needed; 1 tick == 1 simulated second
};

std::vector<BenchRequest> make_trace(int n, std::mt19937 &rng) {
  std::uniform_int_distribution<int> service_dist(1, 20);
  std::uniform_real_distribution<double> arrival_dist(0.0, static_cast<double>(n) / 4.0);
  // Mixed urgency: 20% tight (interactive-latency-class) requests, 80% loose.
  std::uniform_int_distribution<int> urgency_roll(0, 9);
  std::vector<BenchRequest> trace;
  trace.reserve(static_cast<std::size_t>(n));
  for (int i = 0; i < n; ++i) {
    int service = service_dist(rng);
    double budget = (urgency_roll(rng) < 2) ? service * 1.5 : service * 8.0;
    trace.push_back({i, arrival_dist(rng), budget, service});
  }
  return trace;
}

struct BenchResult {
  int num_requests = 0;
  int num_violations = 0;
  double violation_rate_pct() const { return 100.0 * num_violations / num_requests; }
};

// FIFO, no priority, no preemption: same admission rule
// continuous_batching's ContinuousBatcher uses, just reimplemented here
// standalone so this file can track finish ticks against deadlines
// without threading that concern into batcher.h.
BenchResult run_fifo(std::vector<BenchRequest> reqs, int max_batch_size) {
  std::sort(reqs.begin(), reqs.end(), [](const BenchRequest &a, const BenchRequest &b) {
    return a.arrival_time_s < b.arrival_time_s;
  });
  std::deque<BenchRequest> waiting(reqs.begin(), reqs.end());
  std::unordered_set<int> active;
  std::vector<int> remaining(reqs.size());
  std::vector<double> deadline(reqs.size());
  for (const auto &r : reqs) {
    remaining[static_cast<std::size_t>(r.id)] = r.service_ticks;
    deadline[static_cast<std::size_t>(r.id)] = r.arrival_time_s + r.latency_budget_s;
  }

  BenchResult result;
  result.num_requests = static_cast<int>(reqs.size());
  int finished = 0;
  double now = 0.0;
  while (finished < static_cast<int>(reqs.size())) {
    while (!waiting.empty() && waiting.front().arrival_time_s <= now && static_cast<int>(active.size()) < max_batch_size) {
      active.insert(waiting.front().id);
      waiting.pop_front();
    }
    for (auto it = active.begin(); it != active.end();) {
      int id = *it;
      if (--remaining[static_cast<std::size_t>(id)] == 0) {
        if (now > deadline[static_cast<std::size_t>(id)]) ++result.num_violations;
        ++finished;
        it = active.erase(it);
      } else {
        ++it;
      }
    }
    now += 1.0;
  }
  return result;
}

BenchResult run_edf(std::vector<BenchRequest> reqs, int max_batch_size) {
  SlaScheduler sched;
  std::sort(reqs.begin(), reqs.end(), [](const BenchRequest &a, const BenchRequest &b) {
    return a.arrival_time_s < b.arrival_time_s;
  });
  std::vector<int> remaining(reqs.size());
  for (const auto &r : reqs) remaining[static_cast<std::size_t>(r.id)] = r.service_ticks;

  BenchResult result;
  result.num_requests = static_cast<int>(reqs.size());
  std::size_t next_arrival = 0;
  int finished = 0;
  double now = 0.0;
  while (finished < static_cast<int>(reqs.size())) {
    while (next_arrival < reqs.size() && reqs[next_arrival].arrival_time_s <= now) {
      const auto &r = reqs[next_arrival];
      sched.add_request({r.id, r.arrival_time_s, r.latency_budget_s});
      ++next_arrival;
    }
    auto tick = sched.schedule_tick(max_batch_size, now);
    // Preempted requests simply pause (their KV/progress is preserved in
    // this model) — the point under test is deadline outcomes from
    // admission-priority policy, not the added cost of a real evict/
    // resume cycle, which is a separate, backend-specific concern.
    for (int id : tick.active_seq_ids) {
      if (--remaining[static_cast<std::size_t>(id)] == 0) {
        sched.finish_sequence(id);
        ++finished;
      }
    }
    now += 1.0;
  }
  result.num_violations = sched.num_sla_violations();
  return result;
}

} // namespace

int main() {
  std::mt19937 rng(7);
  auto trace = make_trace(200, rng);
  constexpr int kBatchSize = 16;

  BenchResult fifo = run_fifo(trace, kBatchSize);
  BenchResult edf = run_edf(trace, kBatchSize);

  std::printf("%-10s %12s %14s %16s\n", "policy", "requests", "violations", "violation rate");
  std::printf("%-10s %12d %14d %15.1f%%\n", "fifo", fifo.num_requests, fifo.num_violations, fifo.violation_rate_pct());
  std::printf("%-10s %12d %14d %15.1f%%\n", "edf", edf.num_requests, edf.num_violations, edf.violation_rate_pct());

  return 0;
}
