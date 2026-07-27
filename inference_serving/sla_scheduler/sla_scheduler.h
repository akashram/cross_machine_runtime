#pragma once
#include <unordered_map>
#include <unordered_set>
#include <vector>

// PLAN.md Phase 9 step 3: SLA-aware scheduler — latency budget per
// request, preemption when budget exceeded, priority queuing.
//
// Implements Earliest-Deadline-First (EDF): every request carries a
// deadline (arrival_time_s + latency_budget_s), and schedule_tick()
// always makes the active set exactly the max_batch_size
// not-yet-finished requests with the earliest deadlines, evicting
// (preempting) whichever previously-active request no longer qualifies
// back into the waiting pool. EDF is the standard optimal policy for
// this exact problem (single resource, hard deadlines, preemption
// allowed) in real-time scheduling theory — not a bespoke heuristic.

namespace inference_serving {

struct SlaRequest {
  int id;
  double arrival_time_s;
  double latency_budget_s;

  double deadline_s() const { return arrival_time_s + latency_budget_s; }
};

class SlaScheduler {
 public:
  void add_request(SlaRequest r);

  // Recomputes the active set as the max_batch_size not-yet-finished
  // requests with the earliest deadlines as of `now_s`, records an SLA
  // violation (once per request) for anything whose deadline has already
  // passed, and returns the active set's seq_ids plus which ids were
  // just preempted (dropped from active back to waiting) this tick.
  struct TickResult {
    std::vector<int> active_seq_ids;
    std::vector<int> preempted_seq_ids;
  };
  TickResult schedule_tick(int max_batch_size, double now_s);

  void finish_sequence(int seq_id);

  int num_sla_violations() const { return num_violations_; }
  int num_preemptions() const { return num_preemptions_; }
  int num_pending() const { return static_cast<int>(pending_.size()); }

 private:
  std::unordered_map<int, SlaRequest> pending_;  // not yet finished (active or waiting)
  std::unordered_set<int> active_;
  std::unordered_set<int> violated_;  // already-counted violations, avoid double-counting
  int num_violations_ = 0;
  int num_preemptions_ = 0;
};

}  // namespace inference_serving
