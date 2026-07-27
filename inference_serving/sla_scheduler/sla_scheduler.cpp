#include "sla_scheduler.h"

#include <algorithm>

namespace inference_serving {

void SlaScheduler::add_request(SlaRequest r) { pending_[r.id] = r; }

SlaScheduler::TickResult SlaScheduler::schedule_tick(int max_batch_size, double now_s) {
  // Violations: count once per request whose deadline has passed while
  // it's still pending (active or waiting) — a metric, not a reason to
  // drop the request; a real server still serves it, just late.
  for (const auto &[id, req] : pending_) {
    if (req.deadline_s() < now_s && violated_.find(id) == violated_.end()) {
      violated_.insert(id);
      ++num_violations_;
    }
  }

  // EDF: the new active set is exactly the earliest-deadline
  // min(max_batch_size, pending_.size()) requests among ALL pending ones
  // (active or waiting) — this is what makes preemption happen
  // naturally: a request that was active but is no longer among the
  // earliest deadlines drops out.
  std::vector<int> by_deadline;
  by_deadline.reserve(pending_.size());
  for (const auto &[id, req] : pending_) by_deadline.push_back(id);
  std::sort(by_deadline.begin(), by_deadline.end(), [this](int a, int b) {
    return pending_.at(a).deadline_s() < pending_.at(b).deadline_s();
  });

  std::size_t new_active_count = std::min(static_cast<std::size_t>(max_batch_size), by_deadline.size());
  std::unordered_set<int> new_active(by_deadline.begin(), by_deadline.begin() + static_cast<long>(new_active_count));

  TickResult result;
  for (int id : active_) {
    if (new_active.find(id) == new_active.end()) {
      result.preempted_seq_ids.push_back(id);
      ++num_preemptions_;
    }
  }
  active_ = new_active;
  result.active_seq_ids.assign(active_.begin(), active_.end());
  return result;
}

void SlaScheduler::finish_sequence(int seq_id) {
  active_.erase(seq_id);
  pending_.erase(seq_id);
  violated_.erase(seq_id);
}

}  // namespace inference_serving
