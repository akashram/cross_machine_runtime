#pragma once
#include <deque>
#include <string>
#include <unordered_set>
#include <vector>

// PLAN.md Phase 9 step 2: continuous batching — dynamic request arrival,
// batch formation across variable sequence lengths. Real scheduling
// logic, no GPU dependency (same "bookkeeping vs. device bytes" split
// paged_kv/README.md documents for step 1) — `Batch` here holds seq ids
// for this tick's decode step, not device pointers, since which
// sequences run is backend-independent.

namespace inference_serving {

struct Request {
  int id;
  std::string prompt;
  int max_new_tokens;
  double arrival_time_s;
};

// The set of sequence ids that should run one decode step this tick.
// Deliberately just ids, not `std::vector<int*> input_ids` like the
// original stub — see paged_kv/README.md's Design section for why this
// repo keeps scheduling bookkeeping separate from device tensor data.
struct Batch {
  std::vector<int> seq_ids;
};

// Admits requests as slots free up, instead of waiting for a full batch
// to accumulate (static batching) or for every sequence in a group to
// finish (which pads short sequences with wasted compute until the
// longest one in their group completes — see
// continuous_batching_bench.cpp for a real measurement of that waste).
class ContinuousBatcher {
 public:
  void add_request(Request r);

  // Admits waiting requests into any free slots (up to max_batch_size
  // active sequences total) and returns every currently-active sequence
  // id — the set that should run one decode step this tick.
  Batch next_batch(int max_batch_size = 32);

  // Removes a sequence from the active set, freeing its slot for the
  // next call to next_batch() to admit a waiting request into.
  void finish_sequence(int seq_id);

  int num_waiting() const { return static_cast<int>(waiting_.size()); }
  int num_active() const { return static_cast<int>(active_.size()); }

 private:
  std::deque<Request> waiting_;
  std::unordered_set<int> active_;
};

}  // namespace inference_serving
