#include "batcher.h"

namespace inference_serving {

void ContinuousBatcher::add_request(Request r) { waiting_.push_back(std::move(r)); }

Batch ContinuousBatcher::next_batch(int max_batch_size) {
  while (static_cast<int>(active_.size()) < max_batch_size && !waiting_.empty()) {
    Request r = std::move(waiting_.front());
    waiting_.pop_front();
    active_.insert(r.id);
  }

  Batch batch;
  batch.seq_ids.reserve(active_.size());
  for (int id : active_) batch.seq_ids.push_back(id);
  return batch;
}

void ContinuousBatcher::finish_sequence(int seq_id) { active_.erase(seq_id); }

}  // namespace inference_serving
