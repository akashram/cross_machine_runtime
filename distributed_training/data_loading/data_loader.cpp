#include "data_loader.h"

namespace data_loading {

DataLoader::DataLoader(std::vector<std::string> shard_paths, int world_size,
                        int rank, int num_workers, size_t queue_capacity)
    : num_workers_(num_workers), queue_(queue_capacity),
      active_producers_(num_workers) {
  for (size_t i = 0; i < shard_paths.size(); ++i) {
    if (static_cast<int>(i % static_cast<size_t>(world_size)) == rank) {
      rank_shards_.push_back(std::move(shard_paths[i]));
    }
  }
}

DataLoader::~DataLoader() {
  for (auto &t : workers_) {
    if (t.joinable()) t.join();
  }
}

void DataLoader::start() {
  if (started_) return;
  started_ = true;
  if (rank_shards_.empty() || num_workers_ == 0) {
    queue_.close();
    return;
  }
  for (int w = 0; w < num_workers_; ++w) {
    workers_.emplace_back([this] { worker_loop(); });
  }
}

void DataLoader::worker_loop() {
  while (true) {
    size_t idx = next_shard_idx_.fetch_add(1, std::memory_order_relaxed);
    if (idx >= rank_shards_.size()) break;

    WebDatasetShardReader reader(rank_shards_[idx]);
    if (!reader.is_open()) continue; // missing/corrupt shard: skip, don't crash the run

    while (auto sample = reader.next()) {
      queue_.push(std::move(*sample));
    }
  }

  if (active_producers_.fetch_sub(1, std::memory_order_acq_rel) == 1) {
    queue_.close(); // last worker to finish unblocks the consumer
  }
}

std::optional<Sample> DataLoader::next() { return queue_.pop(); }

} // namespace data_loading
