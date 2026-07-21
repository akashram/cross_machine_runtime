//===- data_loader.h - Multi-worker, rank-sharded WebDataset loader -----===//
//
// PLAN.md Phase 6 step 1: distributed data loading. Two independent
// concerns, composed:
//
//  1. Rank sharding: with `world_size` training ranks reading the same
//     dataset, each rank must see a disjoint slice, or samples get
//     duplicated (biases the loss) or dropped (wastes data). Sharding is
//     done at shard-file granularity (rank r owns shard_paths[i] for
//     every i with i % world_size == r) rather than sample granularity,
//     so ranks never need to coordinate mid-epoch — the assignment is
//     computable from `shard_paths.size()` alone. This is the standard
//     WebDataset recommendation (shards >> ranks so the striping stays
//     balanced) and it's why WebDataset shards are usually sized so a
//     dataset has many more shards than the largest anticipated world
//     size.
//
//  2. Multi-worker prefetch: within one rank, `num_workers` threads pull
//     shards from that rank's list (dynamic assignment via an atomic
//     cursor — shards vary in sample count, so static splitting would
//     leave workers idle at different times) and decode samples onto a
//     shared PrefetchQueue, so decode work overlaps whatever the
//     training loop is doing with the previous batch instead of
//     happening serially in front of it.
//
//===----------------------------------------------------------------------===//
#pragma once

#include <atomic>
#include <cstddef>
#include <string>
#include <thread>
#include <vector>

#include "prefetch_queue.h"
#include "webdataset_shard.h"

namespace data_loading {

class DataLoader {
public:
  // `shard_paths` is the full dataset's shard list (same on every rank —
  // sharding happens inside the constructor). `queue_capacity` bounds how
  // many decoded samples may sit prefetched ahead of the consumer.
  DataLoader(std::vector<std::string> shard_paths, int world_size, int rank,
             int num_workers, size_t queue_capacity);
  ~DataLoader();

  DataLoader(const DataLoader &) = delete;
  DataLoader &operator=(const DataLoader &) = delete;

  // Starts the worker threads. Must be called before next().
  void start();

  // Blocks until a sample is available, or returns nullopt once this
  // rank's entire shard list has been fully consumed.
  std::optional<Sample> next();

  // This rank's shard assignment, for tests/introspection.
  const std::vector<std::string> &rank_shards() const { return rank_shards_; }

  size_t queue_depth() const { return queue_.size(); }

private:
  void worker_loop();

  std::vector<std::string> rank_shards_;
  int num_workers_;
  PrefetchQueue<Sample> queue_;
  std::atomic<size_t> next_shard_idx_{0};
  std::atomic<int> active_producers_;
  std::vector<std::thread> workers_;
  bool started_ = false;
};

} // namespace data_loading
