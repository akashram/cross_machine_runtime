#pragma once

// Sharded checkpointing: PLAN.md step 17. Each rank owns and writes ONLY
// its own parameter shard (matching zero3/'s sharded parameter layout) to
// its own file — no gather-to-rank-0 bottleneck, which is the entire point
// of sharded checkpointing at real model scale (gathering a 100B-parameter
// model to one rank to write it is both slow and requires that one rank
// to have enough memory to hold the whole thing). Async write overlaps
// the file I/O with continued training on the same rank, using a real
// background thread — not a cost-model estimate (contrast step 10's
// ZeRO-Infinity offload scheduler, which used an analytical model because
// its "device" wasn't real; here the file write and the wall-clock ARE
// real, so this measures the actual thing, not a stand-in for it).

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <future>
#include <stdexcept>
#include <string>
#include <vector>

namespace distributed_training {

inline void write_shard_sync(const std::string &path, const std::vector<float> &shard) {
  FILE *f = std::fopen(path.c_str(), "wb");
  if (!f) throw std::runtime_error("write_shard_sync: failed to open " + path);
  uint64_t count = shard.size();
  std::fwrite(&count, sizeof(count), 1, f);
  if (count > 0) std::fwrite(shard.data(), sizeof(float), shard.size(), f);
  std::fclose(f);
}

inline std::vector<float> read_shard(const std::string &path) {
  FILE *f = std::fopen(path.c_str(), "rb");
  if (!f) throw std::runtime_error("read_shard: failed to open " + path);
  uint64_t count = 0;
  if (std::fread(&count, sizeof(count), 1, f) != 1) { std::fclose(f); throw std::runtime_error("read_shard: truncated header"); }
  std::vector<float> shard(count);
  if (count > 0 && std::fread(shard.data(), sizeof(float), count, f) != count) {
    std::fclose(f);
    throw std::runtime_error("read_shard: truncated data");
  }
  std::fclose(f);
  return shard;
}

// Writes a shard on a background thread so the caller (mid-training loop)
// is not blocked on disk I/O. A checkpoint step becomes: snapshot the
// shard (a copy — training must not mutate it while the background write
// reads it), hand it to start_write, keep training, wait() only if/when
// the next checkpoint or shutdown needs to guarantee the previous one
// finished.
class AsyncCheckpointWriter {
public:
  void start_write(const std::string &path, std::vector<float> shard) {
    future_ = std::async(std::launch::async, [path, shard = std::move(shard)]() { write_shard_sync(path, shard); });
  }

  bool is_done() const {
    return !future_.valid() || future_.wait_for(std::chrono::seconds(0)) == std::future_status::ready;
  }

  void wait() {
    if (future_.valid()) future_.get();
  }

private:
  std::future<void> future_;
};

} // namespace distributed_training
