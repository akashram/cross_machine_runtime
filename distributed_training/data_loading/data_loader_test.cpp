// data_loader_test.cpp — two things, over a synthetic on-disk WebDataset:
//  1. correctness: 4-way rank sharding is disjoint and complete, and every
//     sample round-trips through the tar codec byte-for-byte.
//  2. bench: single-worker vs 8-worker prefetch throughput and "stall
//     fraction" (how often the consumer found the queue empty) — the
//     local proxy for PLAN.md's "measure GPU utilization with and without
//     pipeline" (no GPU on this Mac; see README.md).
#include "data_loader.h"
#include "webdataset_shard.h"

#include <unistd.h>

#include <chrono>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <set>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using namespace data_loading;

namespace {

std::vector<std::string> write_synthetic_dataset(const std::string &dir, int num_shards,
                                                   int samples_per_shard, size_t payload_bytes) {
  fs::create_directories(dir);
  std::vector<std::string> shard_paths;
  int global_id = 0;
  for (int s = 0; s < num_shards; ++s) {
    std::vector<uint8_t> tar;
    for (int i = 0; i < samples_per_shard; ++i, ++global_id) {
      char key[32];
      std::snprintf(key, sizeof(key), "%08d", global_id);
      std::vector<uint8_t> payload(payload_bytes);
      for (size_t b = 0; b < payload_bytes; ++b) {
        payload[b] = static_cast<uint8_t>((global_id + static_cast<int>(b)) & 0xFF);
      }
      std::vector<uint8_t> label{static_cast<uint8_t>(global_id % 10)};
      tar_append(tar, std::string(key) + ".data", payload);
      tar_append(tar, std::string(key) + ".cls", label);
    }
    tar_finish(tar);
    std::string path = dir + "/shard-" + std::to_string(s) + ".tar";
    std::ofstream out(path, std::ios::binary);
    out.write(reinterpret_cast<const char *>(tar.data()), static_cast<std::streamsize>(tar.size()));
    shard_paths.push_back(path);
  }
  return shard_paths;
}

bool correctness_test(const std::vector<std::string> &shard_paths, int total_expected) {
  constexpr int kWorldSize = 4;
  std::set<std::string> seen_keys;
  bool ok = true;

  for (int rank = 0; rank < kWorldSize; ++rank) {
    DataLoader loader(shard_paths, kWorldSize, rank, /*num_workers=*/3, /*queue_capacity=*/8);
    loader.start();
    while (auto sample = loader.next()) {
      if (seen_keys.count(sample->key)) {
        std::printf("DUPLICATE key %s (rank %d)\n", sample->key.c_str(), rank);
        ok = false;
      }
      seen_keys.insert(sample->key);

      int global_id = std::stoi(sample->key);
      auto data_it = sample->files.find("data");
      auto cls_it = sample->files.find("cls");
      if (data_it == sample->files.end() || cls_it == sample->files.end()) {
        std::printf("MISSING files for key %s\n", sample->key.c_str());
        ok = false;
        continue;
      }
      bool payload_ok = true;
      for (size_t b = 0; b < data_it->second.size(); ++b) {
        if (data_it->second[b] != static_cast<uint8_t>((global_id + static_cast<int>(b)) & 0xFF)) {
          payload_ok = false;
          break;
        }
      }
      if (!payload_ok || cls_it->second[0] != static_cast<uint8_t>(global_id % 10)) {
        std::printf("CORRUPT sample %s\n", sample->key.c_str());
        ok = false;
      }
    }
  }

  if (static_cast<int>(seen_keys.size()) != total_expected) {
    std::printf("expected %d samples total across ranks, saw %zu\n", total_expected, seen_keys.size());
    ok = false;
  }
  return ok;
}

struct BenchResult {
  double wall_seconds;
  double samples_per_sec;
  double mb_per_sec;
  double stall_fraction;
};

BenchResult run_bench(const std::vector<std::string> &shard_paths, int num_workers, size_t total_bytes) {
  DataLoader loader(shard_paths, /*world_size=*/1, /*rank=*/0, num_workers, /*queue_capacity=*/16);
  auto start = std::chrono::steady_clock::now();
  loader.start();

  size_t consumed = 0;
  size_t stalls = 0;
  while (true) {
    if (loader.queue_depth() == 0) stalls++; // racy diagnostic, not a correctness check
    auto sample = loader.next();
    if (!sample.has_value()) break;
    consumed++;
  }
  double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();

  return BenchResult{elapsed, static_cast<double>(consumed) / elapsed, (static_cast<double>(total_bytes) / 1e6) / elapsed,
                      static_cast<double>(stalls) / static_cast<double>(consumed)};
}

} // namespace

int main() {
  std::string dir = fs::temp_directory_path().string() + "/wds_test_" + std::to_string(::getpid());
  constexpr int kNumShards = 20;
  constexpr int kSamplesPerShard = 100;
  constexpr size_t kPayloadBytes = 32 * 1024;
  constexpr int kTotalSamples = kNumShards * kSamplesPerShard;
  constexpr size_t kTotalBytes = static_cast<size_t>(kTotalSamples) * kPayloadBytes;

  auto shard_paths = write_synthetic_dataset(dir, kNumShards, kSamplesPerShard, kPayloadBytes);

  std::printf("=== correctness: %d-way rank sharding, disjoint + complete ===\n", 4);
  bool ok = correctness_test(shard_paths, kTotalSamples);
  std::printf("%s\n", ok ? "PASS" : "FAIL");

  std::printf("\n=== bench: single-worker vs 8-worker prefetch (%zu MB dataset) ===\n", kTotalBytes / (1024 * 1024));
  auto single = run_bench(shard_paths, 1, kTotalBytes);
  auto multi = run_bench(shard_paths, 8, kTotalBytes);
  std::printf("num_workers=1: %.3fs  %.0f samples/s  %.1f MB/s  stall_fraction=%.3f\n", single.wall_seconds,
              single.samples_per_sec, single.mb_per_sec, single.stall_fraction);
  std::printf("num_workers=8: %.3fs  %.0f samples/s  %.1f MB/s  stall_fraction=%.3f\n", multi.wall_seconds,
              multi.samples_per_sec, multi.mb_per_sec, multi.stall_fraction);
  std::printf("speedup: %.2fx\n", single.wall_seconds / multi.wall_seconds);

  fs::remove_all(dir);

  std::printf("\n%s\n", ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
}
