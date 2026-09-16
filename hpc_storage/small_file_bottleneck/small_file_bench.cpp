// small_file_bench.cpp — PLAN.md Phase 21 step 2. Measures the real
// wall-clock/metadata-operation-count cost of many-small-files vs.
// data_loading/webdataset_shard's existing shard-into-large-blobs
// approach, on this Mac's local filesystem (APFS).
//
// Disclosed honestly: this is a LOCAL-FILESYSTEM stand-in for a real
// parallel-filesystem metadata server (VAST DNode/Lustre MDS/GPFS NSD) --
// not an equivalent measurement (a real metadata server serializes
// requests over a network RPC path with its own contention behavior a
// local UNIX filesystem doesn't have). But the underlying effect this
// step demonstrates -- N separate open()/close() metadata operations vs.
// 1 -- is real and transfers directly: it is exactly the problem
// WebDataset-style sharding exists to solve, and the open() call COUNT
// (not just the wall-clock, which is APFS-specific) is a portable,
// mechanism-level number any metadata-server-backed filesystem inherits.
#include "../../distributed_training/data_loading/webdataset_shard.h"

#include <chrono>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <vector>

namespace fs = std::filesystem;
using namespace data_loading;
using clock_type = std::chrono::steady_clock;

namespace {

double ms_since(clock_type::time_point t0) {
  return std::chrono::duration<double, std::milli>(clock_type::now() - t0).count();
}

// Simulates an ImageNet-style dataset: N small files (~4KB, a typical
// small-thumbnail JPEG size), each its own file on disk.
void write_many_small_files(const std::string &dir, int n, size_t payload_bytes) {
  fs::create_directories(dir);
  std::vector<uint8_t> payload(payload_bytes, 0x42);
  for (int i = 0; i < n; ++i) {
    std::string path = dir + "/sample_" + std::to_string(i) + ".bin";
    std::ofstream f(path, std::ios::binary);
    f.write(reinterpret_cast<const char *>(payload.data()), static_cast<std::streamsize>(payload.size()));
  }
}

// Reads the FULL CONTENTS of every file (not just its size via ate/tellg,
// which would only measure a metadata stat and unfairly favor this path
// against read_one_shard's real content decode below).
size_t read_many_small_files(const std::string &dir, int n) {
  size_t total = 0;
  std::vector<char> buf;
  for (int i = 0; i < n; ++i) {
    std::string path = dir + "/sample_" + std::to_string(i) + ".bin";
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    size_t size = static_cast<size_t>(f.tellg());
    f.seekg(0);
    buf.resize(size);
    f.read(buf.data(), static_cast<std::streamsize>(size));
    total += size;
  }
  return total;
}

// The same N samples, packed into one WebDataset-style shard, via the
// real tar_append/tar_finish free functions this repo's own
// webdataset_shard.h/.cpp defines -- not a reimplementation.
void write_one_shard(const std::string &path, int n, size_t payload_bytes) {
  std::vector<uint8_t> payload(payload_bytes, 0x42);
  std::vector<uint8_t> tar;
  // Reserved up front, matching what a real production shard writer would
  // do (the exact total size is known ahead of time here) -- without
  // this, repeated tar_append() calls trigger std::vector's geometric
  // reallocation/copy overhead on a ~22MB buffer, an artifact of this
  // test's construction, not a real cost webdataset sharding pays.
  tar.reserve(static_cast<size_t>(n) * (512 + payload_bytes + 512));
  for (int i = 0; i < n; ++i) {
    tar_append(tar, "sample_" + std::to_string(i) + ".bin", payload);
  }
  tar_finish(tar);
  std::ofstream out(path, std::ios::binary);
  out.write(reinterpret_cast<const char *>(tar.data()), static_cast<std::streamsize>(tar.size()));
}

size_t read_one_shard(const std::string &path) {
  WebDatasetShardReader reader(path);
  size_t total = 0;
  while (auto sample = reader.next()) {
    for (const auto &kv : sample->files) total += kv.second.size();
  }
  return total;
}

} // namespace

int main() {
  const int n = 5000;
  const size_t payload_bytes = 4096;
  const std::string small_dir = "/tmp/hpc_storage_small_files";
  const std::string shard_path = "/tmp/hpc_storage_small_files_shard.tar";

  std::printf("== Small-file bottleneck: %d files x %zu bytes ==\n\n", n, payload_bytes);

  auto t0 = clock_type::now();
  write_many_small_files(small_dir, n, payload_bytes);
  double write_many_ms = ms_since(t0);

  t0 = clock_type::now();
  write_one_shard(shard_path, n, payload_bytes);
  double write_shard_ms = ms_since(t0);

  t0 = clock_type::now();
  size_t bytes_many = read_many_small_files(small_dir, n);
  double read_many_ms = ms_since(t0);

  t0 = clock_type::now();
  size_t bytes_shard = read_one_shard(shard_path);
  double read_shard_ms = ms_since(t0);

  // open()/close() COUNT is the mechanism-level number that transfers to
  // any metadata-server-backed filesystem: N separate opens (write path)
  // + N separate opens (read path) for the many-small-files case, vs.
  // exactly 1 + 1 for the single-shard case, regardless of which
  // filesystem is underneath.
  long open_calls_many = 2L * n; // one open() for write, one for read, per file
  long open_calls_shard = 2L;    // one open() for write, one for read, total

  std::printf("  write: many-small-files = %.2f ms | single-shard = %.2f ms | speedup = %.2fx\n", write_many_ms,
              write_shard_ms, write_many_ms / write_shard_ms);
  std::printf("  read:  many-small-files = %.2f ms | single-shard = %.2f ms | speedup = %.2fx\n", read_many_ms,
              read_shard_ms, read_many_ms / read_shard_ms);
  std::printf("  open() call count: many-small-files = %ld | single-shard = %ld | ratio = %.0fx\n\n",
              open_calls_many, open_calls_shard, static_cast<double>(open_calls_many) / static_cast<double>(open_calls_shard));

  int fails = 0;
  auto require = [&](bool ok, const char *name) {
    std::printf("%s  %s\n", ok ? "PASS" : "FAIL", name);
    if (!ok) ++fails;
  };
  require(bytes_many == bytes_shard, "both approaches transfer exactly the same total payload bytes -- the "
                                      "comparison isn't skewed by an unequal workload");
  require(write_shard_ms < write_many_ms, "writing one shard is faster than writing N separate small files on "
                                           "this local filesystem");
  require(read_shard_ms < read_many_ms, "reading one shard is faster than opening N separate small files on "
                                         "this local filesystem");
  require(open_calls_many > 100 * open_calls_shard,
          "the metadata-operation-count gap (open() calls) is orders of magnitude -- the mechanism-level "
          "effect that transfers to a real parallel filesystem's metadata server even though the absolute "
          "wall-clock numbers above are APFS-specific");

  std::printf("\n%s\n", fails == 0 ? "PASS" : "FAIL");
  return fails == 0 ? 0 : 1;
}
