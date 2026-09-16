//===- io_pattern.h - I/O pattern characterization primitives -----------===//
//
// PLAN.md Phase 21 step 1: characterize the REAL I/O pattern
// `distributed_training/data_loading` and `distributed_training/checkpoint`
// actually produce, as a real measurement (not an assumed profile), to
// ground every later hpc_storage/ step (the small-file study in step 2, the
// thundering-herd capacity model in step 6) in something this repo actually
// does rather than a guessed access shape.
//
// Method, disclosed honestly (see io_pattern_test.cpp and this step's
// README for the full account): this measures at the APPLICATION-REQUEST
// level (call counts, byte sizes, and inter-arrival timing of the real
// public APIs in data_loader.h/webdataset_shard.h/sharded_checkpoint.h,
// called unmodified), not via an OS-level syscall trace (`fs_usage`/dtrace
// need elevated privileges not available in this sandboxed environment).
// One exception: `raw_tar_block_scan()` below directly replicates
// webdataset's documented on-disk block layout (512-byte USTAR blocks,
// read via raw pread()) to get a genuine syscall-level block-size number
// for the read path -- this is a replication of a known, source-verified
// format, not a black-box trace, and is called out as such.
//
//===----------------------------------------------------------------------===//
#pragma once

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <fcntl.h>
#include <string>
#include <unistd.h>
#include <vector>

namespace hpc_storage {

// One observed application-level I/O request.
struct IoEvent {
  bool is_write;
  size_t bytes;
  double t_ns; // time of the call, relative to the harness's own start
};

struct IoPatternSummary {
  size_t num_requests = 0;
  size_t total_bytes = 0;
  double mean_bytes = 0.0;
  double min_bytes = 0.0;
  double max_bytes = 0.0;
  // "sequential fraction": for the read path, the fraction of requests
  // whose source file offset was un-broken from the previous request on
  // the same file (i.e. no seek) -- 1.0 means purely sequential streaming.
  double sequential_fraction = 0.0;
};

inline IoPatternSummary summarize(const std::vector<IoEvent> &events) {
  IoPatternSummary s;
  s.num_requests = events.size();
  if (events.empty()) return s;
  s.min_bytes = static_cast<double>(events.front().bytes);
  s.max_bytes = s.min_bytes;
  for (const auto &e : events) {
    s.total_bytes += e.bytes;
    s.min_bytes = std::min(s.min_bytes, static_cast<double>(e.bytes));
    s.max_bytes = std::max(s.max_bytes, static_cast<double>(e.bytes));
  }
  s.mean_bytes = static_cast<double>(s.total_bytes) / static_cast<double>(s.num_requests);
  return s;
}

// Replicates webdataset's real on-disk block layout: a USTAR archive is a
// sequence of 512-byte blocks (a header block, then ceil(size/512) data
// blocks, per entry). This function opens the shard file directly with
// O_RDONLY and issues one pread() per 512-byte block, in file order --
// exactly the syscall pattern a stdio-buffered TarReader collapses into
// fewer, larger read() calls at the C library's buffer granularity, but
// exactly the LOGICAL block structure the format itself defines. Returns
// the number of 512-byte block reads issued and confirms they cover the
// whole file (a correctness check that this really is replicating the
// format, not guessing at it).
struct RawBlockScanResult {
  size_t num_blocks = 0;
  size_t bytes_covered = 0;
  size_t file_size = 0;
};

inline RawBlockScanResult raw_tar_block_scan(const std::string &path) {
  constexpr size_t kBlock = 512;
  RawBlockScanResult r;
  int fd = ::open(path.c_str(), O_RDONLY);
  if (fd < 0) return r;
  off_t size = ::lseek(fd, 0, SEEK_END);
  ::lseek(fd, 0, SEEK_SET);
  r.file_size = static_cast<size_t>(size);

  std::vector<uint8_t> block(kBlock);
  ssize_t n;
  size_t offset = 0;
  while ((n = ::pread(fd, block.data(), kBlock, static_cast<off_t>(offset))) > 0) {
    ++r.num_blocks;
    r.bytes_covered += static_cast<size_t>(n);
    offset += static_cast<size_t>(n);
    if (static_cast<size_t>(n) < kBlock) break; // final partial/short block at EOF
  }
  ::close(fd);
  return r;
}

using clock_type = std::chrono::steady_clock;
inline double now_ns(clock_type::time_point t0) {
  return static_cast<double>(std::chrono::duration_cast<std::chrono::nanoseconds>(clock_type::now() - t0).count());
}

} // namespace hpc_storage
