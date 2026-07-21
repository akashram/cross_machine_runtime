// gds_bench.cu — GDS direct NVMe->GPU read vs CPU-staged (pread + cudaMemcpy)
// baseline. Writes a scratch checkpoint-sized file, times both paths reading
// it into GPU memory, reports bandwidth and speedup. Requires a GDS-capable
// NVMe device; see README.md for the compatibility-mode fallback if not.
#include "gds_reader.h"

#include <cuda_runtime.h>

#include <fcntl.h>
#include <unistd.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

using namespace distributed_training;

namespace {

constexpr size_t kAlignment = 4096; // O_DIRECT requires aligned buffers/offsets

std::string write_scratch_file(const std::string &path, size_t bytes) {
  int fd = ::open(path.c_str(), O_CREAT | O_WRONLY | O_TRUNC, 0644);
  if (fd < 0) { std::perror("open"); std::exit(1); }
  std::vector<char> chunk(1 << 20, 'x'); // 1MB filler, no O_DIRECT needed for the write
  size_t written = 0;
  while (written < bytes) {
    size_t n = std::min(chunk.size(), bytes - written);
    ssize_t w = ::write(fd, chunk.data(), n);
    if (w <= 0) { std::perror("write"); std::exit(1); }
    written += static_cast<size_t>(w);
  }
  ::close(fd);
  return path;
}

double bench_gds(const std::string &path, size_t bytes) {
  void *dev_ptr = nullptr;
  cudaMalloc(&dev_ptr, bytes);

  GdsFileReader reader(path);
  auto start = std::chrono::steady_clock::now();
  ssize_t n = reader.read(dev_ptr, bytes, 0, 0);
  double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
  cudaFree(dev_ptr);

  if (n != static_cast<ssize_t>(bytes)) {
    std::printf("bench_gds: short read (%zd of %zu bytes)\n", n, bytes);
    return -1.0;
  }
  return (static_cast<double>(bytes) / 1e9) / elapsed; // GB/s
}

double bench_staged(const std::string &path, size_t bytes) {
  int fd = ::open(path.c_str(), O_RDONLY | O_DIRECT);
  if (fd < 0) { std::perror("open"); return -1.0; }

  void *dev_ptr = nullptr;
  cudaMalloc(&dev_ptr, bytes);
  void *pinned = nullptr;
  cudaMallocHost(&pinned, bytes); // page-locked, alignment-satisfying for O_DIRECT

  auto start = std::chrono::steady_clock::now();
  ssize_t n = staged_read(fd, dev_ptr, pinned, bytes, 0);
  double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();

  cudaFreeHost(pinned);
  cudaFree(dev_ptr);
  ::close(fd);

  if (n != static_cast<ssize_t>(bytes)) {
    std::printf("bench_staged: short read (%zd of %zu bytes)\n", n, bytes);
    return -1.0;
  }
  return (static_cast<double>(bytes) / 1e9) / elapsed; // GB/s
}

} // namespace

int main() {
  constexpr size_t kCheckpointBytes = size_t(2) * 1024 * 1024 * 1024; // 2GB, representative shard size
  std::string path = "/tmp/gds_bench_scratch.bin";

  std::printf("writing %zu MB scratch checkpoint...\n", kCheckpointBytes / (1024 * 1024));
  write_scratch_file(path, kCheckpointBytes);

  GdsDriverGuard driver; // brackets all cuFile usage in this process

  double staged_gbps = bench_staged(path, kCheckpointBytes);
  double gds_gbps = bench_gds(path, kCheckpointBytes);

  std::printf("staged (pread + cudaMemcpy H2D): %.2f GB/s\n", staged_gbps);
  std::printf("GDS (cuFileRead direct):         %.2f GB/s\n", gds_gbps);
  if (staged_gbps > 0 && gds_gbps > 0) {
    std::printf("speedup: %.2fx\n", gds_gbps / staged_gbps);
  }

  ::unlink(path.c_str());
  return 0;
}
