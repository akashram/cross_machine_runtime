#pragma once

// GPUDirect Storage (GDS) checkpoint loader
// =========================================================================
//
// PLAN.md Phase 6 step 2: direct NVMe -> GPU checkpoint loading via cuFile,
// measured against a CPU-staged baseline (read into pinned host memory,
// then cudaMemcpy H2D).
//
// GDS's entire value proposition is bypassing the CPU: the cuFile driver
// programs the NVMe controller's DMA engine to write straight into GPU
// HBM over PCIe (or NVLink-C2C on Grace Hopper), so the transfer never
// touches a host-memory bounce buffer or a CPU core. That means, unlike
// step 1's data loader, there is no meaningful CPU-only portable version
// of this component to build and run on a Mac — the thing being measured
// (DMA path bypassing the CPU) doesn't exist without a GDS-capable NVMe
// device and driver stack. This header is real, complete cuFile API usage
// (code-complete, matching Phase 3's convention for GPU code that hasn't
// run yet), not a placeholder.
//
// Requires: libcufile (bundled with CUDA 11.4+ toolkit installs that
// include GDS), a GDS-compatible NVMe device, and either GDS's kernel
// driver (nvidia-fs) or its compatibility mode (bounces through pinned
// host memory but keeps the same API — useful for correctness testing on
// hardware without full GDS support).
//
// =========================================================================

#include <cufile.h>
#include <cuda_runtime.h>

#include <fcntl.h>
#include <unistd.h>

#include <cstddef>
#include <stdexcept>
#include <string>

namespace distributed_training {

// One registered GDS file handle. Construction opens the file with
// O_DIRECT (required — GDS bypasses the page cache) and registers it with
// the cuFile driver; destruction unregisters and closes.
class GdsFileReader {
public:
  explicit GdsFileReader(const std::string &path) {
    fd_ = ::open(path.c_str(), O_RDONLY | O_DIRECT);
    if (fd_ < 0) throw std::runtime_error("GdsFileReader: open failed for " + path);

    CUfileDescr_t descr{};
    descr.handle.fd = fd_;
    descr.type = CU_FILE_HANDLE_TYPE_OPAQUE_FD;

    CUfileError_t status = cuFileHandleRegister(&handle_, &descr);
    if (status.err != CU_FILE_SUCCESS) {
      ::close(fd_);
      throw std::runtime_error("GdsFileReader: cuFileHandleRegister failed, err=" +
                                std::to_string(status.err));
    }
  }

  ~GdsFileReader() {
    if (fd_ >= 0) {
      cuFileHandleDeregister(handle_);
      ::close(fd_);
    }
  }

  GdsFileReader(const GdsFileReader &) = delete;
  GdsFileReader &operator=(const GdsFileReader &) = delete;

  // Reads `size` bytes starting at `file_offset` directly into GPU device
  // memory at `dev_ptr` (must already be cudaMalloc'd — cuFile does not
  // allocate). Returns bytes actually read (may be < size at EOF), or -1
  // on error.
  ssize_t read(void *dev_ptr, size_t size, off_t file_offset, off_t dev_offset = 0) const {
    return cuFileRead(handle_, dev_ptr, size, file_offset, dev_offset);
  }

private:
  int fd_ = -1;
  CUfileHandle_t handle_{};
};

// Driver lifetime guard: cuFileDriverOpen/Close must bracket all
// GdsFileReader usage in the process. One instance, held for the training
// job's lifetime (typically in main()).
class GdsDriverGuard {
public:
  GdsDriverGuard() {
    CUfileError_t status = cuFileDriverOpen();
    if (status.err != CU_FILE_SUCCESS) {
      throw std::runtime_error("GdsDriverGuard: cuFileDriverOpen failed, err=" +
                                std::to_string(status.err));
    }
  }
  ~GdsDriverGuard() { cuFileDriverClose(); }

  GdsDriverGuard(const GdsDriverGuard &) = delete;
  GdsDriverGuard &operator=(const GdsDriverGuard &) = delete;
};

// CPU-staged baseline for comparison: pread() into pinned host memory,
// then cudaMemcpy H2D. This is the path GDS is meant to beat — every byte
// crosses the CPU twice (once into the pinned buffer, once again as the
// memcpy source read) instead of zero times.
inline ssize_t staged_read(int fd, void *dev_ptr, void *pinned_host_buf, size_t size,
                            off_t file_offset) {
  ssize_t n = ::pread(fd, pinned_host_buf, size, file_offset);
  if (n <= 0) return n;
  cudaError_t err = cudaMemcpy(dev_ptr, pinned_host_buf, static_cast<size_t>(n), cudaMemcpyHostToDevice);
  if (err != cudaSuccess) return -1;
  return n;
}

} // namespace distributed_training
