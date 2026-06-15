#pragma once

// GPU Memory Allocators — RAII wrappers for CUDA device and pinned host memory
// =========================================================================
//
// Two types:
//
//   DeviceBuffer  — device-side allocation via cudaMalloc/cudaFree.
//                   Lives in GPU global memory (HBM on T4/A100, GDDR6 on
//                   other cards). The CPU cannot dereference .get() directly.
//
//   PinnedBuffer  — page-locked host allocation via cudaMallocHost/cudaFreeHost.
//                   Lives in CPU DRAM but the OS is told never to page it out.
//                   The DMA engine (PCIe transfer unit) can address it directly,
//                   bypassing the bounce buffer the driver would otherwise need
//                   when copying from regular pageable memory.
//
//
// Why not just use cudaMalloc directly everywhere?
// -------------------------------------------------
// cudaMalloc latency is ~8–15 µs per call (driver round-trip to the CUDA
// context). For kernels that need dynamic allocation on the hot path this is
// a problem. These RAII types are building blocks for the arena allocator
// (Step 3 / streams): allocate large DeviceBuffers once at startup, then
// sub-allocate from them without touching the driver.
//
//
// Lifetime hazard
// ---------------
// DeviceBuffer and PinnedBuffer must be destroyed BEFORE the CUDA context is
// destroyed. Static-duration instances are therefore risky (the context may
// be torn down before destructors run). For now, always hold these on the
// stack or inside objects whose lifetime is explicitly managed. A context
// lifecycle guard will be added in the stream manager (Step 3).

#include <cuda_runtime.h>
#include <cstddef>
#include <utility>

namespace gpu_engine {

class DeviceBuffer {
public:
    DeviceBuffer() = default;

    explicit DeviceBuffer(std::size_t bytes, int device = 0) : bytes_(bytes) {
        if (bytes_ > 0) {
            cudaSetDevice(device);
            if (cudaMalloc(&ptr_, bytes_) != cudaSuccess) {
                ptr_ = nullptr;
                bytes_ = 0;
            }
        }
    }

    ~DeviceBuffer() { reset(); }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    DeviceBuffer(DeviceBuffer&& o) noexcept
        : ptr_(o.ptr_), bytes_(o.bytes_) {
        o.ptr_ = nullptr; o.bytes_ = 0;
    }
    DeviceBuffer& operator=(DeviceBuffer&& o) noexcept {
        if (this != &o) {
            reset();
            ptr_ = o.ptr_; bytes_ = o.bytes_;
            o.ptr_ = nullptr; o.bytes_ = 0;
        }
        return *this;
    }

    void reset() noexcept {
        if (ptr_) { cudaFree(ptr_); ptr_ = nullptr; bytes_ = 0; }
    }

    void*       get()   const noexcept { return ptr_; }
    std::size_t size()  const noexcept { return bytes_; }
    bool        valid() const noexcept { return ptr_ != nullptr; }

    template<typename T>
    T* as() const noexcept { return static_cast<T*>(ptr_); }

private:
    void*       ptr_{nullptr};
    std::size_t bytes_{0};
};


class PinnedBuffer {
public:
    PinnedBuffer() = default;

    explicit PinnedBuffer(std::size_t bytes) : bytes_(bytes) {
        if (bytes_ > 0) {
            if (cudaMallocHost(&ptr_, bytes_) != cudaSuccess) {
                ptr_ = nullptr;
                bytes_ = 0;
            }
        }
    }

    ~PinnedBuffer() { reset(); }

    PinnedBuffer(const PinnedBuffer&) = delete;
    PinnedBuffer& operator=(const PinnedBuffer&) = delete;

    PinnedBuffer(PinnedBuffer&& o) noexcept
        : ptr_(o.ptr_), bytes_(o.bytes_) {
        o.ptr_ = nullptr; o.bytes_ = 0;
    }
    PinnedBuffer& operator=(PinnedBuffer&& o) noexcept {
        if (this != &o) {
            reset();
            ptr_ = o.ptr_; bytes_ = o.bytes_;
            o.ptr_ = nullptr; o.bytes_ = 0;
        }
        return *this;
    }

    void reset() noexcept {
        if (ptr_) { cudaFreeHost(ptr_); ptr_ = nullptr; bytes_ = 0; }
    }

    void*       get()   const noexcept { return ptr_; }
    std::size_t size()  const noexcept { return bytes_; }
    bool        valid() const noexcept { return ptr_ != nullptr; }

    template<typename T>
    T* as() const noexcept { return static_cast<T*>(ptr_); }

private:
    void*       ptr_{nullptr};
    std::size_t bytes_{0};
};

} // namespace gpu_engine
