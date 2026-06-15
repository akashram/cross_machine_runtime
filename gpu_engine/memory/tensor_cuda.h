#pragma once

// CUDA factory functions for TensorHandle
// =========================================================================
//
// Extends foundation::TensorHandle with two GPU-aware factories:
//
//   empty_cuda(shape, dtype, device_index)
//     Allocates in GPU global memory via cudaMalloc. device() returns kCUDA.
//     The CPU cannot dereference .data() — any attempt is undefined behaviour.
//     Buffer is freed via cudaFree when the last handle is destroyed.
//
//   empty_pinned(shape, dtype)
//     Allocates page-locked host memory via cudaMallocHost. device() returns
//     kCPU — the data is accessible from the CPU — but the allocation is
//     pinned so the PCIe DMA engine can transfer it without a staging copy.
//     Use this for tensors that are the source or destination of H2D/D2H
//     transfers on the hot path.
//
//
// Why keep this separate from foundation/tensor/tensor.h?
// --------------------------------------------------------
// foundation/ must build without a CUDA toolkit. Putting cudaMalloc calls
// in tensor.h would force a CUDA dependency on every component that includes
// it (the CPU engine, the compiler, tests). The factories here live in
// gpu_engine/ which is only compiled when CMAKE_CUDA_COMPILER is set.
//
//
// Ownership model
// ---------------
// Same as the CPU factories: std::shared_ptr<void> with a custom deleter
// (cudaFree or cudaFreeHost). Copying a handle shares the buffer; the last
// owner frees it. See gpu_alloc.h for the lifetime hazard note regarding
// CUDA context teardown order.

#include <cuda_runtime.h>
#include "foundation/tensor/tensor.h"
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <vector>

namespace gpu_engine {

namespace detail {

inline int64_t numel(std::span<const int64_t> shape) noexcept {
    int64_t n = 1;
    for (auto d : shape) n *= d;
    return n;
}

inline std::vector<int64_t> contiguous_strides(
        std::span<const int64_t> shape,
        foundation::Dtype dtype) noexcept {
    int n = static_cast<int>(shape.size());
    std::vector<int64_t> s(static_cast<std::size_t>(n));
    if (n == 0) return s;
    s[static_cast<std::size_t>(n - 1)] =
        static_cast<int64_t>(foundation::dtype_size(dtype));
    for (int i = n - 2; i >= 0; --i)
        s[static_cast<std::size_t>(i)] =
            s[static_cast<std::size_t>(i + 1)] * shape[static_cast<std::size_t>(i + 1)];
    return s;
}

} // namespace detail


inline foundation::TensorHandle empty_cuda(
        std::span<const int64_t> shape,
        foundation::Dtype dtype,
        int device_index = 0) noexcept {
    cudaSetDevice(device_index);
    std::size_t bytes = static_cast<std::size_t>(detail::numel(shape))
                        * foundation::dtype_size(dtype);
    void* raw = nullptr;
    if (cudaMalloc(&raw, bytes == 0 ? 1 : bytes) != cudaSuccess) return {};
    auto buf = std::shared_ptr<void>(raw, [](void* p){ cudaFree(p); });
    foundation::Device dev{foundation::DeviceType::kCUDA, device_index};
    return foundation::TensorHandle::from_buffer(
        std::move(buf), raw, shape,
        detail::contiguous_strides(shape, dtype),
        dtype, dev);
}


inline foundation::TensorHandle empty_pinned(
        std::span<const int64_t> shape,
        foundation::Dtype dtype) noexcept {
    std::size_t bytes = static_cast<std::size_t>(detail::numel(shape))
                        * foundation::dtype_size(dtype);
    void* raw = nullptr;
    if (cudaMallocHost(&raw, bytes == 0 ? 1 : bytes) != cudaSuccess) return {};
    auto buf = std::shared_ptr<void>(raw, [](void* p){ cudaFreeHost(p); });
    return foundation::TensorHandle::from_buffer(
        std::move(buf), raw, shape,
        detail::contiguous_strides(shape, dtype),
        dtype, foundation::kCPU);
}

} // namespace gpu_engine
