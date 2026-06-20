#pragma once
// Hopper TMA (Tensor Memory Accelerator) stubs
// TODO: implement on H100 hardware

#if defined(__CUDACC__) && __CUDA_ARCH__ >= 900
#include <cuda/barrier>
#include <cuda/cooperative_groups.h>

// Create a TMA descriptor for a 2D FP16 tensor.
// shape: [rows, cols], stride_bytes: bytes between rows
CUtensorMap make_tma_descriptor_2d_fp16(const void* ptr,
                                          uint64_t rows, uint64_t cols,
                                          uint64_t stride_bytes);

// Async TMA copy: HBM → shared memory using TMA descriptor.
// barrier: cuda::barrier for arrival counting
__device__ void tma_load_async(const CUtensorMap* desc,
                                 void* smem_dst,
                                 uint64_t coord_x, uint64_t coord_y,
                                 cuda::barrier<cuda::thread_scope_block>& barrier);

#else
// Stub definitions for non-Hopper compilation
struct CUtensorMap {};
inline CUtensorMap make_tma_descriptor_2d_fp16(...) { return {}; }
#endif
