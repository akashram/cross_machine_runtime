#pragma once
// Hopper TMA (Tensor Memory Accelerator) — requires sm_90 (H100) and CUDA 12.0+.
//
// TMA enables asynchronous bulk tensor copies from HBM → shared memory using a
// pre-built tensor map descriptor.  The hardware handles stride computation,
// swizzle, and out-of-bounds fill automatically — the kernel only specifies
// which tile to fetch.
//
// Key advantage over cudaMemcpyAsync or cp.async:
//   - One initiation instruction moves an entire tile (up to 256 KB).
//   - The SM does not stall; execution continues while DMA runs.
//   - Eliminates the per-element address calculation that cp.async requires.
//
// API
// ---
// Host:
//   CUtensorMap tmap = make_tma_descriptor_2d_fp16(ptr, rows, cols, stride_bytes);
//   // tmap is small (<= 128 bytes); pass as __constant__ or kernel param.
//
// Device (sm_90 only):
//   // One elected thread initiates; all threads wait on the barrier.
//   tma_load_async(&tmap, smem_ptr, col_coord, row_coord, barrier);
//   barrier.arrive_and_wait();
//   // smem_ptr now contains the tile.

#include <cuda_runtime.h>
#include <cuda_fp16.h>

// TMA requires the CUDA driver API for descriptor creation.
#if defined(__CUDACC__) && __CUDA_ARCH__ >= 900
#   include <cuda.h>           // CUtensorMap, cuTensorMapEncodeTiled
#   include <cuda/barrier>     // cuda::barrier
#endif

// -----------------------------------------------------------------------
// Host-side: build a TMA descriptor for a 2D FP16 row-major tensor.
//
// Parameters:
//   ptr          — base pointer of the tensor in device HBM
//   rows, cols   — logical dimensions of the full tensor
//   stride_bytes — bytes between the start of consecutive rows (≥ cols*2)
//   tile_cols    — width  of the tile this descriptor will load (e.g. 16)
//   tile_rows    — height of the tile (e.g. 8)  — both must be ≥ 1
//
// The tile dimensions select how much data each tma_load_async transfers.
// For GEMM: tile_cols = K_block, tile_rows = M_block.
// -----------------------------------------------------------------------

#if defined(__CUDACC__) && __CUDA_ARCH__ >= 900

inline CUtensorMap make_tma_descriptor_2d_fp16(
    const void* ptr,
    uint64_t rows, uint64_t cols, uint64_t stride_bytes,
    uint32_t tile_rows = 8, uint32_t tile_cols = 16)
{
    CUtensorMap tmap;

    uint64_t global_dim[2]     = {cols, rows};       // [fast=col, slow=row]
    uint64_t global_stride[1]  = {stride_bytes};     // stride for the slow dim
    uint32_t box_dim[2]        = {tile_cols, tile_rows};
    uint32_t element_stride[2] = {1, 1};             // contiguous (no interleave)

    cuTensorMapEncodeTiled(
        &tmap,
        CU_TENSOR_MAP_DATA_TYPE_FLOAT16,
        2,                                   // tensor rank
        const_cast<void*>(ptr),
        global_dim,
        global_stride,                       // NULL would mean tightly packed rows
        box_dim,
        element_stride,
        CU_TENSOR_MAP_INTERLEAVE_NONE,
        CU_TENSOR_MAP_SWIZZLE_NONE,          // no bank-conflict swizzle (add if needed)
        CU_TENSOR_MAP_L2_PROMOTION_NONE,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
    );

    return tmap;
}

// -----------------------------------------------------------------------
// Device-side: initiate an async TMA tile load from HBM to shared memory.
//
// MUST be called by exactly ONE thread per block (typically thread 0).
// All threads then call barrier.arrive_and_wait() to sync.
//
// coord_col, coord_row — coordinates in element units within the full tensor.
// The transaction size is fixed by the box_dim in the descriptor.
// -----------------------------------------------------------------------

__device__ __forceinline__
void tma_load_async(const CUtensorMap*                              desc,
                    void*                                            smem_dst,
                    uint64_t                                         coord_col,
                    uint64_t                                         coord_row,
                    cuda::barrier<cuda::thread_scope_block>&         barrier)
{
    // The TMA hardware decrements the barrier's expected-tx-count by the
    // number of bytes transferred, so the barrier was initialised with
    // the expected byte count (not a thread count) by the caller.
    using cuda::device::cp_async_bulk_tensor_2d_global_shared_cta;
    cp_async_bulk_tensor_2d_global_shared_cta(
        smem_dst,
        desc,
        {coord_col, coord_row},
        barrier);
}

// Convenience: elect thread 0 to initiate, block until done.
// smem_dst must be 16-byte aligned.  barrier must be initialised with
// the expected byte count (tile_rows * tile_cols * sizeof(__half)).
__device__ __forceinline__
void tma_load_sync(const CUtensorMap*                          desc,
                   void*                                        smem_dst,
                   uint64_t                                     coord_col,
                   uint64_t                                     coord_row,
                   cuda::barrier<cuda::thread_scope_block>&     barrier)
{
    if (threadIdx.x == 0)
        tma_load_async(desc, smem_dst, coord_col, coord_row, barrier);
    barrier.arrive_and_wait();
}

#else  // non-Hopper fallback (host or older arch)

struct CUtensorMap {};

inline CUtensorMap make_tma_descriptor_2d_fp16(
    const void*, uint64_t, uint64_t, uint64_t, uint32_t = 8, uint32_t = 16) {
    return {};
}

#endif  // __CUDA_ARCH__ >= 900

// -----------------------------------------------------------------------
// Bandwidth benchmark kernel: TMA vs cudaMemcpyAsync
// (actual comparison done in hopper_bench.cu after timing both paths)
// -----------------------------------------------------------------------

#if defined(__CUDACC__) && __CUDA_ARCH__ >= 900

// Read a tile via TMA and write to output (to prevent dead-code elimination)
template<int TILE_ROWS, int TILE_COLS>
__global__ void tma_bandwidth_kernel(
    const CUtensorMap* __restrict__ desc,
    __half*            __restrict__ out,
    int num_tiles_col, int num_tiles_row)
{
    constexpr size_t TILE_BYTES = TILE_ROWS * TILE_COLS * sizeof(__half);
    __shared__ __half smem[TILE_ROWS][TILE_COLS];

    // Each block loads one tile
    int tile_col = blockIdx.x;
    int tile_row = blockIdx.y;

    __shared__ cuda::barrier<cuda::thread_scope_block> bar;
    if (threadIdx.x == 0) {
        init(&bar, TILE_BYTES);
        tma_load_async(desc,
                       smem,
                       (uint64_t)tile_col * TILE_COLS,
                       (uint64_t)tile_row * TILE_ROWS,
                       bar);
    }
    bar.arrive_and_wait();

    // Copy smem → out to prevent optimisation
    int tid = threadIdx.x;
    int out_offset = (tile_row * TILE_ROWS) * num_tiles_col * TILE_COLS
                   +  tile_col * TILE_COLS;
    for (int i = tid; i < TILE_ROWS * TILE_COLS; i += blockDim.x)
        out[out_offset + (i / TILE_COLS) * num_tiles_col * TILE_COLS + (i % TILE_COLS)] =
            smem[i / TILE_COLS][i % TILE_COLS];
}

#endif
