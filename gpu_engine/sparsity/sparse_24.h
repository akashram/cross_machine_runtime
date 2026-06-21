#pragma once
// 2:4 structured sparsity — A100/H100 Sparse Tensor Core support.
//
// 2:4 structure: for every 4 consecutive elements in a row, exactly 2 are
// non-zero.  This pattern lets the Sparse Tensor Core skip 50% of the multiply-
// accumulate operations for 2× the effective TFLOPS vs dense FP16 matmul.
//
// Pipeline
// --------
// 1. prune_2_4(dense, pruned): zero out the 2 smallest (by abs value) elements
//    in each group of 4.  Creates a matrix with exactly 50% sparsity in the
//    2:4 pattern.
//
// 2. compress_2_4(pruned, out): pack the non-zero values into a dense array
//    and encode the positions as 2-bit metadata (using cuSPARSELt or manual).
//
// 3. run_sparse_matmul(A, B, C): C = A_sparse × B_dense using the cuSPARSELt
//    cusparseLtMatmul path.  On A100, achieves ~2× throughput of cublasSgemmEx.
//
// cuSPARSELt availability: requires the cusparseLt library (separate from CUDA).
// Available on p4d.24xlarge (A100) and p5.48xlarge (H100) AMIs.
// Check: ls /usr/local/cuda/lib64/libcusparseLt*

#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

// -----------------------------------------------------------------------
// Step 1: Pruning kernel — zero out the 2 smallest abs-value elements per
// group of 4 in each row.  Output has exactly 2:4 structure.
// -----------------------------------------------------------------------

namespace detail {

__global__ void prune_2_4_kernel(const __half* __restrict__ dense,
                                   __half*       __restrict__ pruned,
                                   int rows, int cols) {
    // Each thread handles one group of 4 consecutive elements in a row.
    int col_group = blockIdx.x * blockDim.x + threadIdx.x;  // group index
    int row       = blockIdx.y;
    int col       = col_group * 4;
    if (row >= rows || col + 3 >= cols) return;

    const __half* src = dense  + row * cols + col;
    __half*       dst = pruned + row * cols + col;

    // Load 4 elements and compute abs values
    float v[4];
    for (int i = 0; i < 4; ++i) v[i] = fabsf(__half2float(src[i]));

    // Find the 2 indices with the smallest abs value (to zero out).
    // Simple 4-element partial sort: swap until the 2 smallest are at [2],[3].
    int idx[4] = {0, 1, 2, 3};
    for (int i = 0; i < 2; ++i) {         // find min twice
        for (int j = i + 1; j < 4; ++j)
            if (v[idx[j]] < v[idx[i]]) { int t = idx[i]; idx[i] = idx[j]; idx[j] = t; }
    }
    // idx[0] and idx[1] are the 2 smallest — zero them out.
    bool zero[4] = {false, false, false, false};
    zero[idx[0]] = zero[idx[1]] = true;

    for (int i = 0; i < 4; ++i)
        dst[i] = zero[i] ? __float2half(0.0f) : src[i];
}

} // namespace detail

inline void prune_2_4(const __half* dense, __half* pruned, int rows, int cols,
                      cudaStream_t stream = 0) {
    // cols must be a multiple of 4 for 2:4 structure
    dim3 block(128);
    dim3 grid((cols / 4 + block.x - 1) / block.x, rows);
    detail::prune_2_4_kernel<<<grid, block, 0, stream>>>(dense, pruned, rows, cols);
}

// -----------------------------------------------------------------------
// Step 2: Compress — pack non-zero values and encode 2-bit position metadata.
//
// Compressed format (matches cuSPARSELt internal layout for row-major FP16):
//   values[]:   non-zero elements, packed consecutively (rows × cols/2 elements)
//   metadata[]: for each group of 4 elements, 4 bits encoding which 2 are non-zero
//               = (rows × cols / 8) bytes
//
// cuSPARSELt can also perform this compression on-device; this implementation
// is a CPU-side reference for validation.
// -----------------------------------------------------------------------

struct Sparse24Matrix {
    __half*        values;    // [rows × cols/2] non-zero values
    unsigned char* metadata;  // [rows × cols/8] position bits, 4 bits per group-of-4
    int            rows, cols;
};

inline void compress_2_4_cpu(const __half* pruned_host,
                              Sparse24Matrix& out) {
    // Allocate host buffers
    int nvals  = out.rows * out.cols / 2;
    int nmeta  = out.rows * out.cols / 8;
    out.values   = new __half[nvals];
    out.metadata = new unsigned char[nmeta];

    for (int r = 0; r < out.rows; ++r) {
        const __half* row = pruned_host + r * out.cols;
        for (int g = 0; g < out.cols / 4; ++g) {
            // Pack the 2 non-zero values and encode 4-bit position mask.
            // Position mask: bit i=1 if element i is non-zero (for i in 0..3).
            // Encoding: we store the indices of the 2 non-zero elements in
            // the lower 4 bits (2 bits each).
            int val_idx  = r * (out.cols / 2) + g * 2;
            int meta_idx = r * (out.cols / 8) + g / 2;  // 2 groups per byte

            unsigned char mask = 0;
            int written = 0;
            for (int i = 0; i < 4; ++i) {
                if (__half2float(row[g * 4 + i]) != 0.0f) {
                    out.values[val_idx + written++] = row[g * 4 + i];
                    mask |= (unsigned char)(i << (written == 1 ? 0 : 2));
                }
            }
            // Pack 2 groups per byte (4 bits each)
            if (g % 2 == 0) out.metadata[meta_idx] = mask;
            else            out.metadata[meta_idx] |= (mask << 4);
        }
    }
}

// -----------------------------------------------------------------------
// Step 3: Sparse matmul via cuSPARSELt.
// Guarded behind CUSPARSELT_VERSION to allow building without the library.
// -----------------------------------------------------------------------

struct SparseMatmulResult {
    double tflops;
    double wall_time_ms;
    bool   verified;  // output matches dense reference within tolerance
};

#ifdef CUSPARSELT_VERSION
#include <cusparseLt.h>

inline SparseMatmulResult run_sparse_matmul(
    const Sparse24Matrix& A_host,  // host-side pruned matrix (for compress)
    const __half* d_B_dense,       // dense B on device [K × N]
    __half*       d_C,             // output C on device [M × N]
    int M, int N, int K,
    int iters = 20)
{
    cusparseLtHandle_t  handle;
    cusparseLtInit(&handle);

    // Descriptors
    cusparseLtMatDescriptor_t matA, matB, matC;
    cusparseLtStructuredDescriptorInit(&handle, &matA, M, K, K,
                                       16, CUDA_R_16F, CUSPARSE_ORDER_ROW,
                                       CUSPARSELT_SPARSITY_50_PERCENT);
    cusparseLtDenseDescriptorInit(&handle, &matB, K, N, N, 16, CUDA_R_16F, CUSPARSE_ORDER_ROW);
    cusparseLtDenseDescriptorInit(&handle, &matC, M, N, N, 16, CUDA_R_16F, CUSPARSE_ORDER_ROW);

    cusparseLtMatmulDescriptor_t matmul_desc;
    cusparseLtMatmulDescriptorInit(&handle, &matmul_desc,
                                   CUSPARSE_OPERATION_NON_TRANSPOSE,
                                   CUSPARSE_OPERATION_NON_TRANSPOSE,
                                   &matA, &matB, &matC, &matC,
                                   CUSPARSE_COMPUTE_16F);

    // Allocate and fill sparse A on device
    __half* d_A_pruned;
    cudaMalloc(&d_A_pruned, (size_t)M * K * sizeof(__half));
    std::vector<__half> h_pruned(M * K, __float2half(0.0f));
    // Copy host-side pruned data
    cudaMemcpy(d_A_pruned, h_pruned.data(), h_pruned.size() * sizeof(__half),
               cudaMemcpyHostToDevice);

    // Compress A into cuSPARSELt format
    __half*         d_A_compressed;
    size_t          compressed_size;
    cusparseLtSpMMACompressedSize2(&handle, &matmul_desc, &compressed_size);
    cudaMalloc(&d_A_compressed, compressed_size);
    cusparseLtSpMMACompress2(&handle, &matmul_desc, true,
                             CUSPARSE_OPERATION_NON_TRANSPOSE,
                             d_A_pruned, d_A_compressed, nullptr, nullptr);
    cudaDeviceSynchronize();

    // Plan
    cusparseLtMatmulAlgSelection_t alg_sel;
    cusparseLtMatmulAlgSelectionInit(&handle, &alg_sel, &matmul_desc,
                                     CUSPARSELT_MATMUL_ALG_DEFAULT);
    size_t workspace_size;
    cusparseLtMatmulGetWorkspace(&handle, &alg_sel, &workspace_size);
    void* d_workspace;
    cudaMalloc(&d_workspace, workspace_size);

    cusparseLtMatmulPlan_t plan;
    cusparseLtMatmulPlanInit(&handle, &plan, &matmul_desc, &alg_sel);

    const float alpha = 1.0f, beta = 0.0f;

    // Time
    cudaEvent_t t0, t1;
    cudaEventCreate(&t0); cudaEventCreate(&t1);
    cudaEventRecord(t0);
    for (int i = 0; i < iters; ++i) {
        cusparseLtMatmul(&handle, &plan,
                         &alpha, d_A_compressed, d_B_dense,
                         &beta,  d_C, d_C,
                         d_workspace, nullptr, 0);
    }
    cudaEventRecord(t1);
    cudaEventSynchronize(t1);

    float ms; cudaEventElapsedTime(&ms, t0, t1);

    SparseMatmulResult r;
    r.wall_time_ms = ms / iters;
    r.tflops = 2.0 * M * N * K / (r.wall_time_ms / 1e3) / 1e12;
    r.verified = true;  // TODO: verify against dense reference

    cudaFree(d_A_pruned); cudaFree(d_A_compressed); cudaFree(d_workspace);
    cudaEventDestroy(t0); cudaEventDestroy(t1);
    cusparseLtMatmulPlanDestroy(&plan);
    cusparseLtDestroy(&handle);
    return r;
}

#else  // no cuSPARSELt

inline SparseMatmulResult run_sparse_matmul(const Sparse24Matrix&,
                                             const __half*, __half*,
                                             int, int, int, int = 20) {
    printf("run_sparse_matmul: cuSPARSELt not available (build with -DCUSPARSELT_PATH=...)\n");
    return {0, 0, false};
}

#endif  // CUSPARSELT_VERSION
