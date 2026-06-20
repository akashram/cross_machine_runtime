#pragma once
#include <cstddef>
// TODO: implement on A100/H100 with cuSPARSELt

// Prune a dense FP16 weight matrix to 2:4 structure in-place.
// For each group of 4 consecutive elements in a row, zero out the 2 smallest (by abs value).
void prune_2_4(const __half* dense, __half* pruned, int rows, int cols);

// Pack a pruned FP16 matrix into cuSPARSELt compressed format.
struct Sparse24Matrix {
    __half*         values;     // non-zero values (rows * cols / 2)
    unsigned char*  metadata;   // 2-bit indices (rows * cols / 8 bytes)
    int             rows, cols;
};

void compress_2_4(const __half* pruned, Sparse24Matrix& out);

struct SparseMatmulResult {
    double tflops;
    double wall_time_ms;
};

// Run cusparseLtMatmul: C = A_sparse * B_dense
SparseMatmulResult run_sparse_matmul(const Sparse24Matrix& A,
                                      const __half* B, __half* C,
                                      int M, int N, int K);
