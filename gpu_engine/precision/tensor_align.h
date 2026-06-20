#pragma once
#include <cstdio>

enum class DType { FP16, BF16, TF32, FP32, FP8_E4M3, FP8_E5M2 };

// Returns the required alignment (multiple-of) for Tensor Core GEMM.
inline int tensor_core_alignment(DType dtype) {
    switch (dtype) {
        case DType::FP8_E4M3: return 16;
        case DType::FP8_E5M2: return 16;
        case DType::FP16:     return 16;
        case DType::BF16:     return 16;
        case DType::TF32:     return 8;
        case DType::FP32:     return 1;  // scalar path, no Tensor Cores
    }
    return 1;
}

// Warn if M, N, or K are not multiples of the required alignment.
inline void check_tensor_core_alignment(int M, int N, int K, DType dtype) {
    int align = tensor_core_alignment(dtype);
    if (M % align != 0 || N % align != 0 || K % align != 0) {
        printf("WARNING: M=%d N=%d K=%d not multiples of %d — Tensor Core perf cliff!\n",
               M, N, K, align);
    }
}
