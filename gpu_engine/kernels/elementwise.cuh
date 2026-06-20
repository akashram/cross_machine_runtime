#pragma once
// Element-wise GPU kernels — TODO: implement on GPU hardware

__global__ void add_kernel(const float* a, const float* b, float* c, int n);
__global__ void mul_kernel(const float* a, const float* b, float* c, int n);
__global__ void relu_kernel(const float* x, float* y, int n);
__global__ void gelu_kernel(const float* x, float* y, int n);

// Two-pass softmax: pass 1 finds max, pass 2 computes exp(x-max)/sum
__global__ void softmax_max_kernel(const float* x, float* maxvals, int n, int rows);
__global__ void softmax_norm_kernel(const float* x, const float* maxvals,
                                     float* y, int n, int rows);
