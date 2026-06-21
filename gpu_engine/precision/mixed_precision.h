#pragma once
// Mixed precision training utilities — BF16 forward/backward, FP32 master weights,
// dynamic loss scaling.
//
// Why BF16 instead of FP16?
//   FP16 (5e4m10): max value 65504.  Large gradients cause overflow → need loss scaling.
//   BF16 (8e7m): same exponent range as FP32, much less overflow risk.
//   On A100+, BF16 Tensor Cores achieve the same TFLOPS as FP16 while tolerating
//   larger values.  Loss scaling is still recommended but less critical.
//
// Workflow
// --------
// 1. Master weights stored in FP32.
// 2. Before forward: cast FP32 weights → BF16 (cast_fp32_to_bf16).
// 3. Forward + backward in BF16 (uses BF16 GEMM → FP32 accumulator via cuBLAS).
// 4. Multiply loss by scale factor before backward.
// 5. After backward: check BF16 gradients for NaN/Inf (check_overflow).
// 6. If clean: unscale gradients (divide by scale), optimizer step in FP32.
// 7. If overflow: skip optimizer step, halve scale. (LossScaler::step_and_update)
//
// LossScaler
// ----------
// Starts with a large scale (e.g. 65536).  Every `growth_interval` clean steps,
// doubles the scale (pushes more signal into FP16/BF16 mantissa).  On NaN/Inf
// detection, halves immediately to recover.

#include <cmath>
#include <cstdint>
#include <cuda_bf16.h>    // __bfloat16, __float2bfloat16, __bfloat162float
#include <cuda_fp16.h>    // __half (for FP8 section)
#include <cuda_runtime.h>

// -----------------------------------------------------------------------
// LossScaler — dynamic loss scale for mixed-precision training.
// This class lives entirely on the host.
// -----------------------------------------------------------------------
class LossScaler {
public:
    explicit LossScaler(float init_scale      = 65536.0f,
                        int   growth_interval = 2000,
                        float growth_factor   = 2.0f,
                        float backoff_factor  = 0.5f)
        : scale_(init_scale),
          growth_interval_(growth_interval),
          growth_factor_(growth_factor),
          backoff_factor_(backoff_factor) {}

    float scale() const { return scale_; }

    // Call after checking gradients.  `grad_overflow` = true if any gradient
    // is NaN or Inf.  Returns true if the step was valid (gradients are finite).
    bool step_and_update(bool grad_overflow) {
        if (grad_overflow) {
            scale_ = fmaxf(1.0f, scale_ * backoff_factor_);
            steps_since_update_ = 0;
            return false;  // skip optimizer step
        }
        ++steps_since_update_;
        if (steps_since_update_ >= growth_interval_) {
            scale_ *= growth_factor_;
            steps_since_update_ = 0;
        }
        return true;  // step is valid
    }

private:
    float scale_;
    int   growth_interval_;
    float growth_factor_;
    float backoff_factor_;
    int   steps_since_update_ = 0;
};

// -----------------------------------------------------------------------
// Cast kernels — FP32 ↔ BF16
// Included as inline device code so the header can be used from any .cu file.
// -----------------------------------------------------------------------

namespace detail {

__global__ void fp32_to_bf16_kernel(const float* __restrict__ src,
                                     __bfloat16*  __restrict__ dst, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = __float2bfloat16(src[i]);
}

__global__ void bf16_to_fp32_kernel(const __bfloat16* __restrict__ src,
                                     float*            __restrict__ dst, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = __bfloat162float(src[i]);
}

__global__ void scale_bf16_kernel(__bfloat16* __restrict__ x, float scale, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float v = __bfloat162float(x[i]) * scale;
        x[i] = __float2bfloat16(v);
    }
}

// Returns 1 if any element is NaN or Inf, else 0.
__global__ void check_overflow_kernel(const __bfloat16* __restrict__ x,
                                       int* __restrict__ flag, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float v = __bfloat162float(x[i]);
        if (!isfinite(v)) atomicOr(flag, 1);
    }
}

} // namespace detail

// Cast `n` FP32 elements from `src` to BF16 in `dst` on the current device.
inline void cast_fp32_to_bf16(const float* src, __bfloat16* dst, int n,
                               cudaStream_t stream = 0) {
    int blocks = (n + 255) / 256;
    detail::fp32_to_bf16_kernel<<<blocks, 256, 0, stream>>>(src, dst, n);
}

// Cast `n` BF16 elements from `src` to FP32 in `dst`.
inline void cast_bf16_to_fp32(const __bfloat16* src, float* dst, int n,
                               cudaStream_t stream = 0) {
    int blocks = (n + 255) / 256;
    detail::bf16_to_fp32_kernel<<<blocks, 256, 0, stream>>>(src, dst, n);
}

// Multiply all `n` BF16 elements by `scale` in place.
// Use before backward pass to apply loss scale.
inline void scale_bf16(const __bfloat16* x, float scale, int n,
                        cudaStream_t stream = 0) {
    int blocks = (n + 255) / 256;
    detail::scale_bf16_kernel<<<blocks, 256, 0, stream>>>(const_cast<__bfloat16*>(x),
                                                          scale, n);
}

// Check for NaN/Inf in a BF16 gradient tensor.  Returns true if any overflow.
// Uses a device-side flag with atomicOr to avoid a full reduction.
inline bool check_overflow(const __bfloat16* grads, int n, cudaStream_t stream = 0) {
    int* d_flag;
    cudaMalloc(&d_flag, sizeof(int));
    cudaMemsetAsync(d_flag, 0, sizeof(int), stream);

    int blocks = (n + 255) / 256;
    detail::check_overflow_kernel<<<blocks, 256, 0, stream>>>(grads, d_flag, n);

    int h_flag = 0;
    cudaMemcpyAsync(&h_flag, d_flag, sizeof(int), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    cudaFree(d_flag);
    return h_flag != 0;
}
