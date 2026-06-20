#pragma once
#include <cstdint>
// TODO: implement on A100/H100

// Dynamic loss scaler — doubles scale every N clean steps, halves on NaN/Inf
class LossScaler {
public:
    explicit LossScaler(float init_scale = 65536.0f,
                        int growth_interval = 2000,
                        float growth_factor = 2.0f,
                        float backoff_factor = 0.5f);

    float scale() const;

    // Call after unscaling gradients. Returns true if step was valid (no NaN/Inf).
    bool step_and_update(bool grad_overflow);

private:
    float scale_;
    int   growth_interval_;
    float growth_factor_;
    float backoff_factor_;
    int   steps_since_update_ = 0;
};

// Cast FP32 master weights → BF16 for forward/backward kernel input.
void cast_fp32_to_bf16(const float* src, __bfloat16* dst, int n);

// Cast BF16 gradients → FP32 for optimizer step.
void cast_bf16_to_fp32(const __bfloat16* src, float* dst, int n);
