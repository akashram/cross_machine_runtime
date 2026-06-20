#pragma once
#include <cstddef>
// TODO: implement on GPU — GPTQ (Frantar et al. 2022)

// GPTQ INT4 group quantization.
// group_size: number of weights sharing a scale (typically 128).
class GptqQuantizer {
public:
    GptqQuantizer(int group_size = 128, int bits = 4);

    // Quantize weight matrix using Hessian-guided greedy optimization.
    // calibration_data: activation statistics for Hessian computation.
    void quantize(const float* weight,
                  int rows, int cols,
                  const float* calibration_data, int calib_samples,
                  int8_t*  quantized,   // output: INT4 packed as INT8
                  float*   scales,      // output: per-group scales
                  int8_t*  zeros);      // output: per-group zero points

    // Dequantize and multiply: (W_quant * scale + zero) @ x
    void dequant_matmul(const int8_t* quantized, const float* scales,
                         const int8_t* zeros, const float* x,
                         float* y, int rows, int cols, int batch);
};
