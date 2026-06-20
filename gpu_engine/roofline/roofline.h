#pragma once
// GPU Roofline model utilities — TODO: implement on GPU hardware

struct RooflineHardware {
    double peak_fp32_tflops;
    double peak_fp16_tflops;
    double peak_bandwidth_gbs;
};

struct RooflineResult {
    double arithmetic_intensity; // FLOP/byte
    double achieved_tflops;
    double ceiling_tflops;       // min(peak_flops, AI * peak_bandwidth)
    double utilization;          // achieved / ceiling
    bool compute_bound;          // true if ceiling is compute, false if bandwidth
};

// Measure peak FLOPS using cuBLAS SGEMM at maximum occupancy.
double measure_peak_flops_tflops();

// Measure peak HBM bandwidth using STREAM TRIAD kernel (GB/s).
double measure_peak_bandwidth_gbs();

// Classify a kernel given its achieved metrics and hardware ceilings.
RooflineResult classify_kernel(double achieved_tflops,
                                double flops_per_byte,
                                const RooflineHardware& hw);
