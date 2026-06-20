#pragma once
// TODO: calibrate on GPU/FPGA/TPU hardware

enum class DeviceType { CPU, GPU, FPGA, TPU };
enum class OpType     { Matmul, Conv, Elementwise, Reduce, Transfer };

struct Shape { int batch, m, n, k; };

struct DeviceCost {
    double flops_per_sec;       // measured peak
    double mem_bandwidth_gbs;   // measured peak
    double launch_overhead_us;  // per kernel
    double transfer_in_gbs;     // host→device or device→device
    double transfer_out_gbs;
};

// Estimate execution time in microseconds for an op on a device.
double estimate_us(DeviceType device, OpType op, const Shape& shape,
                   const DeviceCost& cost);

// Returns calibrated costs from benchmarks (placeholder values until measured).
DeviceCost get_device_cost(DeviceType device);
