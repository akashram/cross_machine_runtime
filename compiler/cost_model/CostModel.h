#pragma once
#include <cstdint>
// Step 13: device cost model. Pure C++, no MLIR dependency — this is the
// one Phase 4 component that builds and runs without a Linux/MLIR
// toolchain, so it's wired into the top-level build unconditionally (see
// root CMakeLists.txt) instead of being gated behind MLIR_DIR like the
// rest of compiler/. Consumed by the placement pass (step 9) and the
// auto-sharding pass (step 10, for transfer cost of inserted collectives).

enum class DeviceType { CPU, GPU, FPGA, TPU };
enum class OpType     { Matmul, Conv, Elementwise, Reduce, Transfer };

// Generic op-size descriptor. Field meaning is op-dependent (documented in
// CostModel.cpp per case) since Matmul/Conv/Elementwise/Reduce don't share
// a natural shape shape — reusing one struct keeps the estimate_us
// signature uniform for the placement pass's per-op dispatch loop.
struct Shape { int64_t batch, m, n, k; };

struct DeviceCost {
    double flops_per_sec;       // measured (TODO) or spec-sheet peak, FP32
    double mem_bandwidth_gbs;   // measured (TODO) or spec-sheet peak
    double launch_overhead_us;  // per kernel/op dispatch
    double transfer_in_gbs;     // host<->device or device<->device, inbound
    double transfer_out_gbs;
};

// Estimate execution time in microseconds for an op on a device: a
// roofline-style max(compute_time, memory_time) + launch_overhead, plus
// transfer time if op == Transfer. See CostModel.cpp for the per-op FLOP
// and byte-traffic formulas.
double estimate_us(DeviceType device, OpType op, const Shape& shape,
                   const DeviceCost& cost);

// Returns cost figures for each device. Values below are public spec-sheet
// peaks (A100 FP32/HBM2e, VU9P DSP/4xDDR4, TPU v4 bf16/HBM, a generic
// AVX-512 server CPU) — NOT measured. Phases 2/3/7/8 benchmarks will
// replace these with real numbers; the placement pass's *logic* doesn't
// change when that happens, only these constants do.
DeviceCost get_device_cost(DeviceType device);
