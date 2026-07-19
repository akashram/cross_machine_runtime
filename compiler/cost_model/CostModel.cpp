// CostModel.cpp — Step 13. Roofline-style estimator: time = max(compute
// time, memory time) + launch overhead. See DESIGN.md in this project's
// cpu_engine/gpu_engine roofline steps for why max() (not sum) is the
// right combinator — the two costs overlap when compute and memory
// pipelines run concurrently, so runtime is bound by whichever is slower,
// not their total.

#include "CostModel.h"

#include <algorithm>

namespace {
constexpr double kBytesPerElement = 4.0; // FP32 only for now; see header TODO
}

DeviceCost get_device_cost(DeviceType device) {
    switch (device) {
        case DeviceType::CPU:
            // Generic AVX-512 dual-socket server (cpu_engine/'s target):
            // ~2 TFLOPS FP32 (16 cores * 8-wide FMA * ~2 * ~2GHz, rough),
            // dual-channel DDR4-3200 ~50 GB/s, cheap dispatch (no launch
            // queue), no device-to-device transfer concept.
            return {2.0e12, 50.0, 1.0, 50.0, 50.0};
        case DeviceType::GPU:
            // A100 spec sheet: 19.5 TFLOPS FP32 (non-tensor-core),
            // 2039 GB/s HBM2e, ~5us kernel launch overhead, PCIe4 x16
            // ~25 GB/s host<->device.
            return {19.5e12, 2039.0, 5.0, 25.0, 25.0};
        case DeviceType::FPGA:
            // Xilinx VU9P: ~6840 DSP48E2 slices, conservatively 1
            // MAC/cycle/DSP @ 300MHz => ~4.1 TFLOPS ceiling (rarely hit
            // for anything but a hand-tuned systolic kernel), 4x DDR4
            // banks ~77 GB/s aggregate, DMA-heavy so the highest launch
            // overhead of the four devices.
            return {4.1e12, 77.0, 50.0, 12.0, 12.0};
        case DeviceType::TPU:
            // TPU v4 spec sheet: ~275 TFLOPS bf16 (treated as the FP32-
            // equivalent slot here since Shape has no dtype yet — see
            // header TODO), ~1200 GB/s HBM, systolic array fill/drain
            // adds meaningful fixed latency per op.
            return {275.0e12, 1200.0, 10.0, 100.0, 100.0}; // ICI, not PCIe
    }
    return {};
}

namespace {

// FLOPs and byte traffic per op type. Shape field meaning:
//   Matmul:      batch x [m,k] @ [k,n] -> [m,n]
//   Conv:        caller packs im2col-equivalent dims: m=outH*outW,
//                n=outC, k=inC*kH*kW (documented in PlacementPass.cpp
//                at the call site) so it reduces to a matmul-shaped cost.
//   Elementwise: batch x m x n, k unused.
//   Reduce:      reads batch x m x n, writes batch x m (k unused).
//   Transfer:    batch x m x n elements moved, no FLOPs.
double flops(OpType op, const Shape& s) {
    switch (op) {
        case OpType::Matmul:
        case OpType::Conv:
            return 2.0 * s.batch * s.m * s.n * s.k; // multiply + add per MAC
        case OpType::Elementwise:
            return static_cast<double>(s.batch) * s.m * s.n;
        case OpType::Reduce:
            return static_cast<double>(s.batch) * s.m * s.n; // one op per input elem
        case OpType::Transfer:
            return 0.0;
    }
    return 0.0;
}

double bytesMoved(OpType op, const Shape& s) {
    const double elem = kBytesPerElement;
    switch (op) {
        case OpType::Matmul:
        case OpType::Conv:
            return elem * s.batch * (s.m * s.k + s.k * s.n + s.m * s.n);
        case OpType::Elementwise:
            // one read + one write per element (binary ops read 2x; use
            // the conservative unary estimate since op arity isn't
            // threaded through Shape).
            return elem * 2.0 * s.batch * s.m * s.n;
        case OpType::Reduce:
            return elem * static_cast<double>(s.batch) * s.m * s.n; // read-dominated
        case OpType::Transfer:
            return elem * static_cast<double>(s.batch) * s.m * s.n;
    }
    return 0.0;
}

} // namespace

double estimate_us(DeviceType device, OpType op, const Shape& shape,
                   const DeviceCost& cost) {
    (void)device;
    if (op == OpType::Transfer) {
        double gb = bytesMoved(op, shape) / 1.0e9;
        double bw = std::max(cost.transfer_in_gbs, 1e-9);
        return (gb / bw) * 1.0e6 + cost.launch_overhead_us;
    }

    double computeUs = (flops(op, shape) / std::max(cost.flops_per_sec, 1.0)) * 1.0e6;
    double gb = bytesMoved(op, shape) / 1.0e9;
    double memoryUs = (gb / std::max(cost.mem_bandwidth_gbs, 1e-9)) * 1.0e6;

    // Roofline combinator: bound by whichever pipeline is slower, not the
    // sum — compute and memory overlap for a well-pipelined kernel.
    return std::max(computeUs, memoryUs) + cost.launch_overhead_us;
}
