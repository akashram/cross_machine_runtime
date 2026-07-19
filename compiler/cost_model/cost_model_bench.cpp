// cost_model_bench.cpp — sanity-runs estimate_us() across representative op
// shapes and all four devices. Unlike every other Phase 4 component, this
// one has zero MLIR dependency, so it actually builds and runs on the Mac
// right now (see compiler/cost_model/README.md for real captured output).
// The *numbers* are still placeholders (spec-sheet peaks, not measured —
// see get_device_cost's comments); what this validates today is that the
// roofline arithmetic and the device-dispatch logic are correct, not that
// the constants are calibrated.

#include "CostModel.h"

#include <cstdio>
#include <string>
#include <vector>

namespace {

const char* deviceName(DeviceType d) {
    switch (d) {
        case DeviceType::CPU: return "CPU";
        case DeviceType::GPU: return "GPU";
        case DeviceType::FPGA: return "FPGA";
        case DeviceType::TPU: return "TPU";
    }
    return "?";
}

struct NamedShape { const char* name; OpType op; Shape shape; };

} // namespace

int main() {
    std::vector<NamedShape> workloads = {
        {"matmul 4096x4096x4096",       OpType::Matmul,      {1, 4096, 4096, 4096}},
        {"matmul 128x4096x4096 (batch)", OpType::Matmul,      {32, 128, 4096, 4096}},
        {"elementwise relu 1M elems",   OpType::Elementwise, {1, 1024, 1024, 0}},
        {"reduce sum 4096x4096->4096",  OpType::Reduce,      {1, 4096, 4096, 0}},
        {"transfer 64MB",               OpType::Transfer,    {1, 4096, 4096, 0}},
    };
    std::vector<DeviceType> devices = {DeviceType::CPU, DeviceType::GPU,
                                        DeviceType::FPGA, DeviceType::TPU};

    std::printf("%-32s", "workload");
    for (DeviceType d : devices) std::printf("%12s (us)", deviceName(d));
    std::printf("\n");

    for (const auto& w : workloads) {
        std::printf("%-32s", w.name);
        for (DeviceType d : devices) {
            double us = estimate_us(d, w.op, w.shape, get_device_cost(d));
            std::printf("%16.2f", us);
        }
        std::printf("\n");
    }
    return 0;
}
