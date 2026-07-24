// pr_host_driver.cpp — host-side driver that hot-swaps between the two
// partial-reconfiguration kernel variants (RM_A = axi_passthrough, RM_B =
// axi_increment -- see dfx_pblock.tcl) and measures the reconfiguration
// latency, per PLAN.md step 22's "hot-swap at runtime, measure
// reconfiguration time" ask.
//
// On AWS F1 + Vitis, each DFX configuration (a full xclbin containing
// RM_A, a partial xclbin containing only RM_B's region, both produced by
// packaging dfx_pblock.tcl's two bitstreams with v++) is deployed to the
// same device via XRT's xrt::device::load_xclbin -- loading a second
// xclbin whose UUID matches a partial-region PR configuration re-triggers
// the ICAP/PCAP reconfiguration sequence for just that region, without a
// full device reset. That's the real, documented API surface (same
// reasoning as xadc_sensors.cpp/dma_orchestration.cpp preferring XRT's
// native API to hand-rolled PCIe/ICAP register access).
//
// TODO: run on F1 with XRT installed, a card enumerated, and both
// dfx_pblock.tcl-produced bitstreams packaged into xclbins via v++.
// Untested locally -- no XRT runtime or FPGA card available.
#include <chrono>
#include <cstdio>
#include <string>

#include "xrt/xrt_device.h"
#include "xrt/xrt_uuid.h"

int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: %s <full_rm_a.xclbin> <partial_rm_b.xclbin>\n", argv[0]);
        return 1;
    }
    std::string full_xclbin = argv[1];
    std::string partial_xclbin = argv[2];

    xrt::device device(0); // first FPGA device enumerated by XRT

    auto t0 = std::chrono::steady_clock::now();
    xrt::uuid uuid_a = device.load_xclbin(full_xclbin);
    auto t1 = std::chrono::steady_clock::now();
    std::printf("pr_host_driver: loaded RM_A (full bitstream) uuid=%s in %.2f ms\n",
                uuid_a.to_string().c_str(),
                std::chrono::duration<double, std::milli>(t1 - t0).count());

    // Hot-swap: load the partial bitstream for RM_B. Only the
    // reconfigurable partition's fabric is reprogrammed -- the static
    // shell and any other kernels stay live -- so this call's duration
    // is the actual "reconfiguration time" PLAN.md step 22 asks to
    // measure, not the full-bitstream load above.
    auto t2 = std::chrono::steady_clock::now();
    xrt::uuid uuid_b = device.load_xclbin(partial_xclbin);
    auto t3 = std::chrono::steady_clock::now();
    double reconfig_ms = std::chrono::duration<double, std::milli>(t3 - t2).count();
    std::printf("pr_host_driver: hot-swapped to RM_B (partial bitstream) uuid=%s in %.2f ms\n",
                uuid_b.to_string().c_str(), reconfig_ms);

    std::printf("pr_host_driver: next -> compare %.2f ms against reconfig_time_model.cpp's "
                "prediction for this partial bitstream's file size\n",
                reconfig_ms);
    return 0;
}
