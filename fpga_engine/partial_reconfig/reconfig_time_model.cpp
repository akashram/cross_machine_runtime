// reconfig_time_model.cpp — partial-bitstream reconfiguration time model.
//
// PLAN.md step 22 asks to "hot-swap kernels at runtime, measure
// reconfiguration time." There's no F1 instance here (see dfx_pblock.tcl
// and pr_host_driver.cpp), so this file does the part that doesn't need
// one: a first-order model of how long loading a partial bitstream of a
// given size should take over the ICAP/PCAP configuration interface, so
// pr_host_driver.cpp's real measurement (once it exists) has a predicted
// number to be checked against.
//
// Same caveat as clock_gating_model.cpp / critical_path_model.cpp /
// slr_crossing_model.cpp: kIcapBandwidthMBps and kFixedOverheadMs are
// first-order engineering approximations (a commonly cited UltraScale+
// ICAPE3 x32 configuration bandwidth figure and an estimated DRC/
// handshake overhead), not datasheet-verified numbers for AWS F1's
// specific PCAP-mediated shell path -- unconfirmed without a real
// reconfiguration to time. What doesn't depend on getting the constants
// exactly right: the model's structural claim that reconfiguration time
// scales linearly with partial bitstream size plus a fixed per-swap
// overhead -- i.e. a smaller reconfigurable partition (smaller RM,
// smaller partial bitstream) always reconfigures faster, which is the
// actual design lever dfx_pblock.tcl's pblock sizing controls.

#include <cstdio>
#include <vector>

namespace {

// UltraScale+ ICAPE3 x32 configuration interface, commonly cited nominal
// bandwidth at a 100 MHz config clock (32 bits/cycle) -- see file header
// caveat; AWS F1's actual PCAP-mediated path may differ.
constexpr double kIcapBandwidthMBps = 400.0;

// Fixed per-reconfiguration overhead: ICAP enable/disable sequencing,
// CRC/DRC checks, shell handshake -- see file header caveat.
constexpr double kFixedOverheadMs = 5.0;

double reconfig_time_ms(double bitstream_size_mb) {
    return (bitstream_size_mb / kIcapBandwidthMBps) * 1000.0 + kFixedOverheadMs;
}

} // namespace

int main() {
    struct Scenario { const char* label; double size_mb; };
    const std::vector<Scenario> scenarios = {
        {"small RM (single AXI-stream kernel, fraction of an SLR)", 0.5},
        {"medium RM (small MLP kernel, ~1 clock region)", 2.0},
        {"large RM (full SLR)", 8.0},
    };

    std::printf("=== predicted partial reconfiguration time vs. bitstream size "
                "(ICAP bandwidth = %.0f MB/s, fixed overhead = %.1f ms) ===\n",
                kIcapBandwidthMBps, kFixedOverheadMs);
    for (const auto& s : scenarios) {
        double t = reconfig_time_ms(s.size_mb);
        std::printf("%-58s | %5.2f MB | %7.2f ms\n", s.label, s.size_mb, t);
    }

    double predicted = reconfig_time_ms(0.5);
    std::printf(
        "\napplied to dfx_pblock.tcl's RM_A/RM_B pair (axi_passthrough <-> "
        "axi_increment, a single AXI4-Stream kernel, modeled at the 'small RM' "
        "size above): predicted hot-swap latency = %.2f ms -- pr_host_driver.cpp's "
        "measured load_xclbin() duration for the partial xclbin should land near "
        "this once run on F1.\n",
        predicted);

    return 0;
}
