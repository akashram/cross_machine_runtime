// net_latency_model.cpp — FPGA-direct-bypass vs. CPU-mediated network
// latency model.
//
// PLAN.md step 23 asks to "measure latency vs. CPU-mediated networking."
// There's no F1 instance or P4 toolchain here (see rdma_bypass.p4 and
// onic_shell_integration.tcl), so this file does the part that doesn't
// need one: a first-order latency budget for a small-message one-sided
// WRITE over each path, so rdma_bypass.p4's real measured latency (once
// it runs) has a predicted number and -- more importantly -- a predicted
// *reason for the gap* to be checked against.
//
// Both paths share the same physical network hop (NIC-to-NIC wire time,
// serialization) -- what differs is everything a packet goes through
// AFTER arriving at the destination NIC:
//   - FPGA-bypass path (rdma_bypass.p4): parse -> match -> DMA action,
//     entirely inside the P4 pipeline at line rate. No host CPU wakeup.
//   - CPU-mediated path (a standard kernel socket, or an unoptimized
//     software "DMA controller" driven by a kernel network stack):
//     NIC DMA to a kernel ring buffer, interrupt (or NAPI poll) dispatch,
//     protocol stack traversal (IP/UDP demux, socket lookup), a copy from
//     kernel space into the application's user-space buffer, and a
//     context switch to hand control back to the application.
//
// Same caveat as every other *_model.cpp in fpga_engine/: the per-stage
// constants below are commonly cited order-of-magnitude figures for a
// modern x86 Linux kernel network stack and a P4-programmable NIC
// pipeline (the kind of numbers kernel-bypass literature -- DPDK, RDMA
// vendor whitepapers -- consistently reports), not datasheet-verified
// numbers for this specific F1 shell/instance pairing, unconfirmed
// without a real measurement. What doesn't depend on getting the
// constants exactly right: the FPGA-bypass path structurally skips every
// stage in the CPU-mediated list above, so its latency floor is lower
// by construction, and the gap should be dominated by the CPU-mediated
// path's kernel-stack-traversal + copy + context-switch stages, not by
// the shared network hop.

#include <cstdio>
#include <vector>

namespace {

// Shared by both paths: NIC-to-NIC wire time for a small (~64B-1KB)
// message on a same-rack 100G link -- serialization + propagation, not
// path-dependent.
constexpr double kNetworkHopUs = 0.5;

// --- FPGA-bypass path (rdma_bypass.p4) ---
// A P4 pipeline processes headers at line rate in a fixed, small number
// of clock cycles per stage; commonly cited P4-programmable-NIC pipeline
// latency for a handful of match-action stages at a few hundred MHz.
constexpr double kP4PipelineUs = 0.15;
// AXI4-Stream hop from the box's egress into the DMA engine, on-chip,
// no PCIe/host involvement for the WRITE data path itself.
constexpr double kOnChipDmaHandoffUs = 0.1;

// --- CPU-mediated path (kernel socket / unoptimized host driver) ---
constexpr double kNicToHostDmaUs = 0.8;       // NIC DMA into a kernel ring buffer
constexpr double kInterruptDispatchUs = 2.0;  // interrupt or NAPI poll wakeup
constexpr double kKernelStackTraversalUs = 3.5; // IP/UDP demux, socket lookup
constexpr double kUserCopyUs = 1.5;           // kernel-to-user-space buffer copy
constexpr double kContextSwitchUs = 2.0;      // handing control back to the app

double fpga_bypass_latency_us() {
    return kNetworkHopUs + kP4PipelineUs + kOnChipDmaHandoffUs;
}

double cpu_mediated_latency_us() {
    return kNetworkHopUs + kNicToHostDmaUs + kInterruptDispatchUs +
           kKernelStackTraversalUs + kUserCopyUs + kContextSwitchUs;
}

} // namespace

int main() {
    double bypass = fpga_bypass_latency_us();
    double mediated = cpu_mediated_latency_us();

    std::printf("=== predicted one-sided WRITE latency: FPGA-bypass vs. CPU-mediated ===\n");
    std::printf("FPGA-bypass (rdma_bypass.p4):   network=%.2fus + P4 pipeline=%.2fus + "
                "on-chip DMA handoff=%.2fus = %.2fus\n",
                kNetworkHopUs, kP4PipelineUs, kOnChipDmaHandoffUs, bypass);
    std::printf("CPU-mediated (kernel socket):   network=%.2fus + NIC->host DMA=%.2fus + "
                "interrupt dispatch=%.2fus + kernel stack=%.2fus + user copy=%.2fus + "
                "context switch=%.2fus = %.2fus\n",
                kNetworkHopUs, kNicToHostDmaUs, kInterruptDispatchUs,
                kKernelStackTraversalUs, kUserCopyUs, kContextSwitchUs, mediated);

    double speedup = mediated / bypass;
    double savings_us = mediated - bypass;
    std::printf("\npredicted speedup = %.2fx (%.2fus saved, %.1f%% of the CPU-mediated total)\n",
                speedup, savings_us, (savings_us / mediated) * 100.0);
    std::printf("the gap is dominated by kernel-stack-traversal + interrupt-dispatch + "
                "context-switch (%.2fus of %.2fus, %.1f%%) -- the exact stages the FPGA-bypass "
                "path has no equivalent of, not the shared network hop.\n",
                kInterruptDispatchUs + kKernelStackTraversalUs + kContextSwitchUs, mediated,
                ((kInterruptDispatchUs + kKernelStackTraversalUs + kContextSwitchUs) / mediated) * 100.0);

    return 0;
}
