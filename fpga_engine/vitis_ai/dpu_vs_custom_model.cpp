// dpu_vs_custom_model.cpp — Vitis AI DPU deployment vs. ml_kernel.cpp's
// custom HLS kernel, for the same 16 -> 32 (ReLU) -> 8 MLP (768 total
// multiplies).
//
// PLAN.md step 24 asks to "run Vitis AI compiled model vs. custom HLS
// kernel on same workload, document resource/latency/power comparison,
// justify custom implementation." There's no Vitis AI toolchain, DPU
// overlay, or F1 instance here (see vai_compile_flow.sh, mlp_model.py),
// so this file does the part that doesn't need one: a first-order
// latency/resource budget for both deployment paths on the identical
// network shape, so vai_compile_flow.sh's real measurement (once it runs)
// has a predicted number -- and a predicted *reason* for the gap -- to be
// checked against. Same structure as fpga_net/net_latency_model.cpp:
// two paths sharing part of the work, differing in fixed overhead.
//
// --- Custom kernel side (ml_kernel_mlp) ---
// Latency-in-cycles and clock come directly from two *other* portable
// models already in this repo, not fresh numbers invented here:
//   - Pipeline structure (iteration counts) from ml_kernel.cpp itself:
//     layer1 is 32 PIPELINE-II=1 iterations over a 16-wide unrolled
//     reduction, layer2 is 8 iterations over a 32-wide reduction.
//   - Per-layer pipeline-fill depth and achievable clock from
//     timing_closure/critical_path_model.cpp's analysis of this exact
//     kernel: a balanced-tree-restructured reduction (the fix
//     pipelined_tree_reduce.cpp applies) has tree depth 4 (layer1) / 5
//     (layer2), and closes a 300MHz target under that model's optimistic
//     ("uncongested") delay assumption -- see timing_closure/README.md's
//     results table. The unmodified *flat* ml_kernel_mlp does NOT close
//     300MHz per that same model; this file uses the retimed number
//     because that is step 15's own stated fix, not because the flat
//     kernel as literally committed has been shown to hit it.
//
// --- DPU side ---
// Every constant below (PE array width, clock, dispatch/DMA overhead,
// resource footprint) is a representative order-of-magnitude figure for
// Xilinx's DPU IP family in its smallest standard configuration, the same
// kind of literature-sourced approximation net_latency_model.cpp and
// clock_gating_model.cpp use for their non-local-measurable stages -- not
// a datasheet number for whatever specific DPU fingerprint an AWS F1
// Vitis AI shell would actually load. What the model does not depend on
// getting exactly right: a DPU is architecturally a shared, instruction-
// driven engine sized for large CNN/transformer layers, so it pays a
// fixed per-inference dispatch + weight-DMA overhead a bare point-design
// kernel has no equivalent of -- structurally the same argument
// net_latency_model.cpp makes about kernel-mediated vs. bypass networking.

#include <cmath>
#include <cstdio>

namespace {

// --- Custom kernel (ml_kernel_mlp, tree-retimed per timing_closure/) ---
constexpr int kLayer1Iters = 32;   // hidden neurons, PIPELINE II=1
constexpr int kLayer1TreeDepth = 4;  // ceil(log2(16)), from critical_path_model.cpp
constexpr int kLayer2Iters = 8;    // output neurons, PIPELINE II=1
constexpr int kLayer2TreeDepth = 5;  // ceil(log2(32)), from critical_path_model.cpp
constexpr double kCustomClockMHz = 300.0; // timing_closure/README.md's closed target (uncongested, tree-retimed)
constexpr int kTotalMultiplies = 16 * 32 + 32 * 8; // 768, matches ml_kernel.cpp's own comment

// Peak concurrent DSP48E2 instances: layer1's inner loop unrolls 16
// multiplies per pipeline stage, layer2's unrolls 32. Whether Vitis HLS
// shares those instances across the two (sequential, non-overlapping)
// loops or allocates them separately is a real binding decision the
// tool makes at synthesis time -- ml_kernel/README.md's own DSP column
// is TODO for exactly this reason. Report both bounds instead of
// guessing which one holds.
constexpr int kCustomDspSharedBound = 32;   // max(16, 32): full resource sharing
constexpr int kCustomDspUnsharedBound = 48; // 16 + 32: no sharing

// --- DPU (Vitis AI, smallest standard config, representative figures) ---
constexpr double kDpuPeArrayMacsPerCycle = 512.0; // e.g. a "B512"-class config
constexpr double kDpuClockMHz = 300.0;            // typical DPU clock, same order as custom
constexpr double kDpuDispatchOverheadUs = 2.0;    // job submission through VART/XRT, analogous
                                                   // in kind to net_latency_model.cpp's
                                                   // interrupt-dispatch stage
constexpr double kDpuWeightDmaOverheadUs = 1.0;   // weight-buffer load from DDR into the DPU's
                                                   // on-chip buffer, paid even for tiny models
constexpr double kDpuReadbackOverheadUs = 0.5;    // result readback through the same runtime path
constexpr int kDpuTypicalLutFootprint = 50000;    // fixed cost of the PE array + instruction/
constexpr int kDpuTypicalDspFootprint = 500;      // weight-buffer infrastructure, independent of
                                                   // how small the workload is

double custom_latency_ns() {
    double cycles = (kLayer1Iters + kLayer1TreeDepth) + (kLayer2Iters + kLayer2TreeDepth);
    return cycles / (kCustomClockMHz * 1e6) * 1e9;
}

double dpu_latency_ns() {
    double compute_cycles = std::ceil(kTotalMultiplies / kDpuPeArrayMacsPerCycle);
    double compute_ns = compute_cycles / (kDpuClockMHz * 1e6) * 1e9;
    double overhead_ns = (kDpuDispatchOverheadUs + kDpuWeightDmaOverheadUs + kDpuReadbackOverheadUs) * 1e3;
    return overhead_ns + compute_ns;
}

} // namespace

int main() {
    double custom_ns = custom_latency_ns();
    double dpu_ns = dpu_latency_ns();
    double dpu_compute_ns = std::ceil(kTotalMultiplies / kDpuPeArrayMacsPerCycle) /
                             (kDpuClockMHz * 1e6) * 1e9;
    double dpu_overhead_ns = dpu_ns - dpu_compute_ns;

    std::printf("=== predicted single-inference latency: DPU vs. custom HLS kernel ===\n");
    std::printf("workload: 16 -> 32 (ReLU) -> 8 MLP, %d total INT8 multiplies\n\n", kTotalMultiplies);

    int custom_cycles = (kLayer1Iters + kLayer1TreeDepth) + (kLayer2Iters + kLayer2TreeDepth);
    std::printf("custom (ml_kernel_mlp, tree-retimed @ %.0fMHz): "
                "layer1(%d iters+%d fill) + layer2(%d iters+%d fill) = %d cycles = %.1fns\n",
                kCustomClockMHz, kLayer1Iters, kLayer1TreeDepth, kLayer2Iters, kLayer2TreeDepth,
                custom_cycles, custom_ns);
    std::printf("custom peak DSP48E2 use: %d (if layers share instances) - %d (if not)\n",
                kCustomDspSharedBound, kCustomDspUnsharedBound);

    std::printf("\nDPU (B512-class @ %.0fMHz): dispatch=%.1fus + weight DMA=%.1fus + "
                "readback=%.1fus + compute=%.2fns (%d MACs / %.0f MACs/cycle, rounds to %d cycle) "
                "= %.1fns\n",
                kDpuClockMHz, kDpuDispatchOverheadUs, kDpuWeightDmaOverheadUs, kDpuReadbackOverheadUs,
                dpu_compute_ns, kTotalMultiplies, kDpuPeArrayMacsPerCycle,
                (int)std::ceil(kTotalMultiplies / kDpuPeArrayMacsPerCycle), dpu_ns);
    std::printf("DPU fixed resource footprint (order-of-magnitude, config-dependent): "
                "~%d LUT, ~%d DSP -- provisioned for large CNN layers, independent of this "
                "workload's size\n", kDpuTypicalLutFootprint, kDpuTypicalDspFootprint);

    double speedup = dpu_ns / custom_ns;
    std::printf("\npredicted speedup = %.1fx (custom kernel), dominated by DPU dispatch+DMA "
                "overhead (%.1fns of %.1fns, %.1f%%) -- compute time itself is near-identical "
                "since %d multiplies fits in a single DPU pass; the gap is architectural "
                "(instruction-driven shared engine vs. bare point-design pipeline), not "
                "arithmetic throughput.\n",
                speedup, dpu_overhead_ns, dpu_ns, (dpu_overhead_ns / dpu_ns) * 100.0, kTotalMultiplies);

    std::printf("\nresource verdict: custom kernel needs <= %d DSP48E2 and 0 BRAM/URAM (both "
                "layers' operands are ARRAY_PARTITION-complete, i.e. registers, not memories) "
                "vs. the DPU's ~%d DSP + ~%d LUT footprint that exists regardless of workload "
                "size -- for a network this small, embedded inside a larger pipeline (e.g. "
                "thermal_router/), the custom kernel's near-zero incremental footprint is the "
                "justification PLAN.md step 24 asks for; the DPU's advantage is reprogrammability "
                "across model shapes without resynthesis, which this single fixed MLP doesn't need.\n",
                kCustomDspUnsharedBound, kDpuTypicalDspFootprint, kDpuTypicalLutFootprint);

    return 0;
}
