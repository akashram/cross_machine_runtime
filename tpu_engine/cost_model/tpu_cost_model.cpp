// tpu_cost_model.cpp — $/FLOP and FLOPS/Watt across TPU v4, A100, H100.
//
// PLAN.md Phase 8 step 12: "for each major op type: $/FLOP on TPU v4 vs.
// A100 vs. H100, FLOPS/Watt comparison, document when to choose each."
// Same convention as compiler/cost_model/CostModel.cpp (Phase 4 step 13,
// "the one Phase 4 component with no MLIR dependency... locally run"):
// pure arithmetic over spec-sheet constants, no toolchain/hardware
// dependency, so this compiles and runs directly on this Mac — unlike
// almost everything else in tpu_engine/, which needs JAX or a TPU.
//
// Compute-peak constants for A100/TPU v4 intentionally match
// compiler/cost_model/CostModel.cpp's existing DeviceCost table (19.5
// TFLOPS FP32 for A100's non-tensor-core path, 275 TFLOPS bf16 for TPU
// v4) *where that table's numbers apply* — but this file's comparison is
// tensor-core-vs-MXU (both devices' actual ML-relevant peak), a different,
// larger number than CostModel.cpp's placement-pass estimator uses, since
// placement's estimator predates tensor-core-aware costing (see that
// file's header TODO on dtype-awareness). $/hr and TDP figures are public
// list-price/spec-sheet snapshots, not measurements — same "spec-sheet
// placeholder pending real benchmarks" caveat CostModel.cpp states
// explicitly for its own constants.

#include <cstdio>
#include <string>
#include <vector>

namespace {

struct DeviceSpec {
    std::string name;
    double peak_tflops_bf16;   // tensor-core / MXU peak, dense (no sparsity)
    double tdp_watts;
    double on_demand_usd_per_hr;  // per-chip/per-GPU, public list price snapshot
    std::string price_source;
};

// Snapshot, 2026-07-27 -- cloud on-demand list prices change often and
// vary by region/commitment; treat these as order-of-magnitude reference
// points to recompute against real current pricing before trusting a
// procurement decision, not as a live quote.
const std::vector<DeviceSpec> kDevices = {
    {"TPU v4",  275.0,  192.0,  3.22,  "GCP v4 on-demand, per chip (or free via TRC)"},
    {"A100 80GB SXM4", 312.0, 400.0,  3.67,  "GCP a2-ultragpu-1g on-demand, per GPU"},
    {"H100 80GB SXM5", 989.0, 700.0, 11.06,  "GCP a3-highgpu-8g on-demand / 8, per GPU"},
};

struct Derived {
    double flops_per_watt;       // peak TFLOPS*1e12 / TDP
    double usd_per_pflop_hour;   // $/hr / (peak TFLOPS / 1000) -- cost to rent 1 PFLOP/s of compute for an hour
};

Derived derive(const DeviceSpec &d) {
    Derived r;
    r.flops_per_watt = (d.peak_tflops_bf16 * 1e12) / d.tdp_watts;
    r.usd_per_pflop_hour = d.on_demand_usd_per_hr / (d.peak_tflops_bf16 / 1000.0);
    return r;
}

} // namespace

int main() {
    std::printf("%-18s %12s %8s %10s %16s %18s\n",
                "device", "TFLOPS(bf16)", "TDP(W)", "$/hr", "TFLOPS/Watt", "$/PFLOP-hr");
    for (const auto &d : kDevices) {
        Derived r = derive(d);
        std::printf("%-18s %12.1f %8.0f %10.2f %16.3f %18.2f\n",
                    d.name.c_str(), d.peak_tflops_bf16, d.tdp_watts,
                    d.on_demand_usd_per_hr, r.flops_per_watt / 1e12, r.usd_per_pflop_hour);
    }

    std::printf("\nprice sources (snapshot, verify before procurement):\n");
    for (const auto &d : kDevices)
        std::printf("  %-18s %s\n", d.name.c_str(), d.price_source.c_str());

    std::printf(
        "\nNote: these are peak dense bf16/tensor-core numbers -- real "
        "achieved throughput on any of the three depends entirely on "
        "hitting each device's alignment sweet spot (128x128 MXU tiles for "
        "TPU, per layout_opt's ceiling model; tensor-core tile shapes for "
        "A100/H100). A workload that can't reach that ceiling on one "
        "device may still win on $/useful-FLOP on a different device even "
        "if this table's peak $/PFLOP-hr ranks them the other way.\n");
    return 0;
}
