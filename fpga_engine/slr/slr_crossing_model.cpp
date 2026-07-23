// slr_crossing_model.cpp — SLR crossing penalty model, extending
// timing_closure/critical_path_model.cpp to a second delay source.
//
// VU9P is a 3-SLR SSI (stacked silicon interconnect) device: SLR0/1/2 are
// separate dies connected by Laguna sites / SLLs (super long lines), which
// add measurable routing delay any signal crossing an SLR boundary pays
// that a same-SLR signal doesn't. PLAN.md step 16 asks to measure that
// penalty and derive placement design rules from it. This model does the
// part that doesn't need Vivado: given a design's resource footprint and
// step 15's already-computed critical-path numbers, predict (a) whether a
// design's logic *needs* to span SLRs at all, and (b) how much headroom
// an unavoidable crossing costs when added to an already-tight path.
//
// Same caveat as critical_path_model.cpp: kSllDelayLowNs/HighNs are
// first-order approximations (a Laguna/SLL crossing has real, measurable,
// but device- and route-specific delay), not a datasheet quote -- the
// device topology constants (kDspTotalVU9P, kNumSlrsVU9p) are the part's
// actual published specs and are not approximations.

#include <cmath>
#include <cstdio>
#include <initializer_list>

namespace {

// VU9P published specs (not approximations).
constexpr int kDspTotalVU9P = 6840;
constexpr int kNumSlrsVU9p = 3;

// Even split across SLRs is an assumption for this model, not a measured
// fact -- VU9P's actual per-SLR DSP count may be uneven. Confirm against
// `report_utilization -slr` once on real hardware (see slr_pblocks.tcl).
constexpr int kDspPerSlrApprox = kDspTotalVU9P / kNumSlrsVU9p;

// First-order approximation for one Laguna/SLL crossing's added delay,
// same "low/high" framing as critical_path_model.cpp's stage delay, for
// the same reason: the true number depends on route congestion and is
// unknown without real place & route.
constexpr double kSllDelayLowNs = 0.15;
constexpr double kSllDelayHighNs = 0.30;

// Reused from critical_path_model.cpp's model so a crossing's cost can be
// stated in the same units as step 15's already-computed critical paths.
constexpr double kMultiplyDelayNs = 1.5;
constexpr double kTreeStageDelayLowNs = 0.30;
constexpr double kTreeStageDelayHighNs = 0.60;

double tree_path_delay_ns(int depth, double stage_delay_ns, int crossings, double crossing_delay_ns) {
    return kMultiplyDelayNs + depth * stage_delay_ns + crossings * crossing_delay_ns;
}

void report_resource_fit(const char* label, int dsps_used) {
    double pct_of_one_slr = 100.0 * dsps_used / kDspPerSlrApprox;
    std::printf("%-16s | %4d DSPs used, ~%d DSPs/SLR budget -> %.0f%% of one SLR%s\n",
                label, dsps_used, kDspPerSlrApprox, pct_of_one_slr,
                pct_of_one_slr < 100.0 ? " (single-SLR feasible by resource count)"
                                       : " (WILL NOT fit in one SLR)");
}

void report_crossing_cost(int tree_depth, double stage_delay_ns, const char* stage_label,
                            double target_mhz) {
    double period_ns = 1000.0 / target_mhz;
    double no_cross = tree_path_delay_ns(tree_depth, stage_delay_ns, 0, 0.0);
    for (double crossing_delay : {kSllDelayLowNs, kSllDelayHighNs}) {
        double one_cross = tree_path_delay_ns(tree_depth, stage_delay_ns, 1, crossing_delay);
        double wns_no_cross = period_ns - no_cross;
        double wns_one_cross = period_ns - one_cross;
        std::printf(
            "tree_reduce32 depth=%d, stage=%-11s, crossing=%.2fns | 0 crossings: delay=%.2fns WNS=%+.2fns (%s)"
            " | 1 crossing: delay=%.2fns WNS=%+.2fns (%s)\n",
            tree_depth, stage_label, crossing_delay,
            no_cross, wns_no_cross, wns_no_cross >= 0 ? "MEETS" : "FAILS",
            one_cross, wns_one_cross, wns_one_cross >= 0 ? "MEETS" : "FAILS");
    }
}

} // namespace

int main() {
    std::printf("=== resource fit: does this kernel need to span SLRs at all? ===\n");
    // ml_kernel_mlp (fpga_engine/ml_kernel/): layer1 16x32 + layer2 32x8
    // fully unrolled multiplies.
    report_resource_fit("ml_kernel_mlp", 16 * 32 + 32 * 8);
    // dot_product (fpga_engine/dot_product/): 4-way partial-accumulator
    // variant, one multiplier per accumulator lane.
    report_resource_fit("dot_product", 4);

    std::printf("\n=== crossing cost added to timing_closure's tree_reduce32 path"
                " (tree_depth=5, from critical_path_model.cpp) at a 300MHz target ===\n");
    report_crossing_cost(5, kTreeStageDelayLowNs, "uncongested", 300.0);
    report_crossing_cost(5, kTreeStageDelayHighNs, "congested", 300.0);

    return 0;
}
