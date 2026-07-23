// critical_path_model.cpp — analytical critical-path model: flat vs.
// balanced-tree reduction, for timing_closure's step-15 retiming fix.
//
// PLAN.md step 15 asks to "analyze critical paths ... apply retiming ...
// document each fix" against real Vivado report_timing output. There is
// no F1 instance here (see close_timing.tcl), so this file does the part
// that doesn't need one: a first-order combinational-depth model that
// predicts *which* structure in ml_kernel.cpp's accumulation loops is
// likely to be the worst path, and quantifies the Fmax gain retiming it
// into a balanced tree should buy, before ever touching Vivado.
//
// ml_kernel_mlp (fpga_engine/ml_kernel/ml_kernel.cpp) fully unrolls both
// layers' inner loop under a single PIPELINE II=1 region:
//     acc_t acc = 0;
//     for (...) { #pragma HLS UNROLL   acc += in[i] * w[i][h]; }
// N parallel multiplies feed a sequential add chain in source order.
// Vitis HLS's scheduler *can* retime this automatically under a clock
// period constraint, but its pre-place delay estimates are optimistic
// (no real routing congestion, no SLR crossings) -- so getting the
// combinational-depth story right in source, rather than trusting the
// tool to paper over it at P&R time, is the thing worth doing before F1
// time is spent iterating in Vivado.
//
// This model deliberately does NOT claim precise absolute ns numbers --
// see kStageDelayLowNs/kStageDelayHighNs below, both first-order
// engineering approximations, not datasheet quotes. What it does claim,
// and what does not depend on getting the constant exactly right, is the
// *relative* logic-depth reduction: replacing an N-term flat accumulate
// chain with a balanced binary tree cuts depth from O(N) to O(log2 N)
// stages -- for N=32 (ml_kernel's layer2) that's 31 stages -> 5 stages.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

namespace {

struct ReductionShape {
    const char* label;
    int n_terms;
};

// One INT8 x INT8 multiply, mapped to a DSP48E2, is the first stage of
// every path regardless of reduction structure -- it doesn't change the
// flat-vs-tree comparison, so it's added once as a constant offset.
constexpr double kMultiplyDelayNs = 1.5;

// Per-stage delay for combining two partial sums (one adder level),
// modeled at two points to show the flat-vs-tree gap holds across a
// plausible range rather than hinging on one guessed number:
//   low  = an uncongested LUT carry-chain adder, best case
//   high = same adder post-place, with real routing/congestion delay
constexpr double kStageDelayLowNs = 0.30;
constexpr double kStageDelayHighNs = 0.60;

int flat_chain_depth(int n_terms) {
    return n_terms - 1; // N parallel multiplies feed an (N-1)-deep add chain
}

int tree_depth(int n_terms) {
    return static_cast<int>(std::ceil(std::log2(static_cast<double>(n_terms))));
}

double path_delay_ns(int depth, double stage_delay_ns) {
    return kMultiplyDelayNs + depth * stage_delay_ns;
}

double fmax_mhz(double delay_ns) {
    return 1000.0 / delay_ns;
}

void report(const ReductionShape& shape, double stage_delay_ns, const char* stage_label) {
    int flat_d = flat_chain_depth(shape.n_terms);
    int tree_d = tree_depth(shape.n_terms);
    double flat_delay = path_delay_ns(flat_d, stage_delay_ns);
    double tree_delay = path_delay_ns(tree_d, stage_delay_ns);

    std::printf(
        "%-7s (N=%2d) | stage=%.2fns (%-11s) | flat: depth=%2d delay=%5.2fns Fmax=%4.0fMHz"
        " | tree: depth=%2d delay=%5.2fns Fmax=%4.0fMHz | speedup=%.2fx\n",
        shape.label, shape.n_terms, stage_delay_ns, stage_label,
        flat_d, flat_delay, fmax_mhz(flat_delay),
        tree_d, tree_delay, fmax_mhz(tree_delay),
        flat_delay / tree_delay);
}

void report_wns(const ReductionShape& shape, double stage_delay_ns, const char* stage_label,
                 double target_mhz) {
    double period_ns = 1000.0 / target_mhz;
    int flat_d = flat_chain_depth(shape.n_terms);
    int tree_d = tree_depth(shape.n_terms);
    double flat_delay = path_delay_ns(flat_d, stage_delay_ns);
    double tree_delay = path_delay_ns(tree_d, stage_delay_ns);
    double flat_wns = period_ns - flat_delay;
    double tree_wns = period_ns - tree_delay;
    std::printf(
        "%-7s @ %.0fMHz (period=%.2fns, stage=%-11s) | flat WNS=%+6.2fns (%s) | tree WNS=%+6.2fns (%s)\n",
        shape.label, target_mhz, period_ns, stage_label,
        flat_wns, flat_wns >= 0 ? "MEETS" : "FAILS",
        tree_wns, tree_wns >= 0 ? "MEETS" : "FAILS");
}

} // namespace

int main() {
    const std::vector<ReductionShape> shapes = {
        {"layer1", 16}, // ml_kernel_mlp: 16 inputs -> hidden, per-output reduction
        {"layer2", 32}, // ml_kernel_mlp: 32 hidden -> output, per-output reduction
    };

    std::printf("=== flat accumulate chain vs. balanced binary tree ===\n");
    for (const auto& shape : shapes) {
        report(shape, kStageDelayLowNs, "uncongested");
        report(shape, kStageDelayHighNs, "congested");
    }

    std::printf("\n=== timing closure at a 300MHz target clock ===\n");
    for (const auto& shape : shapes) {
        report_wns(shape, kStageDelayLowNs, "uncongested", 300.0);
        report_wns(shape, kStageDelayHighNs, "congested", 300.0);
    }

    return 0;
}
