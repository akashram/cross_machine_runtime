// pipelined_tree_reduce.cpp — explicit staged reduction, the source-level
// retiming fix motivated by critical_path_model.cpp's numbers.
//
// ml_kernel_mlp's layer1/layer2 loops (fpga_engine/ml_kernel/ml_kernel.cpp)
// write their reduction as a single unrolled expression:
//     acc_t acc = 0;
//     for (...) { #pragma HLS UNROLL   acc += in[i] * w[i][h]; }
// Written this way, Vitis HLS has one combinational expression to
// schedule, and per critical_path_model.cpp a flat O(N) add chain is the
// likely result unless the tool infers a balanced tree on its own. The
// model further shows a *combinational* tree alone isn't always enough
// (layer1/layer2 both still fail a 300MHz target under the "congested"
// delay assumption) -- the fix needs concrete statement boundaries the
// scheduler can place pipeline registers at, not just fewer logic levels.
//
// Splitting the same reduction across explicit intermediate arrays, one
// per tree level, gives Vitis HLS exactly that: each level is its own
// statement writing a fully-partitioned array, so the scheduler has a
// natural per-level cut point to register against when it schedules
// against the HLS project's target clock period, instead of one giant
// expression it must fully resolve inside a single cycle.
//
// Two versions, matching ml_kernel_mlp's exact widths (16 for layer1, 32
// for layer2) so a resource/timing comparison against the original flat
// accumulate is apples-to-apples if/when this runs through Vitis HLS.
// Deliberately NOT wired into ml_kernel.cpp itself -- step 14 is already
// a complete, committed deliverable, and this fix is unverified against
// real hardware; folding it in would silently change a finished step's
// behavior based on a model, not a measurement. Whoever picks up
// ml_kernel again after F1 timing numbers come back can adopt this
// pattern once report_timing confirms the gain actually materializes.
//
// TODO: synthesize on F1, compare report_timing/report_utilization
// against ml_kernel_mlp's flat version.

#include <ap_int.h>

using acc_t = ap_int<32>;

// N=16 (ml_kernel_mlp layer1): 4 explicit levels, 16 -> 8 -> 4 -> 2 -> 1.
acc_t tree_reduce16(const acc_t products[16]) {
#pragma HLS ARRAY_PARTITION variable=products complete
    acc_t l1[8], l2[4], l3[2];
#pragma HLS ARRAY_PARTITION variable=l1 complete
#pragma HLS ARRAY_PARTITION variable=l2 complete
#pragma HLS ARRAY_PARTITION variable=l3 complete

l1_stage:
    for (int i = 0; i < 8; ++i) {
#pragma HLS UNROLL
        l1[i] = products[2 * i] + products[2 * i + 1];
    }
l2_stage:
    for (int i = 0; i < 4; ++i) {
#pragma HLS UNROLL
        l2[i] = l1[2 * i] + l1[2 * i + 1];
    }
l3_stage:
    for (int i = 0; i < 2; ++i) {
#pragma HLS UNROLL
        l3[i] = l2[2 * i] + l2[2 * i + 1];
    }
    return l3[0] + l3[1];
}

// N=32 (ml_kernel_mlp layer2): 5 explicit levels, 32 -> 16 -> 8 -> 4 -> 2 -> 1.
acc_t tree_reduce32(const acc_t products[32]) {
#pragma HLS ARRAY_PARTITION variable=products complete
    acc_t l1[16], l2[8], l3[4], l4[2];
#pragma HLS ARRAY_PARTITION variable=l1 complete
#pragma HLS ARRAY_PARTITION variable=l2 complete
#pragma HLS ARRAY_PARTITION variable=l3 complete
#pragma HLS ARRAY_PARTITION variable=l4 complete

l1_stage:
    for (int i = 0; i < 16; ++i) {
#pragma HLS UNROLL
        l1[i] = products[2 * i] + products[2 * i + 1];
    }
l2_stage:
    for (int i = 0; i < 8; ++i) {
#pragma HLS UNROLL
        l2[i] = l1[2 * i] + l1[2 * i + 1];
    }
l3_stage:
    for (int i = 0; i < 4; ++i) {
#pragma HLS UNROLL
        l3[i] = l2[2 * i] + l2[2 * i + 1];
    }
l4_stage:
    for (int i = 0; i < 2; ++i) {
#pragma HLS UNROLL
        l4[i] = l3[2 * i] + l3[2 * i + 1];
    }
    return l4[0] + l4[1];
}
