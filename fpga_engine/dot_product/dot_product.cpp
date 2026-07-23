// dot_product.cpp — Vitis HLS dot product kernel
//
// Two variants, both exported so tcl_pipeline can build them separately and
// loop_opt.md's II comparison has a real baseline: `dot_product_naive`
// (unpipelined) and `dot_product` (II=1 target).
//
// TODO: synthesize + cosimulate on F1 with Vitis HLS. Untested — no Vitis
// toolchain on this Mac, so neither the achieved II nor Fmax is known yet.

#include <hls_stream.h>
#include <ap_fixed.h>

// Naive baseline: a single floating-point accumulator creates a loop-carried
// dependency through the FP adder's own pipeline latency (~5 cycles on
// UltraScale+ DSP48E2-backed float add). Without breaking that chain, II
// cannot reach 1 no matter what pragma is attached to the loop — the adder
// itself hasn't produced this iteration's result before the next iteration
// needs it. This variant intentionally has no PIPELINE pragma, so its
// achieved II reflects Vitis HLS's default (II = FP adder latency).
void dot_product_naive(
    const float* a,
    const float* b,
    float*       result,
    int          n
) {
#pragma HLS INTERFACE m_axi port=a      bundle=MAXI0 depth=1024
#pragma HLS INTERFACE m_axi port=b      bundle=MAXI1 depth=1024
#pragma HLS INTERFACE m_axi port=result bundle=MAXI2 depth=1
#pragma HLS INTERFACE s_axilite port=n
#pragma HLS INTERFACE s_axilite port=return

    float acc = 0.0f;
    for (int i = 0; i < n; ++i) {
        acc += a[i] * b[i];
    }
    *result = acc;
}

// II=1 target: four independent partial accumulators break the
// loop-carried dependency chain that limits dot_product_naive — each
// accumulator only depends on its own value from 4 iterations ago, which
// gives the FP adder enough cycles to finish before that operand is needed
// again. PIPELINE II=1 can then actually schedule one (a[i], b[i]) pair
// consumed per cycle instead of stalling on adder latency. The four
// partials are summed once, outside the hot loop, so that combine cost is
// paid once rather than every iteration.
static constexpr int kNumAccumulators = 4;

void dot_product(
    const float* a,
    const float* b,
    float*       result,
    int          n
) {
#pragma HLS INTERFACE m_axi port=a      bundle=MAXI0 depth=1024
#pragma HLS INTERFACE m_axi port=b      bundle=MAXI1 depth=1024
#pragma HLS INTERFACE m_axi port=result bundle=MAXI2 depth=1
#pragma HLS INTERFACE s_axilite port=n
#pragma HLS INTERFACE s_axilite port=return

    float acc[kNumAccumulators];
#pragma HLS ARRAY_PARTITION variable=acc complete
    for (int k = 0; k < kNumAccumulators; ++k) {
#pragma HLS UNROLL
        acc[k] = 0.0f;
    }

    for (int i = 0; i < n; ++i) {
#pragma HLS PIPELINE II=1
        acc[i % kNumAccumulators] += a[i] * b[i];
    }

    float total = 0.0f;
    for (int k = 0; k < kNumAccumulators; ++k) {
#pragma HLS UNROLL
        total += acc[k];
    }
    *result = total;
}
