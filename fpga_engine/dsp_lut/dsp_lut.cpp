// dsp_lut.cpp — DSP48E2 vs LUT-fabric multiply tradeoff
//
// Identical fixed-point multiply-accumulate, forced onto DSP48E2 blocks in
// one variant and onto LUT fabric in the other via #pragma HLS BIND_OP —
// same algorithm, only the implementation binding differs, so any resource
// difference measured is attributable to that choice alone.
//
// TODO: synthesize on F1 with Vitis HLS. Untested — no toolchain locally.

#include <ap_fixed.h>

static constexpr int kBlockSize = 256;
using fixed_t = ap_fixed<16, 4>;

// DSP-bound: uses DSP48E2's dedicated multiplier (pre-adder + 27x18
// multiplier + accumulator hardware). Near-zero LUT cost per multiply, but
// DSP48E2 count on VU9P is finite (6840) — a design with enough parallel
// multiplies eventually runs out of DSPs before it runs out of LUTs.
void dsp_lut_dsp(const fixed_t* a, const fixed_t* b, fixed_t* out) {
#pragma HLS INTERFACE m_axi port=a   bundle=MAXI0 depth=256
#pragma HLS INTERFACE m_axi port=b   bundle=MAXI1 depth=256
#pragma HLS INTERFACE m_axi port=out bundle=MAXI2 depth=256
#pragma HLS INTERFACE s_axilite port=return

    for (int i = 0; i < kBlockSize; ++i) {
#pragma HLS PIPELINE II=1
        fixed_t result = a[i] * b[i];
#pragma HLS BIND_OP variable=result op=mul impl=dsp
        out[i] = result;
    }
}

// LUT-bound: same multiply, forced onto general logic fabric instead of
// DSP48E2. Costs LUTs/FFs (a 16x16-bit fixed multiply synthesized in LUTs
// is a real amount of combinational logic) in exchange for leaving DSP
// blocks free for other kernels sharing the same device, and avoiding DSP
// cascade routing congestion on designs with many small multiplies.
void dsp_lut_fabric(const fixed_t* a, const fixed_t* b, fixed_t* out) {
#pragma HLS INTERFACE m_axi port=a   bundle=MAXI0 depth=256
#pragma HLS INTERFACE m_axi port=b   bundle=MAXI1 depth=256
#pragma HLS INTERFACE m_axi port=out bundle=MAXI2 depth=256
#pragma HLS INTERFACE s_axilite port=return

    for (int i = 0; i < kBlockSize; ++i) {
#pragma HLS PIPELINE II=1
        fixed_t result = a[i] * b[i];
#pragma HLS BIND_OP variable=result op=mul impl=fabric
        out[i] = result;
    }
}
