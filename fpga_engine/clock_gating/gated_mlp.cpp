// gated_mlp.cpp — ml_kernel_mlp variant with an explicit valid-gated
// register bank, the source-level half of clock_gating's fix.
//
// ml_kernel_mlp (fpga_engine/ml_kernel/ml_kernel.cpp) writes `hidden[h]`
// and `out[o]` unconditionally every PIPELINE II=1 iteration. Vivado's
// `-gated_clock_conversion` (see power_gating.tcl) looks for a register
// bank sharing a common write-enable and promotes it to a real gated
// clock -- but there's nothing to find if every write is unconditional.
// This variant adds one input, `valid`, and every register write is
// conditioned on it, which is exactly the "common enable across a
// register bank" pattern gated_clock_conversion is documented to detect
// and convert.
//
// Whether this is worth adding to a given deployment depends entirely on
// duty cycle -- see clock_gating_model.cpp, which predicts a net *loss*
// if `valid` is asserted every cycle (nothing to gate, pure BUFGCE
// overhead) and a large win if it's bursty. That's exactly why this is a
// separate file rather than a change to ml_kernel_mlp itself: it's not a
// strict improvement, it's a traffic-pattern-dependent tradeoff, and
// step 14's flat version stays correct as the always-active baseline.
//
// TODO: synthesize on F1 with and without -gated_clock_conversion, using
// power_gating.tcl, and compare against clock_gating_model.cpp's
// predicted reduction at the traffic pattern actually measured.

#include <ap_int.h>

static constexpr int kInputs = 16;
static constexpr int kHidden = 32;
static constexpr int kOutputs = 8;

using weight_t = ap_int<8>;
using act_t = ap_int<8>;
using acc_t = ap_int<32>;

void gated_mlp(
    bool         valid,
    const act_t  in[kInputs],
    const weight_t w1[kInputs][kHidden],
    const weight_t w2[kHidden][kOutputs],
    act_t        out[kOutputs]
) {
#pragma HLS INTERFACE s_axilite port=valid
#pragma HLS INTERFACE s_axilite port=in
#pragma HLS INTERFACE m_axi port=w1  bundle=MAXI0 depth=512
#pragma HLS INTERFACE m_axi port=w2  bundle=MAXI1 depth=256
#pragma HLS INTERFACE s_axilite port=out
#pragma HLS INTERFACE s_axilite port=return
#pragma HLS ARRAY_PARTITION variable=in complete

    static act_t hidden[kHidden];
#pragma HLS ARRAY_PARTITION variable=hidden complete

layer1:
    for (int h = 0; h < kHidden; ++h) {
#pragma HLS PIPELINE II=1
        if (valid) {
            acc_t acc = 0;
            for (int i = 0; i < kInputs; ++i) {
#pragma HLS UNROLL
                acc += in[i] * w1[i][h];
            }
            hidden[h] = (acc < 0) ? act_t(0)
                      : (acc > 127) ? act_t(127)
                      : act_t(acc);
        }
        // else: hold hidden[h] at its previous value -- this is the
        // conditional write that gives the whole hidden[] register bank
        // a common enable (valid) for gated_clock_conversion to find.
    }

layer2:
    for (int o = 0; o < kOutputs; ++o) {
#pragma HLS PIPELINE II=1
        if (valid) {
            acc_t acc = 0;
            for (int h = 0; h < kHidden; ++h) {
#pragma HLS UNROLL
                acc += hidden[h] * w2[h][o];
            }
            out[o] = (acc < -128) ? act_t(-128)
                   : (acc > 127)  ? act_t(127)
                   : act_t(acc);
        }
        // else: hold out[o] -- same common-enable pattern as hidden[].
    }
}
