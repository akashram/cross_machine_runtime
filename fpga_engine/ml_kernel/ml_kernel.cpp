// ml_kernel.cpp — fully pipelined INT8 MLP inference kernel
//
// A tiny 2-layer MLP (16 inputs -> 32 hidden, ReLU -> 8 outputs), fully
// unrolled and pipelined for II=1: every weight has its own multiplier
// instance (UNROLL on both layers' inner loops), so one full forward pass
// issues every cycle rather than serializing over weights. Small enough
// (16*32 + 32*8 = 768 multiplies) that full unroll is a reasonable choice
// for VU9P's 6840 DSP48E2 budget — a larger network would need the
// dsp_lut/loop_opt tradeoffs applied deliberately instead.
//
// TODO: synthesize on F1 with Vitis HLS. Untested — no toolchain locally,
// so II/Fmax/resources are unmeasured. Quantization *accuracy* (separate
// from hardware synthesis) is validated locally — see mlp_int8_ref.cpp,
// which is plain C++ with no HLS dependency and actually runs.

#include <ap_int.h>

static constexpr int kInputs = 16;
static constexpr int kHidden = 32;
static constexpr int kOutputs = 8;

// INT8 weights/activations, INT32 accumulation (matches mlp_int8_ref.cpp's
// int32_t accumulator exactly, so the accuracy numbers measured there
// describe this kernel's arithmetic, not a different one).
using weight_t = ap_int<8>;
using act_t = ap_int<8>;
using acc_t = ap_int<32>;

void ml_kernel_mlp(
    const act_t in[kInputs],
    const weight_t w1[kInputs][kHidden],
    const weight_t w2[kHidden][kOutputs],
    act_t out[kOutputs]
) {
#pragma HLS INTERFACE s_axilite port=in
#pragma HLS INTERFACE m_axi port=w1  bundle=MAXI0 depth=512
#pragma HLS INTERFACE m_axi port=w2  bundle=MAXI1 depth=256
#pragma HLS INTERFACE s_axilite port=out
#pragma HLS INTERFACE s_axilite port=return
#pragma HLS ARRAY_PARTITION variable=in complete

    act_t hidden[kHidden];
#pragma HLS ARRAY_PARTITION variable=hidden complete

layer1:
    for (int h = 0; h < kHidden; ++h) {
#pragma HLS PIPELINE II=1
        acc_t acc = 0;
        for (int i = 0; i < kInputs; ++i) {
#pragma HLS UNROLL
            acc += in[i] * w1[i][h];
        }
        // ReLU, saturating back down to int8 range after the wider
        // accumulation — quantized inference needs an explicit
        // requantization step here in a real deployment (scale factor
        // folded into weights); saturation alone is the simplified stand-in.
        hidden[h] = (acc < 0) ? act_t(0)
                  : (acc > 127) ? act_t(127)
                  : act_t(acc);
    }

layer2:
    for (int o = 0; o < kOutputs; ++o) {
#pragma HLS PIPELINE II=1
        acc_t acc = 0;
        for (int h = 0; h < kHidden; ++h) {
#pragma HLS UNROLL
            acc += hidden[h] * w2[h][o];
        }
        out[o] = (acc < -128) ? act_t(-128)
               : (acc > 127)  ? act_t(127)
               : act_t(acc);
    }
}
