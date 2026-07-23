// fixed_point_matmul.cpp — ap_fixed<W,I> precision/resource/latency study
//
// One templated matmul body, instantiated at three fixed-point precisions
// as three separate HLS top functions (Vitis HLS top-level kernels must be
// concrete, not templates, so each instantiation gets its own synthesizable
// entry point) — same algorithm, only W/I differ, isolating the precision
// choice's effect on resources/latency/accuracy.
//
// TODO: synthesize on F1 with Vitis HLS. Untested — no toolchain locally,
// so no resource numbers or accuracy-vs-float error is measured yet.

#include <ap_fixed.h>

static constexpr int kN = 8; // NxN matmul — small enough to fully unroll/pipeline

// Templated body: C = A * B for NxN matrices of fixed-point type T.
// Accumulation happens in T itself (not a wider intermediate type) so that
// bit growth from repeated MACs is part of what gets measured — a design
// that accumulates in the same width as its inputs is exactly the case
// where overflow/precision tradeoffs matter most.
template <typename T>
static void matmul_fixed(const T a[kN][kN], const T b[kN][kN], T c[kN][kN]) {
    for (int i = 0; i < kN; ++i) {
        for (int j = 0; j < kN; ++j) {
#pragma HLS PIPELINE II=1
            T acc = 0;
            for (int k = 0; k < kN; ++k) {
#pragma HLS UNROLL
                acc += a[i][k] * b[k][j];
            }
            c[i][j] = acc;
        }
    }
}

// Narrow: 8-bit total, 4 integer bits -> 4 fractional bits. Cheapest in
// resources, most exposed to quantization error, most likely to overflow
// on inputs near the representable range.
using fixed8_t = ap_fixed<8, 4>;
void fixed_point_matmul_8(const fixed8_t a[kN][kN], const fixed8_t b[kN][kN],
                           fixed8_t c[kN][kN]) {
#pragma HLS INTERFACE m_axi port=a bundle=MAXI0
#pragma HLS INTERFACE m_axi port=b bundle=MAXI1
#pragma HLS INTERFACE m_axi port=c bundle=MAXI2
#pragma HLS INTERFACE s_axilite port=return
    matmul_fixed<fixed8_t>(a, b, c);
}

// Medium: 16-bit total, 6 integer bits. The usual default choice for
// ML inference weights/activations when INT8 isn't precise enough.
using fixed16_t = ap_fixed<16, 6>;
void fixed_point_matmul_16(const fixed16_t a[kN][kN], const fixed16_t b[kN][kN],
                            fixed16_t c[kN][kN]) {
#pragma HLS INTERFACE m_axi port=a bundle=MAXI0
#pragma HLS INTERFACE m_axi port=b bundle=MAXI1
#pragma HLS INTERFACE m_axi port=c bundle=MAXI2
#pragma HLS INTERFACE s_axilite port=return
    matmul_fixed<fixed16_t>(a, b, c);
}

// Wide: 32-bit total, 10 integer bits — close to float32's usable dynamic
// range for this workload, expected to match float accuracy closely while
// still being cheaper than a real floating-point multiplier/adder.
using fixed32_t = ap_fixed<32, 10>;
void fixed_point_matmul_32(const fixed32_t a[kN][kN], const fixed32_t b[kN][kN],
                            fixed32_t c[kN][kN]) {
#pragma HLS INTERFACE m_axi port=a bundle=MAXI0
#pragma HLS INTERFACE m_axi port=b bundle=MAXI1
#pragma HLS INTERFACE m_axi port=c bundle=MAXI2
#pragma HLS INTERFACE s_axilite port=return
    matmul_fixed<fixed32_t>(a, b, c);
}
