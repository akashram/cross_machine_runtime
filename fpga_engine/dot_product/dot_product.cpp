// dot_product.cpp — Vitis HLS dot product kernel
// TODO: synthesize on F1 with Vivado/Vitis HLS
// Target: II=1 after pipelining, ~450 MHz Fmax on UltraScale+

#include <hls_stream.h>
#include <ap_fixed.h>

// TODO: implement on F1 with Vitis HLS
void dot_product(
    const float* a,      // input vector A (m_axi port — DDR4)
    const float* b,      // input vector B (m_axi port — DDR4)
    float*       result, // output scalar (m_axi port — DDR4)
    int          n       // vector length
) {
#pragma HLS INTERFACE m_axi port=a     bundle=MAXI0 depth=1024
#pragma HLS INTERFACE m_axi port=b     bundle=MAXI1 depth=1024
#pragma HLS INTERFACE m_axi port=result bundle=MAXI2 depth=1
#pragma HLS INTERFACE s_axilite port=n
#pragma HLS INTERFACE s_axilite port=return

    float acc = 0.0f;
    // TODO: add HLS PIPELINE pragma, unroll inner loop
    for (int i = 0; i < n; ++i) {
#pragma HLS PIPELINE II=1
        acc += a[i] * b[i];
    }
    *result = acc;
}
