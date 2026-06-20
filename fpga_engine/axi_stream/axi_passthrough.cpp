// axi_passthrough.cpp — AXI4-Stream passthrough kernel for interface validation
// TODO: synthesize on F1 with Vitis HLS

#include <hls_stream.h>
#include <ap_axi_sdata.h>

typedef ap_axiu<32, 1, 1, 1> axi_word_t;

// Passthrough: copy input AXI stream to output unchanged
void axi_passthrough(
    hls::stream<axi_word_t>& in_stream,
    hls::stream<axi_word_t>& out_stream
) {
#pragma HLS INTERFACE axis port=in_stream
#pragma HLS INTERFACE axis port=out_stream
#pragma HLS INTERFACE s_axilite port=return

    axi_word_t word;
    do {
#pragma HLS PIPELINE II=1
        word = in_stream.read();
        out_stream.write(word);
    } while (!word.last);
}
