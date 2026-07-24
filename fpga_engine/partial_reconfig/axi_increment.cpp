// axi_increment.cpp — second AXI4-Stream kernel, same port interface as
// axi_stream/axi_passthrough.cpp, for use as the alternate reconfigurable
// module (RM_B) in dfx_pblock.tcl's partial-reconfiguration flow.
//
// A DFX (Dynamic Function eXchange) reconfigurable partition can only
// swap between modules that present the identical interface to the
// static shell around them -- same stream types, same port names, same
// AXI4-Stream side-channel widths. This kernel exists purely to be that
// second, interface-compatible RM: axi_passthrough forwards each word
// unchanged (RM_A), this increments the data field by one before
// forwarding it (RM_B), so hot-swapping between them produces an
// observable functional difference pr_host_driver.cpp / a downstream
// cocotb-style test could check for, not just a bitstream that happens
// to load.
// TODO: synthesize on F1 with Vitis HLS (same as axi_passthrough.cpp).

#include <hls_stream.h>
#include <ap_axi_sdata.h>

typedef ap_axiu<32, 1, 1, 1> axi_word_t;

// Increment: add 1 to each input word's data, forward tlast unchanged.
void axi_increment(
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
        word.data = word.data + 1;
        out_stream.write(word);
    } while (!word.last);
}
