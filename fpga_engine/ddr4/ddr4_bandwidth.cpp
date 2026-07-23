// ddr4_bandwidth.cpp — multi-bank DDR4 bandwidth kernel
//
// F1's VU9P card has 4 independent DDR4 banks (~19.25 GB/s each, ~77 GB/s
// aggregate). A kernel that only ever touches one bank can never exceed
// that bank's ~19.25 GB/s regardless of how well everything else is
// optimized — this kernel exists to prove the multi-bank path: four
// independent streaming copies, one per bank, run concurrently via
// DATAFLOW so their DMA traffic overlaps instead of serializing.
//
// TODO: synthesize + link on F1 with Vitis HLS/v++, using ddr4.cfg to bind
// each m_axi bundle to a distinct DDR4 bank. Untested — no toolchain
// locally, so no per-bank or aggregate bandwidth number is known yet.

static constexpr int kElemsPerBank = 1 << 20; // 1M x 4B = 4MB per bank, 16MB total

static void copy_bank(const int* in, int* out, int n) {
    for (int i = 0; i < n; ++i) {
#pragma HLS PIPELINE II=1
        out[i] = in[i];
    }
}

// Four independent m_axi bundles — MAXI0..MAXI7, in/out pairs per bank.
// ddr4.cfg's `sp=` directives are what actually place bundle N on DDR
// bank N; the kernel side only needs distinct bundle names to give v++
// something to bind independently.
void ddr4_bandwidth(
    const int* in0, int* out0,
    const int* in1, int* out1,
    const int* in2, int* out2,
    const int* in3, int* out3
) {
#pragma HLS INTERFACE m_axi port=in0  bundle=MAXI0 depth=1048576
#pragma HLS INTERFACE m_axi port=out0 bundle=MAXI1 depth=1048576
#pragma HLS INTERFACE m_axi port=in1  bundle=MAXI2 depth=1048576
#pragma HLS INTERFACE m_axi port=out1 bundle=MAXI3 depth=1048576
#pragma HLS INTERFACE m_axi port=in2  bundle=MAXI4 depth=1048576
#pragma HLS INTERFACE m_axi port=out2 bundle=MAXI5 depth=1048576
#pragma HLS INTERFACE m_axi port=in3  bundle=MAXI6 depth=1048576
#pragma HLS INTERFACE m_axi port=out3 bundle=MAXI7 depth=1048576
#pragma HLS INTERFACE s_axilite port=return
#pragma HLS DATAFLOW

    copy_bank(in0, out0, kElemsPerBank);
    copy_bank(in1, out1, kElemsPerBank);
    copy_bank(in2, out2, kElemsPerBank);
    copy_bank(in3, out3, kElemsPerBank);
}
