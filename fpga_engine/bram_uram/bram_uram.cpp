// bram_uram.cpp — BRAM (true dual-port) vs URAM (single-port) weight storage
//
// Same weight-lookup workload against a large on-chip weight table, forced
// into BRAM in one variant and URAM in the other via BIND_STORAGE, with an
// access pattern chosen to actually stress the port-count difference: two
// independent reads per cycle. BRAM (36Kb, true dual-port) should sustain
// that natively; URAM (288Kb, single-port) should have to arbitrate the two
// reads across cycles, which is exactly the tradeoff this step measures —
// URAM's much larger capacity per block against its single-port limit.
//
// TODO: synthesize on F1 with Vitis HLS. Untested — no toolchain locally.

#include <ap_int.h>

static constexpr int kWeightCount = 16384; // large enough to matter for BRAM vs URAM capacity
using weight_t = ap_int<16>;

// BRAM variant: 36Kb blocks, true dual-port — two independent read ports
// natively supported in one cycle, no arbitration. At 16384 x 16-bit =
// 256Kb, this already spans multiple BRAM36 blocks (each holds 36Kb / 16b
// = ~2K entries), so part of what's measured here is also BRAM block count.
void bram_uram_bram(const weight_t* weights_in, const int* idx_a,
                     const int* idx_b, weight_t* out_a, weight_t* out_b,
                     int n) {
#pragma HLS INTERFACE m_axi port=weights_in bundle=MAXI0 depth=16384
#pragma HLS INTERFACE m_axi port=idx_a      bundle=MAXI1 depth=4096
#pragma HLS INTERFACE m_axi port=idx_b      bundle=MAXI2 depth=4096
#pragma HLS INTERFACE m_axi port=out_a      bundle=MAXI3 depth=4096
#pragma HLS INTERFACE m_axi port=out_b      bundle=MAXI4 depth=4096
#pragma HLS INTERFACE s_axilite port=n
#pragma HLS INTERFACE s_axilite port=return

    static weight_t weights[kWeightCount];
#pragma HLS BIND_STORAGE variable=weights type=RAM_2P impl=BRAM

    for (int i = 0; i < kWeightCount; ++i) {
#pragma HLS PIPELINE II=1
        weights[i] = weights_in[i];
    }

    for (int i = 0; i < n; ++i) {
#pragma HLS PIPELINE II=1
        // Two independent lookups per cycle — the access pattern BRAM's
        // true dual-port is meant to serve without stalling.
        out_a[i] = weights[idx_a[i]];
        out_b[i] = weights[idx_b[i]];
    }
}

// URAM variant: 288Kb blocks, single read/write port. Same two-lookups-
// per-cycle pattern, but URAM can only actually issue one; Vitis HLS must
// either serialize the two reads (doubling II for this loop) or the
// mapping has to change. Left as a single port on purpose — this step
// exists to measure exactly that II penalty, not to hide it by only
// issuing one lookup per cycle for the URAM variant.
void bram_uram_uram(const weight_t* weights_in, const int* idx_a,
                     const int* idx_b, weight_t* out_a, weight_t* out_b,
                     int n) {
#pragma HLS INTERFACE m_axi port=weights_in bundle=MAXI0 depth=16384
#pragma HLS INTERFACE m_axi port=idx_a      bundle=MAXI1 depth=4096
#pragma HLS INTERFACE m_axi port=idx_b      bundle=MAXI2 depth=4096
#pragma HLS INTERFACE m_axi port=out_a      bundle=MAXI3 depth=4096
#pragma HLS INTERFACE m_axi port=out_b      bundle=MAXI4 depth=4096
#pragma HLS INTERFACE s_axilite port=n
#pragma HLS INTERFACE s_axilite port=return

    static weight_t weights[kWeightCount];
#pragma HLS BIND_STORAGE variable=weights type=RAM_1P impl=URAM

    for (int i = 0; i < kWeightCount; ++i) {
#pragma HLS PIPELINE II=1
        weights[i] = weights_in[i];
    }

    for (int i = 0; i < n; ++i) {
#pragma HLS PIPELINE II=1
        out_a[i] = weights[idx_a[i]];
        out_b[i] = weights[idx_b[i]];
    }
}
