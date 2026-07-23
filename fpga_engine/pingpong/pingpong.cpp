// pingpong.cpp — double-buffered compute/transfer overlap
//
// Processes a large DDR4-resident array in blocks too big to stream
// element-by-element (unlike loop_opt's DATAFLOW variant, which streams
// single elements through 3 stages continuously). Instead, each block is
// staged into on-chip BRAM before compute runs on it, and two on-chip
// buffers alternate (ping/pong) so the DMA load of block i+1 overlaps the
// compute+store of block i rather than the two serializing.
//
// TODO: synthesize on F1 with Vitis HLS. Untested — no toolchain locally,
// so the actual throughput improvement over a single-buffered baseline
// (which the DATAFLOW scheduler wouldn't be able to overlap) is unmeasured.

static constexpr int kBlockSize = 1024;
static constexpr int kNumBlocks = 64;
static constexpr int kTotal = kBlockSize * kNumBlocks;

// Single-buffered baseline: one on-chip buffer, reused every block. Load,
// compute, and store for block i must fully serialize before block i+1's
// load can start writing into the same buffer — no overlap possible no
// matter what DATAFLOW does, since there's only one copy of the storage.
void pingpong_single_buffered(const float* in, float* out) {
#pragma HLS INTERFACE m_axi port=in  bundle=MAXI0 depth=65536
#pragma HLS INTERFACE m_axi port=out bundle=MAXI1 depth=65536
#pragma HLS INTERFACE s_axilite port=return

    float buf[kBlockSize];

    for (int blk = 0; blk < kNumBlocks; ++blk) {
        for (int i = 0; i < kBlockSize; ++i) {
#pragma HLS PIPELINE II=1
            buf[i] = in[blk * kBlockSize + i];
        }
        for (int i = 0; i < kBlockSize; ++i) {
#pragma HLS PIPELINE II=1
            buf[i] = buf[i] * 2.0f + 1.0f; // stand-in compute
        }
        for (int i = 0; i < kBlockSize; ++i) {
#pragma HLS PIPELINE II=1
            out[blk * kBlockSize + i] = buf[i];
        }
    }
}

// Double-buffered (ping-pong): two on-chip buffers, block index selects
// which one is "active" this iteration. DATAFLOW lets Vitis HLS overlap
// the *next* block's load into buffer B with *this* block's compute+store
// out of buffer A, since they no longer touch the same storage — the
// throughput win this step exists to measure.
static void load_block(const float* in, float buf[2][kBlockSize], int blk) {
    for (int i = 0; i < kBlockSize; ++i) {
#pragma HLS PIPELINE II=1
        buf[blk % 2][i] = in[blk * kBlockSize + i];
    }
}

static void compute_store_block(float buf[2][kBlockSize], float* out, int blk) {
    for (int i = 0; i < kBlockSize; ++i) {
#pragma HLS PIPELINE II=1
        out[blk * kBlockSize + i] = buf[blk % 2][i] * 2.0f + 1.0f;
    }
}

void pingpong_double_buffered(const float* in, float* out) {
#pragma HLS INTERFACE m_axi port=in  bundle=MAXI0 depth=65536
#pragma HLS INTERFACE m_axi port=out bundle=MAXI1 depth=65536
#pragma HLS INTERFACE s_axilite port=return

    static float buf[2][kBlockSize];
#pragma HLS ARRAY_PARTITION variable=buf dim=1 complete
#pragma HLS BIND_STORAGE variable=buf type=RAM_2P impl=BRAM

    for (int blk = 0; blk < kNumBlocks; ++blk) {
#pragma HLS DATAFLOW
        load_block(in, buf, blk);
        compute_store_block(buf, out, blk);
    }
}
