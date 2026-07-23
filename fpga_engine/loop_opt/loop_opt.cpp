// loop_opt.cpp — UNROLL / PIPELINE / DATAFLOW comparison
//
// Same workload (block-wise multiply-accumulate: out[i] = a[i]*b[i] + c[i]
// over BLOCK_SIZE elements) implemented four ways, so each pragma's effect
// can be isolated instead of measured in combination with the others.
//
// TODO: synthesize on F1 with Vitis HLS. Untested — no toolchain locally,
// so none of the II/resource/Fmax numbers below are known yet.

#include <hls_stream.h>

static constexpr int kBlockSize = 256;
static constexpr int kUnrollFactor = 4;

// 1. Baseline: no pragmas. Vitis HLS's default scheduling — sequential,
// one iteration's worth of work fully retired before the next starts.
void loop_opt_baseline(const float* a, const float* b, const float* c, float* out) {
#pragma HLS INTERFACE m_axi port=a   bundle=MAXI0 depth=256
#pragma HLS INTERFACE m_axi port=b   bundle=MAXI1 depth=256
#pragma HLS INTERFACE m_axi port=c   bundle=MAXI2 depth=256
#pragma HLS INTERFACE m_axi port=out bundle=MAXI3 depth=256
#pragma HLS INTERFACE s_axilite port=return

    for (int i = 0; i < kBlockSize; ++i) {
        out[i] = a[i] * b[i] + c[i];
    }
}

// 2. PIPELINE only: overlaps loop iterations in the same hardware — one
// iteration's worth of *new* work can start every cycle (II=1 target) even
// though a single iteration's result still takes multiple cycles to appear
// (pipeline depth = per-iteration latency). Costs control logic to track
// multiple in-flight iterations; does not duplicate the multiply/add units.
void loop_opt_pipeline(const float* a, const float* b, const float* c, float* out) {
#pragma HLS INTERFACE m_axi port=a   bundle=MAXI0 depth=256
#pragma HLS INTERFACE m_axi port=b   bundle=MAXI1 depth=256
#pragma HLS INTERFACE m_axi port=c   bundle=MAXI2 depth=256
#pragma HLS INTERFACE m_axi port=out bundle=MAXI3 depth=256
#pragma HLS INTERFACE s_axilite port=return

    for (int i = 0; i < kBlockSize; ++i) {
#pragma HLS PIPELINE II=1
        out[i] = a[i] * b[i] + c[i];
    }
}

// 3. UNROLL only (factor 4): duplicates the multiply/add datapath 4x and
// runs 4 iterations' worth of logic per loop trip, but — without PIPELINE —
// each unrolled trip still runs to completion before the next starts.
// Trades LUT/DSP (4x the arithmetic units) for fewer loop trips; does not
// by itself give the cycle-level overlap PIPELINE does.
void loop_opt_unroll(const float* a, const float* b, const float* c, float* out) {
#pragma HLS INTERFACE m_axi port=a   bundle=MAXI0 depth=256
#pragma HLS INTERFACE m_axi port=b   bundle=MAXI1 depth=256
#pragma HLS INTERFACE m_axi port=c   bundle=MAXI2 depth=256
#pragma HLS INTERFACE m_axi port=out bundle=MAXI3 depth=256
#pragma HLS INTERFACE s_axilite port=return

    for (int i = 0; i < kBlockSize; ++i) {
#pragma HLS UNROLL factor=kUnrollFactor
        out[i] = a[i] * b[i] + c[i];
    }
}

// 4. DATAFLOW: splits the same workload into three task-level stages
// (load into a stream, compute, store from a stream) that Vitis HLS runs
// as concurrent hardware blocks connected by FIFOs, so stage N+1 can start
// consuming stage N's output before stage N has finished the whole block —
// pipelining across *function calls*, not just loop iterations. Costs FIFO
// depth (BRAM) between stages instead of duplicated arithmetic units.
static void load_stage(const float* a, const float* b, const float* c,
                        hls::stream<float>& sa, hls::stream<float>& sb,
                        hls::stream<float>& sc) {
    for (int i = 0; i < kBlockSize; ++i) {
#pragma HLS PIPELINE II=1
        sa.write(a[i]);
        sb.write(b[i]);
        sc.write(c[i]);
    }
}

static void compute_stage(hls::stream<float>& sa, hls::stream<float>& sb,
                           hls::stream<float>& sc, hls::stream<float>& sout) {
    for (int i = 0; i < kBlockSize; ++i) {
#pragma HLS PIPELINE II=1
        sout.write(sa.read() * sb.read() + sc.read());
    }
}

static void store_stage(hls::stream<float>& sout, float* out) {
    for (int i = 0; i < kBlockSize; ++i) {
#pragma HLS PIPELINE II=1
        out[i] = sout.read();
    }
}

void loop_opt_dataflow(const float* a, const float* b, const float* c, float* out) {
#pragma HLS INTERFACE m_axi port=a   bundle=MAXI0 depth=256
#pragma HLS INTERFACE m_axi port=b   bundle=MAXI1 depth=256
#pragma HLS INTERFACE m_axi port=c   bundle=MAXI2 depth=256
#pragma HLS INTERFACE m_axi port=out bundle=MAXI3 depth=256
#pragma HLS INTERFACE s_axilite port=return
#pragma HLS DATAFLOW

    hls::stream<float> sa, sb, sc, sout;
#pragma HLS STREAM variable=sa depth=32
#pragma HLS STREAM variable=sb depth=32
#pragma HLS STREAM variable=sc depth=32
#pragma HLS STREAM variable=sout depth=32

    load_stage(a, b, c, sa, sb, sc);
    compute_stage(sa, sb, sc, sout);
    store_stage(sout, out);
}
