// hbm_sram_model.cpp — double-buffered HBM->VMEM streaming overlap model.
//
// PLAN.md Phase 8 step 6: "understand that TPU has no hardware cache: the
// compiler (XLA) schedules all data movement." Unlike a CPU/GPU, there is
// no hardware prefetcher or cache hierarchy hiding HBM latency behind
// speculation — every tile that reaches the MXU crosses HBM -> VMEM (the
// on-chip scratchpad XLA schedules explicitly, analogous to a GPU's
// software-managed shared memory but with no hardware-managed fallback at
// all) via an explicit DMA the compiler inserted. This file models the
// resulting double-buffering tradeoff: whether MXU compute time for a tile
// is enough to hide the DMA time for the *next* tile, the same
// double-buffered compute/transfer overlap idea fpga_engine's DDR step
// applies to a PCIe DMA engine, just one memory level further in.
//
// This is a purely analytical model (roofline-style: compute time from
// peak MXU TFLOPS, transfer time from peak HBM GB/s, same published specs
// tpu_benchmarks/ uses) -- there's no real transfer to time without a TPU,
// so unlike fpga_engine's double-buffering step (which measured PCIe DMA
// for real, hardware being cheaply available there), this stays a model
// until step 2's real HBM bandwidth number exists to calibrate it against.

#include <algorithm>
#include <cstdio>
#include <cstdint>
#include <vector>

namespace {

constexpr double kPeakBf16Tflops = 275.0;  // TPU v4, matches tpu_benchmarks/
constexpr double kPeakHbmGBps = 1200.0;    // TPU v4, matches tpu_benchmarks/

struct TileResult {
    int64_t tile_m, tile_n, tile_k;
    double compute_us, transfer_us;
    bool compute_bound;
    double overlap_efficiency_pct;  // useful MXU time / wall time, double-buffered
};

// A matmul tile of tile_m x tile_n x tile_k: 2*m*n*k FLOPs, and one DMA
// bringing in the tile_m x tile_k (lhs) + tile_k x tile_n (rhs) slice —
// the output tile is assumed resident in VMEM across the K-reduction
// (accumulate-in-place), so it isn't re-transferred per K-step.
TileResult analyze(int64_t tile_m, int64_t tile_n, int64_t tile_k) {
    double flops = 2.0 * tile_m * tile_n * tile_k;
    double compute_us = flops / (kPeakBf16Tflops * 1e12) * 1e6;

    double bytes_in = 2.0 * (tile_m * tile_k + tile_k * tile_n);  // bf16 = 2 bytes
    double transfer_us = bytes_in / (kPeakHbmGBps * 1e9) * 1e6;

    bool compute_bound = compute_us >= transfer_us;
    // Double-buffered: wall time per tile is max(compute, transfer) once
    // steady-state is reached (the DMA for tile i+1 overlaps compute for
    // tile i). Overlap efficiency is how much of that wall time is useful
    // MXU work.
    double wall_us = std::max(compute_us, transfer_us);
    double overlap_eff = 100.0 * compute_us / wall_us;

    return {tile_m, tile_n, tile_k, compute_us, transfer_us, compute_bound, overlap_eff};
}

} // namespace

int main() {
    struct Case { const char *label; int64_t m, n, k; };
    const std::vector<Case> cases = {
        {"small tile 128x128x128",    128, 128, 128},
        {"medium tile 512x512x512",   512, 512, 512},
        {"large tile 2048x2048x2048", 2048, 2048, 2048},
        {"tall-skinny 4096x128x4096", 4096, 128, 4096},
        {"wide 128x4096x128",         128, 4096, 128},
    };

    std::printf("%-30s %12s %12s %10s %16s\n",
                "tile shape", "compute(us)", "transfer(us)", "bound", "overlap eff.");
    for (const auto &c : cases) {
        TileResult r = analyze(c.m, c.n, c.k);
        std::printf("%-30s %12.2f %12.2f %10s %15.1f%%\n",
                    c.label, r.compute_us, r.transfer_us,
                    r.compute_bound ? "compute" : "transfer",
                    r.overlap_efficiency_pct);
    }

    std::printf(
        "\nStructural claim (independent of the peak-spec constants above): "
        "a cubic-in-size tile (m=n=k) becomes compute-bound as size grows, "
        "since FLOPs scale as size^3 while transferred bytes scale as "
        "size^2 -- arithmetic intensity grows with tile size. A "
        "tall-skinny or wide tile (one dim held small) stays "
        "transfer-bound regardless of the other two dims growing, since "
        "the small dim caps arithmetic intensity. This is why XLA's tiling "
        "heuristics favor roughly cubic tiles when VMEM capacity allows -- "
        "the double-buffering win (step 6) only materializes once compute "
        "can actually hide the DMA.\n");
    return 0;
}
