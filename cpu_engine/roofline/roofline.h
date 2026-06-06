#pragma once

// Roofline model — analysis and ASCII chart
// ==========================================
//
// The roofline model (Williams et al., 2009) characterises every kernel's
// performance with two numbers:
//
//   Arithmetic Intensity (AI) = FLOPs / bytes_transferred   [FLOP/byte]
//   Achieved GFLOPS            = FLOPs / time                [GFLOPS]
//
// Two hardware ceilings bound what a kernel can achieve:
//
//   Bandwidth ceiling:  GFLOPS ≤ peak_bw_gbps × AI   (slope of the ridge)
//   Compute ceiling:    GFLOPS ≤ peak_flops_gflops    (flat top)
//
// The ridge point is the AI where the two ceilings meet:
//   AI_ridge = peak_flops / peak_bw_gbps
//
// Classification:
//   AI < AI_ridge  →  bandwidth-bound  (more FLOPs per byte needed to use the FPUs)
//   AI > AI_ridge  →  compute-bound    (already saturating the FPUs)
//
// Note: "bytes transferred" here means *minimum* bytes — i.e., each element
// loaded or stored exactly once.  A kernel with poor cache reuse has a higher
// actual byte count and therefore a lower effective AI.  That is precisely
// what the tiling step 8 was designed to improve.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace cpu_engine::roofline {

struct KernelPoint {
    std::string name;
    double      flops;             // FLOPs per call
    double      min_bytes;        // minimum bytes transferred per call
    double      achieved_gflops;  // measured

    double ai()              const noexcept { return flops / min_bytes; }
    double achieved_bytes_s(double ns) const noexcept { // for reference
        return min_bytes / (ns * 1e-9);
    }
};

struct RooflineModel {
    double peak_gflops;      // measured: FMA micro-benchmark
    double peak_bw_gbps;     // measured: STREAM Triad

    double ridge_point_ai()             const noexcept { return peak_gflops / peak_bw_gbps; }
    double ceiling_gflops(double ai)    const noexcept { return std::min(peak_gflops, peak_bw_gbps * ai); }
    bool   compute_bound(double ai)     const noexcept { return ai >= ridge_point_ai(); }
    double utilisation(double achieved, double ai) const noexcept {
        double ceil = ceiling_gflops(ai);
        return ceil > 0.0 ? achieved / ceil : 0.0;
    }

    // ---- Text table --------------------------------------------------------
    void print_table(const std::vector<KernelPoint>& kernels) const {
        printf("\nRoofline Analysis\n");
        printf("  Peak compute:   %6.1f GFLOPS\n", peak_gflops);
        printf("  Peak bandwidth: %6.1f GB/s\n",   peak_bw_gbps);
        printf("  Ridge point:    %6.2f FLOP/byte\n\n", ridge_point_ai());

        printf("  %-28s  %8s  %8s  %8s  %6s  %s\n",
               "Kernel", "AI(F/B)", "Achvd GF", "Ceil GF", "Util%", "Bound");
        printf("  %s\n", std::string(76, '-').c_str());

        for (const auto& k : kernels) {
            double ai   = k.ai();
            double ceil = ceiling_gflops(ai);
            double util = utilisation(k.achieved_gflops, ai);
            printf("  %-28s  %8.3f  %8.2f  %8.2f  %5.0f%%  %s\n",
                   k.name.c_str(), ai,
                   k.achieved_gflops, ceil,
                   util * 100.0,
                   compute_bound(ai) ? "compute" : "bandwidth");
        }
    }

    // ---- ASCII roofline chart ----------------------------------------------
    // X: log10(AI) from x_lo to x_hi, W columns
    // Y: GFLOPS from 0 to y_max, H rows (top = y_max)
    void print_chart(const std::vector<KernelPoint>& kernels) const {
        constexpr int W = 60;
        constexpr int H = 18;

        const double x_lo  = -1.5;          // log10(0.03)
        const double x_hi  =  2.5;          // log10(316)
        const double y_max = peak_gflops * 1.15;
        const double ridge = ridge_point_ai();

        // grid[row][col] is a display character; row 0 = top (max GFLOPS)
        std::vector<std::string> grid(static_cast<std::size_t>(H),
                                      std::string(static_cast<std::size_t>(W), ' '));

        auto x_col = [&](double ai) -> int {
            return static_cast<int>((std::log10(ai) - x_lo) / (x_hi - x_lo) * W);
        };
        auto y_row = [&](double gf) -> int {
            int r = H - 1 - static_cast<int>(gf / y_max * (H - 1));
            return std::clamp(r, 0, H - 1);
        };

        // Draw the roofline
        for (int col = 0; col < W; ++col) {
            double log_ai = x_lo + (x_hi - x_lo) * col / W;
            double ai     = std::pow(10.0, log_ai);
            double gf     = std::min(peak_gflops, peak_bw_gbps * ai);
            int    row    = y_row(gf);
            if (row >= 0 && row < H) {
                char ch = (ai < ridge) ? '/' : '=';
                // Only draw if grid cell is empty (kernel labels take priority below)
                if (grid[static_cast<std::size_t>(row)][static_cast<std::size_t>(col)] == ' ')
                    grid[static_cast<std::size_t>(row)][static_cast<std::size_t>(col)] = ch;
            }
        }

        // Place kernel labels (A, B, C, ...) at their roofline position
        std::vector<std::string> legend;
        char label = 'A';
        for (const auto& k : kernels) {
            int col = x_col(k.ai());
            int row = y_row(k.achieved_gflops);
            if (col >= 0 && col < W && row >= 0 && row < H) {
                grid[static_cast<std::size_t>(row)][static_cast<std::size_t>(col)] = label;
            }
            legend.push_back(std::string(1, label) + " = " + k.name);
            ++label;
        }

        // Print chart
        printf("\n  Roofline Chart  (X = log-scale AI, Y = GFLOPS)\n");
        printf("  %5.0f ┤", y_max);
        for (int col = 0; col < W; ++col) printf(" ");
        printf("← peak compute (%.0f GFLOPS)\n", peak_gflops);

        for (int row = 0; row < H; ++row) {
            double gf = y_max * (H - 1 - row) / (H - 1);
            if (row % 3 == 0)
                printf("  %5.0f │%s\n", gf, grid[static_cast<std::size_t>(row)].c_str());
            else
                printf("        │%s\n", grid[static_cast<std::size_t>(row)].c_str());
        }

        printf("      0 └");
        for (int col = 0; col < W; ++col) printf("─");
        printf("\n");

        // X axis tick labels (log scale)
        printf("         ");
        const double ticks[] = {0.05, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0};
        int prev_col = -8;
        for (double t : ticks) {
            int col = x_col(t);
            if (col < 0 || col >= W) continue;
            int gap = col - prev_col;
            if (gap > 0) {
                char buf[16];
                snprintf(buf, sizeof(buf), "%g", t);
                printf("%*s", gap, buf);
                prev_col = col + static_cast<int>(std::strlen(buf));
            }
        }
        printf("\n         FLOP/byte →\n");

        // Legend
        printf("\n  Legend (ridge = %.2f FLOP/byte):\n", ridge);
        for (const auto& s : legend)
            printf("    %s\n", s.c_str());
    }
};

} // namespace cpu_engine::roofline
