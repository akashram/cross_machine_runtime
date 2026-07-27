// layout_opt_model.cpp — MXU tile-padding overhead model.
//
// PLAN.md Phase 8 step 5: "tile padding for systolic array alignment,
// measure MXU utilization before/after." There's no TPU here, so this
// file does the part that doesn't need one: the pure geometry of how much
// wasted compute a given (M, N, K) matmul incurs once every dimension is
// padded up to the MXU's 128x128 systolic array tile size, before any
// hardware measurement enters the picture.
//
// The TPU v4 MXU consumes a 128x128 tile per cycle-group; a matmul whose
// M/N/K aren't multiples of 128 still occupies whole tiles at the edges,
// so the compiler pads up rather than leaving the array partially idle
// per-element. Padded FLOPs are real cycles spent multiplying by zero —
// this model's "utilization ceiling" is the fraction of that padded work
// that was ever going to matter, an upper bound on achievable MXU
// utilization independent of anything the scheduler or compiler does
// afterward (mxu_util_bench.py's real profiler numbers, once run, should
// sit at or below this ceiling, never above it).

#include <cstdio>
#include <cstdint>
#include <vector>
#include <string>

namespace {

constexpr int64_t kTileSize = 128;

int64_t pad_up(int64_t dim) {
    return ((dim + kTileSize - 1) / kTileSize) * kTileSize;
}

struct MatmulShape {
    std::string label;
    int64_t m, n, k;
};

struct PaddingResult {
    int64_t padded_m, padded_n, padded_k;
    double useful_flops, padded_flops, utilization_ceiling_pct;
};

PaddingResult analyze(const MatmulShape &shape) {
    PaddingResult r;
    r.padded_m = pad_up(shape.m);
    r.padded_n = pad_up(shape.n);
    r.padded_k = pad_up(shape.k);
    r.useful_flops = 2.0 * shape.m * shape.n * shape.k;
    r.padded_flops = 2.0 * r.padded_m * r.padded_n * r.padded_k;
    r.utilization_ceiling_pct = 100.0 * r.useful_flops / r.padded_flops;
    return r;
}

} // namespace

int main() {
    // A mix of aligned (multiples of 128) and misaligned shapes drawn from
    // common transformer dimensions: attention projections, MLP up/down
    // projections at a few model scales, and a deliberately awkward
    // batch/head-dim combination to show the worst case.
    const std::vector<MatmulShape> shapes = {
        {"attn qkv proj, d_model=512 (aligned)",     512, 1536, 512},
        {"lm_head proj, vocab=50257 (misaligned)",   4096, 50257, 768},
        {"mlp up-proj, d=4096 (aligned)",            4096, 16384, 4096},
        {"mlp up-proj, d=4097 (off-by-one)",         4097, 16384, 4097},
        {"small batch=17 x d=768",                   17, 768, 768},
        {"batch=1 decode step, d=4096",               1, 4096, 4096},
    };

    std::printf("%-40s %8s %8s %8s %14s\n", "shape", "M", "N", "K", "util ceiling");
    for (const auto &shape : shapes) {
        PaddingResult r = analyze(shape);
        std::printf("%-40s %8lld %8lld %8lld %13.1f%%\n",
                    shape.label.c_str(),
                    (long long)shape.m, (long long)shape.n, (long long)shape.k,
                    r.utilization_ceiling_pct);
    }

    std::printf(
        "\nExpected cliff: utilization ceiling drops sharply once any "
        "dimension is 1 past a multiple of 128 (worst at dim=129, where "
        "padding to 256 wastes ~50%% of that axis) and drops further at "
        "very small batch (batch=1 decode: M pads 1 -> 128, a 128x "
        "ceiling loss on that axis alone) -- exactly the padding waste "
        "step 9 (mxu_opt) and PLAN.md's stated performance cliff refer to.\n");
    return 0;
}
