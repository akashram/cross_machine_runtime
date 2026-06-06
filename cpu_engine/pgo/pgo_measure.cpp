// PGO measurement binary — identical in both the instrument and pgo-use builds.
// Run this binary from both builds and compare the GFLOPS numbers to measure
// the PGO speedup.
//
// Kernels chosen: those most likely to benefit from PGO —
//   MLP forward: activation-switch branch, inlining of forward()
//   matmul_tiled: boundary branch (cold path after PGO), loop trip counts
//   dot_f32:      minimal branching; control to show PGO has no downside

#include "avx512/kernels.h"
#include "tiling/matmul.h"
#include "inference/mlp.h"
#include "affinity/affinity.h"
#include "foundation/bench/bench.h"

#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

using namespace cpu_engine;
using bench::tsc_now;
using bench::tsc_to_ns;

static constexpr int kWarmup = 20;
static constexpr int kPasses = 500;

struct R { double ns; double gflops; };

template<typename Fn>
static R measure(Fn fn, double flops) {
    for (int i = 0; i < kWarmup; ++i) fn();
    uint64_t t0 = tsc_now();
    for (int i = 0; i < kPasses; ++i) fn();
    uint64_t t1 = tsc_now();
    double ns = tsc_to_ns(t1 - t0) / kPasses;
    return {ns, flops / (ns * 1e-9) / 1e9};
}

static inference::MlpInferenceEngine make_engine(const inference::MlpConfig& cfg) {
    const auto nl = static_cast<std::size_t>(cfg.n_layers());
    std::vector<std::vector<float>> w(nl), b(nl);
    std::mt19937 rng{99};
    std::uniform_real_distribution<float> d{-0.1f, 0.1f};
    for (std::size_t l = 0; l < nl; ++l) {
        w[l].resize(static_cast<std::size_t>(cfg.dims[l] * cfg.dims[l+1]));
        b[l].resize(static_cast<std::size_t>(cfg.dims[l+1]));
        for (auto& v : w[l]) v = d(rng);
        for (auto& v : b[l]) v = d(rng);
    }
    return {cfg, std::move(w), std::move(b)};
}

int main() {
    ThreadPinner::pin(0);
    (void)bench::tsc_ticks_per_ns();

    printf("%-40s  %9s  %8s\n", "kernel", "ns/call", "GFLOPS");
    printf("%s\n", std::string(61, '-').c_str());

    // MLP tiny — kRelu dominant, same as training distribution
    {
        inference::MlpConfig cfg{
            .dims={64,128,128,64,32},
            .acts={inference::Activation::kRelu, inference::Activation::kRelu,
                   inference::Activation::kRelu, inference::Activation::kNone}};
        auto engine = make_engine(cfg);
        std::vector<float> in(64, 0.5f), out(32);
        volatile float sink = 0;
        double flops = 2.0*(64*128 + 128*128 + 128*64 + 64*32);
        auto r = measure([&]{ engine.forward(in.data(),out.data()); sink=out[0]; }, flops);
        (void)sink;
        printf("%-40s  %9.0f  %8.2f\n", "mlp 64→128→128→64→32 (kRelu)", r.ns, r.gflops);
    }

    // MLP with sigmoid output — tests the cold branch
    {
        inference::MlpConfig cfg{
            .dims={32,64,32,1},
            .acts={inference::Activation::kRelu,
                   inference::Activation::kRelu,
                   inference::Activation::kSigmoid}};
        auto engine = make_engine(cfg);
        std::vector<float> in(32, 0.5f), out(1);
        volatile float sink = 0;
        double flops = 2.0*(32*64 + 64*32 + 32*1);
        auto r = measure([&]{ engine.forward(in.data(),out.data()); sink=out[0]; }, flops);
        (void)sink;
        printf("%-40s  %9.0f  %8.2f\n", "mlp 32→64→32→1 (kSigmoid out)", r.ns, r.gflops);
    }

    // Matmul tiled — tile-boundary branch should be predicted cold after PGO
    {
        constexpr int M = 256;
        std::vector<float> A(M*M,0.5f), B(M*M,0.3f), C(M*M,0.0f);
        volatile float sink = 0;
        double flops = 2.0*M*M*M;
        auto r = measure([&]{
            std::memset(C.data(), 0, static_cast<std::size_t>(M*M)*sizeof(float));
            tiling::matmul_tiled_f32(A.data(), B.data(), C.data(), M, M, M, 64);
            sink = C[0];
        }, flops);
        (void)sink;
        printf("%-40s  %9.0f  %8.2f\n", "matmul_tiled T=64 256×256", r.ns, r.gflops);
    }

    // dot_f32 — minimal branching; PGO should be neutral here
    {
        constexpr int N = 16384;
        std::vector<float> a(N,1.0f), b(N,2.0f);
        volatile float sink = 0;
        auto r = measure([&]{ sink = avx512::dot_f32(a.data(), b.data(), N); }, 2.0*N);
        (void)sink;
        printf("%-40s  %9.0f  %8.2f\n", "dot_f32 16K (L1)", r.ns, r.gflops);
    }

    ThreadPinner::unpin();
}
