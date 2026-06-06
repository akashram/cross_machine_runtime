// PGO training workload — exercises all hot paths with realistic distributions.
//
// Compiled with -fprofile-instr-generate, this binary emits a .profraw file
// that records basic-block execution counts.  Run it, then merge the profile
// with llvm-profdata, and recompile the project with -fprofile-instr-use.
//
// What the profiler learns from this workload:
//   - MLP activation-switch: kRelu is the dominant branch (~70% of calls),
//     kSigmoid appears ~20%, kNone ~10%. Helps the compiler order cases and
//     lay out the hot path first.
//   - Matmul tile-boundary branch: the std::min(i0+T, M) guard is almost
//     never taken (M is a multiple of T here). PGO marks it as cold.
//   - Inline decisions: dot_f32, matvec_f32, eltwise calls get hot counts,
//     encouraging more aggressive inlining of their callers.
//   - Loop trip counts: PGO records how many times each loop iterates,
//     enabling better unrolling / software pipelining decisions.

#include "avx512/kernels.h"
#include "tiling/matmul.h"
#include "inference/mlp.h"
#include "affinity/affinity.h"

#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

using namespace cpu_engine;

static inference::MlpInferenceEngine make_engine(const inference::MlpConfig& cfg) {
    const auto nl = static_cast<std::size_t>(cfg.n_layers());
    std::vector<std::vector<float>> w(nl), b(nl);
    std::mt19937 rng{42};
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

    // --- MLP forward (dominant path: kRelu, ~10 000 calls per network) -----
    // Gives the profiler a strong signal on the activation switch.

    // kRelu-dominant (most realistic: all hidden layers use ReLU)
    {
        inference::MlpConfig cfg{
            .dims={64,128,128,64,32},
            .acts={inference::Activation::kRelu, inference::Activation::kRelu,
                   inference::Activation::kRelu, inference::Activation::kNone}};
        auto engine = make_engine(cfg);
        std::vector<float> in(64, 0.3f), out(32);
        volatile float sink = 0;
        for (int i = 0; i < 20000; ++i) {
            in[static_cast<std::size_t>(i % 64)] = static_cast<float>(i) * 0.001f;
            engine.forward(in.data(), out.data());
            sink = out[0];
        }
        (void)sink;
    }

    // kSigmoid in final layer (binary classifiers)
    {
        inference::MlpConfig cfg{
            .dims={32,64,32,1},
            .acts={inference::Activation::kRelu,
                   inference::Activation::kRelu,
                   inference::Activation::kSigmoid}};
        auto engine = make_engine(cfg);
        std::vector<float> in(32, 0.5f), out(1);
        volatile float sink = 0;
        for (int i = 0; i < 5000; ++i) {
            engine.forward(in.data(), out.data());
            sink = out[0];
        }
        (void)sink;
    }

    // --- Matmul (square, tile-multiple sizes → boundary branch stays cold) --
    {
        constexpr int M = 256;
        std::vector<float> A(M*M, 0.5f), B(M*M, 0.3f), C(M*M);
        volatile float sink = 0;
        for (int i = 0; i < 500; ++i) {
            std::memset(C.data(), 0, static_cast<std::size_t>(M*M)*sizeof(float));
            tiling::matmul_tiled_f32(A.data(), B.data(), C.data(), M, M, M, 64);
            sink = C[0];
        }
        (void)sink;
    }

    // --- dot / matvec / eltwise (streaming kernels) -------------------------
    {
        constexpr int N = 16384;
        std::vector<float> a(N, 1.0f), b(N, 2.0f), c(N, 0.0f);
        volatile float sink = 0;
        for (int i = 0; i < 5000; ++i) {
            sink = avx512::dot_f32(a.data(), b.data(), N);
            avx512::eltwise_relu_f32(a.data(), c.data(), N);
            avx512::eltwise_add_f32_autovec(a.data(), b.data(), c.data(), N);
        }
        (void)sink;
    }

    {
        constexpr int M = 256, Nv = 256;
        std::vector<float> A(M*Nv, 0.5f), x(Nv, 1.0f), y(M, 0.0f);
        volatile float sink = 0;
        for (int i = 0; i < 5000; ++i) {
            avx512::matvec_f32(A.data(), x.data(), y.data(), M, Nv);
            sink = y[0];
        }
        (void)sink;
    }

    ThreadPinner::unpin();
    printf("Training complete. Profile data written to $LLVM_PROFILE_FILE "
           "(or default.profraw).\n");
}
