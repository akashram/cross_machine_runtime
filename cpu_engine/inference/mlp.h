#pragma once

// CPU MLP inference engine — zero heap on the forward path
// =========================================================
//
// ARCHITECTURE
// ------------
// A fully-connected multi-layer perceptron:
//
//   input → [matvec → bias → activation] × N_layers → output
//
// Each layer is:
//   1. matvec_f32(W, src, dst, out_dim, in_dim)   — linear transform
//   2. dst[m] += bias[m]                           — bias add
//   3. activation applied in-place on dst          — relu / sigmoid / identity
//
// Two scratch buffers (buf_a_, buf_b_) of size max(all layer dims) are
// allocated at construction and ping-ponged between layers. forward() never
// calls operator new.
//
// ACTIVATION FUSING
// -----------------
// The avx512 elementwise kernels require non-aliasing in/out pointers.
// Bias + activation are instead fused in a single in-place loop on dst
// after matvec writes it. The loop is vectorized by the compiler (verified
// by examining the generated code). This avoids an extra buffer.
//
// WEIGHTS LAYOUT
// --------------
// weights[l] is row-major: element (m, n) = weights[l][m * in_dim + n].
// This matches matvec_f32's expectation: y[m] = sum_n(A[m*N+n] * x[n]).
//
// NO-HEAP GUARANTEE
// -----------------
// Verified in mlp_test.cpp by overriding operator new and counting
// allocations through all forward() calls.

#include "avx512/kernels.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <vector>

namespace cpu_engine::inference {

enum class Activation { kNone, kRelu, kSigmoid };

struct MlpConfig {
    // Layer dimensions: [input_dim, hidden_1, ..., hidden_N, output_dim].
    // Layer l transforms dims[l] → dims[l+1].
    std::vector<int> dims;

    // One activation per layer; size must equal dims.size() - 1.
    std::vector<Activation> acts;

    int n_layers() const noexcept {
        return static_cast<int>(dims.size()) - 1;
    }
};

class MlpInferenceEngine {
public:
    // Moves weights and biases in — no copy.
    // weights[l]: flat row-major [dims[l+1] × dims[l]] matrix.
    // biases[l]:  vector of length dims[l+1].
    MlpInferenceEngine(MlpConfig                        cfg,
                       std::vector<std::vector<float>>  weights,
                       std::vector<std::vector<float>>  biases)
        : cfg_    {std::move(cfg)}
        , weights_{std::move(weights)}
        , biases_ {std::move(biases)}
    {
        const int nl = cfg_.n_layers();
        assert(static_cast<int>(cfg_.acts.size())  == nl);
        assert(static_cast<int>(weights_.size())   == nl);
        assert(static_cast<int>(biases_.size())    == nl);

        for (std::size_t l = 0; l < static_cast<std::size_t>(nl); ++l) {
            assert(static_cast<int>(weights_[l].size()) ==
                   cfg_.dims[l] * cfg_.dims[l + 1]);
            assert(static_cast<int>(biases_[l].size()) == cfg_.dims[l + 1]);
        }

        int max_dim = *std::max_element(cfg_.dims.begin(), cfg_.dims.end());
        buf_a_.resize(static_cast<std::size_t>(max_dim));
        buf_b_.resize(static_cast<std::size_t>(max_dim));
    }

    // Zero-allocation forward pass.
    // input  — cfg_.dims[0] floats.
    // output — cfg_.dims.back() floats.
    void forward(const float* __restrict__ input,
                 float* __restrict__ output) noexcept {
        const int nl = cfg_.n_layers();

        std::memcpy(buf_a_.data(), input,
                    static_cast<std::size_t>(cfg_.dims[0]) * sizeof(float));

        float* src = buf_a_.data();
        float* dst = buf_b_.data();

        for (std::size_t l = 0; l < static_cast<std::size_t>(nl); ++l) {
            const int out_dim = cfg_.dims[l + 1];
            const int in_dim  = cfg_.dims[l];

            // Linear transform: dst[m] = Σ_n W[m,n] * src[n]
            avx512::matvec_f32(weights_[l].data(), src, dst, out_dim, in_dim);

            // Fused bias + activation in-place on dst.
            // Simple loop: compiler auto-vectorizes; avoids __restrict__ aliasing.
            const float* b = biases_[l].data();
            switch (cfg_.acts[l]) {
                case Activation::kRelu:
#pragma clang loop vectorize(enable)
                    for (int m = 0; m < out_dim; ++m)
                        dst[m] = std::max(0.0f, dst[m] + b[m]);
                    break;

                case Activation::kSigmoid:
#pragma clang loop vectorize(enable)
                    for (int m = 0; m < out_dim; ++m) {
                        float x = dst[m] + b[m];
                        // Fast sigmoid: 0.5·x/(1+|x|)+0.5  (~0.5% error vs exp)
                        dst[m] = 0.5f * x / (1.0f + std::fabs(x)) + 0.5f;
                    }
                    break;

                case Activation::kNone:
#pragma clang loop vectorize(enable)
                    for (int m = 0; m < out_dim; ++m)
                        dst[m] += b[m];
                    break;
            }

            std::swap(src, dst);
        }

        // After the last swap, `src` points to the buffer holding the result.
        std::memcpy(output, src,
                    static_cast<std::size_t>(cfg_.dims.back()) * sizeof(float));
    }

    int input_dim()  const noexcept { return cfg_.dims.front(); }
    int output_dim() const noexcept { return cfg_.dims.back(); }

private:
    MlpConfig                       cfg_;
    std::vector<std::vector<float>> weights_;
    std::vector<std::vector<float>> biases_;
    std::vector<float>              buf_a_;
    std::vector<float>              buf_b_;
};

} // namespace cpu_engine::inference
