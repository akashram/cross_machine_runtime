#include "inference/mlp.h"

#include <cassert>
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <new>
#include <vector>

using namespace cpu_engine::inference;

// ---------------------------------------------------------------------------
// Allocation counter — no-heap verification
//
// Override the four operator new/delete forms so we can count calls during
// forward(). The engine constructor is allowed to allocate; we only check
// allocations that happen *after* we flip g_tracking on.
// ---------------------------------------------------------------------------

static bool g_tracking         = false;
static int  g_allocs_tracked   = 0;

void* operator new(std::size_t n) {
    if (g_tracking) ++g_allocs_tracked;
    void* p = std::malloc(n);
    if (!p) throw std::bad_alloc{};
    return p;
}
void* operator new[](std::size_t n) {
    if (g_tracking) ++g_allocs_tracked;
    void* p = std::malloc(n);
    if (!p) throw std::bad_alloc{};
    return p;
}
void  operator delete(void* p)                 noexcept { std::free(p); }
void  operator delete[](void* p)               noexcept { std::free(p); }
void  operator delete(void* p, std::size_t)    noexcept { std::free(p); }
void  operator delete[](void* p, std::size_t)  noexcept { std::free(p); }

// ---------------------------------------------------------------------------
// Scalar reference MLP (no SIMD, no optimisation pragmas)
// ---------------------------------------------------------------------------

static float ref_sigmoid(float x) noexcept {
    return 0.5f * x / (1.0f + std::fabs(x)) + 0.5f;
}

static void ref_forward(const MlpConfig&                          cfg,
                         const std::vector<std::vector<float>>&   weights,
                         const std::vector<std::vector<float>>&   biases,
                         const float*                             input,
                         float*                                   output) {
    const int nl = cfg.n_layers();
    std::vector<float> buf(static_cast<std::size_t>(
        *std::max_element(cfg.dims.begin(), cfg.dims.end())));
    std::vector<float> cur(input, input + cfg.dims[0]);

    for (std::size_t l = 0; l < static_cast<std::size_t>(nl); ++l) {
        const int out_dim = cfg.dims[l + 1];
        const int in_dim  = cfg.dims[l];
        buf.assign(static_cast<std::size_t>(out_dim), 0.0f);

        for (int m = 0; m < out_dim; ++m) {
            float acc = biases[l][static_cast<std::size_t>(m)];
            for (int n = 0; n < in_dim; ++n)
                acc += weights[l][static_cast<std::size_t>(m * in_dim + n)]
                     * cur[static_cast<std::size_t>(n)];
            switch (cfg.acts[l]) {
                case Activation::kRelu:    buf[static_cast<std::size_t>(m)] = std::max(0.0f, acc); break;
                case Activation::kSigmoid: buf[static_cast<std::size_t>(m)] = ref_sigmoid(acc);    break;
                case Activation::kNone:    buf[static_cast<std::size_t>(m)] = acc;                 break;
            }
        }
        cur = buf;
    }
    std::copy(cur.begin(), cur.end(), output);
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static bool allclose(const float* a, const float* b, int n, float rtol = 5e-5f) {
    for (int i = 0; i < n; ++i) {
        float diff = std::fabs(a[i] - b[i]);
        if (diff > rtol * (std::fabs(a[i]) + 1.0f)) {
            printf("  mismatch at [%d]: ref=%.8f got=%.8f (diff=%.2e)\n",
                   i, static_cast<double>(a[i]), static_cast<double>(b[i]),
                   static_cast<double>(diff));
            return false;
        }
    }
    return true;
}

static std::vector<float> make_weights(int out_dim, int in_dim, int seed) {
    std::vector<float> w(static_cast<std::size_t>(out_dim * in_dim));
    for (std::size_t i = 0; i < w.size(); ++i)
        w[i] = static_cast<float>((static_cast<int>(i + static_cast<std::size_t>(seed)) % 11) - 5) * 0.1f;
    return w;
}

static std::vector<float> make_bias(int dim, int seed) {
    std::vector<float> b(static_cast<std::size_t>(dim));
    for (std::size_t i = 0; i < b.size(); ++i)
        b[i] = static_cast<float>((static_cast<int>(i + static_cast<std::size_t>(seed)) % 7) - 3) * 0.05f;
    return b;
}

static std::vector<float> make_input(int dim, int seed) {
    std::vector<float> x(static_cast<std::size_t>(dim));
    for (std::size_t i = 0; i < x.size(); ++i)
        x[i] = static_cast<float>((static_cast<int>(i + static_cast<std::size_t>(seed)) % 13) - 6) * 0.2f;
    return x;
}

// ---------------------------------------------------------------------------
// Test cases
// ---------------------------------------------------------------------------

static bool test_correctness(const char* name, const MlpConfig& cfg) {
    const int nl = cfg.n_layers();
    std::vector<std::vector<float>> weights(static_cast<std::size_t>(nl));
    std::vector<std::vector<float>> biases (static_cast<std::size_t>(nl));
    for (std::size_t l = 0; l < static_cast<std::size_t>(nl); ++l) {
        weights[l] = make_weights(cfg.dims[l + 1], cfg.dims[l], static_cast<int>(l) * 3 + 7);
        biases [l] = make_bias  (cfg.dims[l + 1], static_cast<int>(l) * 5 + 2);
    }

    auto input = make_input(cfg.dims[0], 42);
    std::vector<float> ref_out(static_cast<std::size_t>(cfg.dims.back()), 0);
    std::vector<float> got_out(static_cast<std::size_t>(cfg.dims.back()), 0);

    ref_forward(cfg, weights, biases, input.data(), ref_out.data());

    // Need separate copies for the engine (it moves them)
    auto w2 = weights;
    auto b2 = biases;
    MlpInferenceEngine engine{cfg, std::move(w2), std::move(b2)};
    engine.forward(input.data(), got_out.data());

    if (!allclose(ref_out.data(), got_out.data(), cfg.dims.back())) {
        printf("FAIL: %s\n", name);
        return false;
    }
    printf("PASS: %s\n", name);
    return true;
}

static bool test_no_heap_alloc(const char* name, const MlpConfig& cfg) {
    const int nl = cfg.n_layers();
    std::vector<std::vector<float>> weights(static_cast<std::size_t>(nl));
    std::vector<std::vector<float>> biases (static_cast<std::size_t>(nl));
    for (std::size_t l = 0; l < static_cast<std::size_t>(nl); ++l) {
        weights[l] = make_weights(cfg.dims[l + 1], cfg.dims[l], static_cast<int>(l));
        biases [l] = make_bias  (cfg.dims[l + 1], static_cast<int>(l));
    }

    MlpInferenceEngine engine{cfg, std::move(weights), std::move(biases)};

    auto input = make_input(cfg.dims[0], 7);
    std::vector<float> output(static_cast<std::size_t>(cfg.dims.back()));

    // Warm up so any lazy init inside stdlib is done before we start counting.
    engine.forward(input.data(), output.data());
    engine.forward(input.data(), output.data());

    g_allocs_tracked = 0;
    g_tracking       = true;
    for (int i = 0; i < 1000; ++i)
        engine.forward(input.data(), output.data());
    g_tracking = false;

    if (g_allocs_tracked != 0) {
        printf("FAIL no-heap: %s — %d allocation(s) during forward()\n",
               name, g_allocs_tracked);
        return false;
    }
    printf("PASS no-heap: %s (0 allocations over 1000 forward() calls)\n", name);
    return true;
}

int main() {
    bool ok = true;

    // --- correctness ---
    // Single layer, identity activation (pure linear)
    ok &= test_correctness("single-layer linear",
        {.dims={4, 4}, .acts={Activation::kNone}});

    // Single layer with relu
    ok &= test_correctness("single-layer relu",
        {.dims={8, 4}, .acts={Activation::kRelu}});

    // Two hidden layers, relu
    ok &= test_correctness("2-layer relu [16,32,16,4]",
        {.dims={16,32,16,4}, .acts={Activation::kRelu, Activation::kRelu, Activation::kNone}});

    // Mixed activations
    ok &= test_correctness("mixed [8,16,8,4] relu+sigmoid+none",
        {.dims={8,16,8,4},
         .acts={Activation::kRelu, Activation::kSigmoid, Activation::kNone}});

    // Representative router-style MLP: 64-in, two hidden 128, 32-out
    ok &= test_correctness("router MLP [64,128,128,32]",
        {.dims={64,128,128,32},
         .acts={Activation::kRelu, Activation::kRelu, Activation::kNone}});

    // Non-square, non-power-of-two dims
    ok &= test_correctness("asymmetric [13,37,11,5]",
        {.dims={13,37,11,5},
         .acts={Activation::kRelu, Activation::kSigmoid, Activation::kNone}});

    // Single-element output (binary classifier)
    ok &= test_correctness("binary classifier [32,64,1]",
        {.dims={32,64,1}, .acts={Activation::kRelu, Activation::kSigmoid}});

    // --- no-heap verification ---
    ok &= test_no_heap_alloc("router MLP [64,128,128,32]",
        {.dims={64,128,128,32},
         .acts={Activation::kRelu, Activation::kRelu, Activation::kNone}});

    ok &= test_no_heap_alloc("deep MLP [256,512,256,128,64]",
        {.dims={256,512,256,128,64},
         .acts={Activation::kRelu, Activation::kRelu, Activation::kRelu, Activation::kNone}});

    return ok ? 0 : 1;
}
