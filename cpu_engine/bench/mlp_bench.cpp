#include "inference/mlp.h"
#include "affinity/affinity.h"
#include "foundation/bench/bench.h"
#include "foundation/perf/perf.h"

#include <cstdio>
#include <random>
#include <vector>

using namespace cpu_engine::inference;
using cpu_engine::ThreadPinner;
using foundation::PerfCounters;

// ---------------------------------------------------------------------------
// MLP inference throughput benchmark
//
// Three network sizes representing realistic use cases:
//
//   Tiny  [64 → 128 → 64 → 32]     — per-request routing decision
//   Small [256 → 512 → 256 → 128]  — lightweight feature head
//   Large [1024 → 2048 → 1024 → 512 → 128] — heavier embedding projection
//
// Each layer uses ReLU except the last (identity).  Weights are random
// so the compiler cannot constant-fold the computation away.
//
// Metrics:
//   ns/call  — end-to-end forward pass latency
//   GFLOPS   — FLOPs = 2 * Σ_l(in_dim * out_dim) (one matvec per layer)
//   infer/s  — throughput at batch size 1
//   IPC      — instruction-level parallelism (Linux only via PerfCounters)
// ---------------------------------------------------------------------------

static constexpr int kWarmup = 20;
static constexpr int kPasses = 200;

struct BenchResult {
    double ns;
    double gflops;
    double infer_per_sec;
    double ipc;
};

template <typename Fn>
static BenchResult measure(Fn fn, double flops) {
    PerfCounters ctr;
    for (int i = 0; i < kWarmup; ++i) fn();
    ctr.start();
    uint64_t t0 = bench::tsc_now();
    for (int i = 0; i < kPasses; ++i) fn();
    uint64_t t1 = bench::tsc_now();
    auto snap = ctr.stop();
    double ns = bench::tsc_to_ns(t1 - t0) / kPasses;
    return {ns, flops / (ns * 1e-9) / 1e9, 1e9 / ns, snap.ipc()};
}

static double network_flops(const MlpConfig& cfg) {
    double total = 0.0;
    for (std::size_t l = 0; l < static_cast<std::size_t>(cfg.n_layers()); ++l)
        total += 2.0 * cfg.dims[l] * cfg.dims[l + 1];
    return total;
}

static MlpInferenceEngine build_engine(const MlpConfig& cfg, std::mt19937& rng) {
    std::uniform_real_distribution<float> dist{-0.1f, 0.1f};
    const auto nl = static_cast<std::size_t>(cfg.n_layers());
    std::vector<std::vector<float>> weights(nl);
    std::vector<std::vector<float>> biases (nl);
    for (std::size_t l = 0; l < nl; ++l) {
        auto sz_w = static_cast<std::size_t>(cfg.dims[l] * cfg.dims[l + 1]);
        auto sz_b = static_cast<std::size_t>(cfg.dims[l + 1]);
        weights[l].resize(sz_w);
        biases [l].resize(sz_b);
        for (auto& v : weights[l]) v = dist(rng);
        for (auto& v : biases [l]) v = dist(rng);
    }
    return {cfg, std::move(weights), std::move(biases)};
}

static void bench_net(const char* label, const MlpConfig& cfg) {
    std::mt19937 rng{42};
    auto engine = build_engine(cfg, rng);

    // Random input — different each call to defeat optimistic branch prediction
    std::uniform_real_distribution<float> dist{-1.0f, 1.0f};
    std::vector<float> input (static_cast<std::size_t>(engine.input_dim()));
    std::vector<float> output(static_cast<std::size_t>(engine.output_dim()));
    for (auto& v : input) v = dist(rng);

    volatile float sink = 0.0f; // prevent dead-code elimination of output

    auto r = measure([&] {
        engine.forward(input.data(), output.data());
        sink = output[0];
    }, network_flops(cfg));
    (void)sink;

    printf("\n%s\n", label);
    printf("  dims:       ");
    for (int i = 0; i < static_cast<int>(cfg.dims.size()); ++i)
        printf("%d%s", cfg.dims[static_cast<std::size_t>(i)],
               i + 1 < static_cast<int>(cfg.dims.size()) ? " → " : "\n");
    printf("  FLOPs/call: %.3g\n", network_flops(cfg));
    printf("  latency:    %.0f ns\n", r.ns);
    printf("  throughput: %.0f inferences/sec\n", r.infer_per_sec);
    printf("  GFLOPS:     %.2f\n", r.gflops);
    if (r.ipc > 0.0)
        printf("  IPC:        %.2f\n", r.ipc);
    else
        printf("  IPC:        n/a (macOS)\n");
}

int main() {
    ThreadPinner::pin(0);
    (void)bench::tsc_ticks_per_ns();

    printf("MLP Inference Engine Benchmark\n");
    printf("================================\n");

    if (PerfCounters{}.available())
        printf("PerfCounters: available — IPC shown\n");
    else
        printf("PerfCounters: not available on macOS — timing only\n");

#if defined(__AVX512F__)
    printf("ISA: AVX-512 (matvec uses AVX-512 FMA path)\n");
#elif defined(__AVX2__)
    printf("ISA: AVX2 (matvec uses auto-vectorised path)\n");
#else
    printf("ISA: scalar\n");
#endif

    // Tiny: per-request routing — fits in L1
    bench_net("Tiny network — routing/dispatch head",
        {.dims={64, 128, 64, 32},
         .acts={Activation::kRelu, Activation::kRelu, Activation::kNone}});

    // Small: lightweight feature head — fits in L2
    bench_net("Small network — lightweight feature head",
        {.dims={256, 512, 256, 128},
         .acts={Activation::kRelu, Activation::kRelu, Activation::kNone}});

    // Large: heavier embedding projection — spills to L3
    bench_net("Large network — embedding projection",
        {.dims={1024, 2048, 1024, 512, 128},
         .acts={Activation::kRelu, Activation::kRelu, Activation::kRelu, Activation::kNone}});

    ThreadPinner::unpin();
}
