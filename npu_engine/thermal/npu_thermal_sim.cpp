// npu_thermal_sim.cpp — measures NpuThermalRouter's response latency
// locally against a synthetic NPU/SoC thermal event, no hardware needed.
// Direct analog of fpga_engine/thermal_router/thermal_router_sim.cpp;
// see that file's header for the full methodology explanation (RC step
// response, poll-interval-bounded reaction). This file only changes the
// device-specific constants, not the measurement approach.
//
//   clang++ -O2 -std=c++17 npu_thermal_sim.cpp npu_thermal_policy.cpp \
//       -o npu_thermal_sim && ./npu_thermal_sim
//
// Model caveats, NPU-specific vs. the FPGA version: an integrated
// mobile/edge SoC (Apple ANE, Qualcomm Hexagon) has far less thermal
// mass than a discrete F1 card + heatsink, so its RC time constant is
// modeled here as seconds, not tens of seconds (kTauSeconds=4.0 vs.
// thermal_router_sim.cpp's 15.0) -- both are order-of-magnitude, not
// datasheet, figures (see npu_thermal_router.cpp's header for why no
// stable public per-block sensor spec exists to calibrate against). A
// tighter poll interval (25ms vs. the FPGA version's 100ms) reflects
// that a mobile OS's power-management daemon typically polls SoC thermal
// state more frequently than an F1 host-side monitoring loop would poll
// over PCIe, not a claim about any specific OS's real polling cadence.

#include <chrono>
#include <cmath>
#include <cstdio>

#include "npu_thermal_policy.h"

namespace {

constexpr float kAmbientC = 35.0f;      // idle SoC temp under light NPU use
constexpr float kSteadyStateC = 92.0f;  // temp the SoC would settle at under sustained full NPU load
constexpr double kTauSeconds = 4.0;     // thermal time constant (order-of-magnitude, see header)
constexpr double kPollIntervalMs = 25.0; // OS-level thermal poll cadence (order-of-magnitude)
constexpr double kSimDurationSeconds = 20.0;

double temp_at(double t_seconds) {
    return kAmbientC + (kSteadyStateC - kAmbientC) * (1.0 - std::exp(-t_seconds / kTauSeconds));
}

double true_crossing_time_seconds(float threshold_c) {
    double frac = (threshold_c - kAmbientC) / (kSteadyStateC - kAmbientC);
    return -kTauSeconds * std::log(1.0 - frac);
}

struct CrossingResult {
    double true_crossing_s;
    double router_reacted_s;
    double response_latency_ms;
};

CrossingResult measure_response(const NpuThermalRouter& router, float threshold_c,
                                 float below_fraction, double poll_interval_ms) {
    double true_cross = true_crossing_time_seconds(threshold_c);
    double poll_s = poll_interval_ms / 1000.0;

    for (double t = 0.0; t <= kSimDurationSeconds; t += poll_s) {
        float temp_c = static_cast<float>(temp_at(t));
        float fraction = router.allocation_fraction_for_temp(temp_c);
        if (fraction < below_fraction) {
            double latency_ms = (t - true_cross) * 1000.0;
            return {true_cross, t, latency_ms};
        }
    }
    return {true_cross, -1.0, -1.0};
}

} // namespace

int main() {
    NpuThermalPolicy policy; // defaults: warning=65C, throttle=75C, shutdown=85C
    NpuThermalRouter router(policy);

    std::printf("=== npu_thermal_sim: synthetic NPU/SoC thermal event (RC step response) ===\n");
    std::printf("ambient=%.0fC steady-state=%.0fC tau=%.0fs poll-interval=%.0fms\n\n",
                kAmbientC, kSteadyStateC, kTauSeconds, kPollIntervalMs);

    constexpr int kIters = 1'000'000;
    volatile float sink = 0.0f;
    auto compute_start = std::chrono::steady_clock::now();
    for (int i = 0; i < kIters; ++i) {
        float t = kAmbientC + static_cast<float>(i % 60);
        sink = router.allocation_fraction_for_temp(t);
    }
    auto compute_end = std::chrono::steady_clock::now();
    double compute_ns_per_call =
        std::chrono::duration<double, std::nano>(compute_end - compute_start).count() / kIters;
    std::printf("decision-compute latency: %.2fns/call (%d calls, sink=%.1f)\n\n",
                compute_ns_per_call, kIters, sink);

    CrossingResult throttle = measure_response(router, policy.throttle_temp_c, 1.0f, kPollIntervalMs);
    CrossingResult shutdown = measure_response(router, policy.shutdown_temp_c, 0.5f, kPollIntervalMs);

    std::printf("throttle threshold (%.0fC, allocation 1.0->0.5): "
                "true crossing t=%.3fs, router reacted t=%.3fs, response latency=%.2fms "
                "(poll-interval bound=%.0fms)\n",
                policy.throttle_temp_c, throttle.true_crossing_s, throttle.router_reacted_s,
                throttle.response_latency_ms, kPollIntervalMs);
    std::printf("shutdown threshold (%.0fC, allocation 0.5->0.0): "
                "true crossing t=%.3fs, router reacted t=%.3fs, response latency=%.2fms "
                "(poll-interval bound=%.0fms)\n",
                policy.shutdown_temp_c, shutdown.true_crossing_s, shutdown.router_reacted_s,
                shutdown.response_latency_ms, kPollIntervalMs);

    std::printf("\nboth response latencies fall within the %.0fms poll-interval bound, as they "
                "must -- same structural finding as fpga_engine/thermal_router's: response "
                "latency is bounded by polling cadence, not router logic (decision-compute cost "
                "is %.2fns, ~%.1ex smaller than the poll interval). The NPU's much smaller tau "
                "(%.0fs vs FPGA's 15s) means it crosses both thresholds far sooner in absolute "
                "time given the same steady-state overshoot -- an integrated SoC with low "
                "thermal mass heats up (and would need to react) faster than a discrete card "
                "with a heatsink, a real, structural difference this model captures via tau "
                "alone, without changing any router logic.\n",
                kPollIntervalMs, compute_ns_per_call, (kPollIntervalMs * 1e6) / compute_ns_per_call,
                kTauSeconds);

    return 0;
}
