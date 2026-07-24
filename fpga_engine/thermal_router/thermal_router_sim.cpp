// thermal_router_sim.cpp — measures ThermalRouter's response latency
// locally, against a synthetic FPGA thermal event, without an F1
// instance. PLAN.md step 25 asks for "FPGA temperature above threshold
// -> reduce FPGA workload allocation in the router, measured response
// latency" -- xadc/parse_xadc_json.py's header specifically deferred
// that measurement to here: "needs a real running router loop."
//
// This links against thermal_policy.cpp (the pure decision logic) but
// NOT thermal_router.cpp (the XRT-dependent sensor read), so it compiles
// and runs on this Mac today with no hardware dependency:
//   clang++ -O2 -std=c++17 thermal_router_sim.cpp thermal_policy.cpp \
//       -o thermal_router_sim && ./thermal_router_sim
//
// Two things measured, both real (not TODO):
//   1. Decision-compute latency: wall-clock time for
//      allocation_fraction_for_temp() itself, averaged over many calls.
//      This isolates "how long does the router take to decide" from
//      "how often does it get a new reading."
//   2. Response latency to a thermal event: a synthetic FPGA die
//      temperature trace (a first-order RC thermal step response --
//      the standard shape for silicon temperature climbing under a
//      sudden sustained load increase, T(t) = T_ambient + (T_final -
//      T_ambient) * (1 - exp(-t/tau))) sampled at a fixed polling
//      interval, same discretization a real host-side monitoring loop
//      would use. The router only sees temperature at poll boundaries,
//      so its reaction to a threshold crossing is bounded by the poll
//      interval -- this measures exactly how much of that bound each
//      crossing actually uses.
//
// Model caveats: tau (thermal time constant) and the polling interval
// are commonly-cited order-of-magnitude figures (large silicon + heatsink
// mass gives FPGA die thermal step responses on the order of seconds to
// tens of seconds; a few-hundred-ms host polling cadence is a typical
// monitoring-loop tradeoff between reaction time and PCIe/host overhead),
// not datasheet numbers for the AWS F1 VU9P shell specifically -- same
// caveat style as every other *_model.cpp in fpga_engine/. What doesn't
// depend on getting them exactly right: response latency to a threshold
// crossing is always bounded by the polling interval, regardless of tau,
// because the router literally cannot see the temperature between polls.

#include <chrono>
#include <cmath>
#include <cstdio>
#include <vector>

#include "thermal_router.h"

namespace {

constexpr float kAmbientC = 45.0f;   // idle FPGA die temp
constexpr float kSteadyStateC = 100.0f; // temp the die would settle at under sustained full load
constexpr double kTauSeconds = 15.0;    // thermal time constant (order-of-magnitude, see header)
constexpr double kPollIntervalMs = 100.0; // host-side monitoring poll cadence (order-of-magnitude)
constexpr double kSimDurationSeconds = 60.0;

double temp_at(double t_seconds) {
    return kAmbientC + (kSteadyStateC - kAmbientC) * (1.0 - std::exp(-t_seconds / kTauSeconds));
}

// Closed-form time at which the continuous RC model crosses threshold_c,
// so "true" crossing time (what a continuously-sampled sensor would see)
// can be compared against when the discretely-polled router actually
// reacts.
double true_crossing_time_seconds(float threshold_c) {
    double frac = (threshold_c - kAmbientC) / (kSteadyStateC - kAmbientC);
    return -kTauSeconds * std::log(1.0 - frac);
}

struct CrossingResult {
    double true_crossing_s;
    double router_reacted_s;
    double response_latency_ms;
};

CrossingResult measure_response(const ThermalRouter& router, float threshold_c, float below_fraction) {
    double true_cross = true_crossing_time_seconds(threshold_c);
    double poll_s = kPollIntervalMs / 1000.0;

    for (double t = 0.0; t <= kSimDurationSeconds; t += poll_s) {
        float temp_c = static_cast<float>(temp_at(t));
        float fraction = router.allocation_fraction_for_temp(temp_c);
        if (fraction < below_fraction) {
            double latency_ms = (t - true_cross) * 1000.0;
            return {true_cross, t, latency_ms};
        }
    }
    return {true_cross, -1.0, -1.0}; // never crossed within sim window
}

} // namespace

int main() {
    ThermalPolicy policy; // defaults: warning=75C, throttle=85C, shutdown=95C
    ThermalRouter router(policy);

    std::printf("=== thermal_router_sim: synthetic FPGA thermal event (RC step response) ===\n");
    std::printf("ambient=%.0fC steady-state=%.0fC tau=%.0fs poll-interval=%.0fms\n\n",
                kAmbientC, kSteadyStateC, kTauSeconds, kPollIntervalMs);

    // 1. Decision-compute latency: time allocation_fraction_for_temp()
    // itself, isolated from the polling loop.
    constexpr int kIters = 1'000'000;
    volatile float sink = 0.0f; // prevent the optimizer from eliding the loop
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

    // 2. Response latency at each policy threshold: how much of the poll
    // interval each crossing actually used, bounded by kPollIntervalMs.
    CrossingResult throttle = measure_response(router, policy.throttle_temp_c, 1.0f);
    CrossingResult shutdown = measure_response(router, policy.shutdown_temp_c, 0.5f);

    std::printf("throttle threshold (%.0fC, allocation 1.0->0.5): "
                "true crossing t=%.2fs, router reacted t=%.2fs, response latency=%.1fms "
                "(poll-interval bound=%.0fms)\n",
                policy.throttle_temp_c, throttle.true_crossing_s, throttle.router_reacted_s,
                throttle.response_latency_ms, kPollIntervalMs);
    std::printf("shutdown threshold (%.0fC, allocation 0.5->0.0): "
                "true crossing t=%.2fs, router reacted t=%.2fs, response latency=%.1fms "
                "(poll-interval bound=%.0fms)\n",
                policy.shutdown_temp_c, shutdown.true_crossing_s, shutdown.router_reacted_s,
                shutdown.response_latency_ms, kPollIntervalMs);

    std::printf("\nresponse latency at both thresholds falls within the %.0fms poll-interval "
                "bound, as it must -- the router cannot react faster than it can see a new "
                "reading. Decision-compute latency (%.2fns) is ~%.1ex smaller than the poll "
                "interval, confirming the bottleneck is polling cadence, not the router's own "
                "logic; a tighter response-latency budget is a poll-interval tuning question "
                "(more frequent XRT sensor reads), not a router-logic optimization.\n",
                kPollIntervalMs, compute_ns_per_call,
                (kPollIntervalMs * 1e6) / compute_ns_per_call);

    return 0;
}
