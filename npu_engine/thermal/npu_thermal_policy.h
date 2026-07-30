#pragma once
// npu_thermal_policy.h — NPU/SoC die temperature -> workload allocation
// policy. Mirrors fpga_engine/thermal_router/thermal_router.h's exact
// split (pure decision function vs. hardware-touching read) so the two
// thermal routers in this repo can never diverge in shape, only in the
// policy constants that reflect real hardware differences:
//
//   - NPUs are almost always integrated on the same die/package as the
//     CPU/GPU (Apple ANE, Qualcomm Hexagon) rather than a discrete card
//     with its own heatsink like an F1 FPGA -- much less thermal mass,
//     so the RC time constant (see npu_thermal_sim.cpp) is seconds, not
//     tens of seconds.
//   - Mobile/edge SoC thermal thresholds are lower than a datacenter
//     FPGA card's, since sustained high junction temperatures are more
//     aggressively avoided on unfanned/passively-cooled edge hardware.
//
// PLAN.md Phase 15 step 5: "mirrors fpga_engine/thermal_router's real
// thermal-response model; NPUs are power-efficiency-first hardware,
// on-theme for this pattern."

struct NpuThermalPolicy {
    float warning_temp_c  = 65.0f;  // log warning, no allocation change
    float throttle_temp_c = 75.0f;  // reduce NPU allocation by 50% (route to CPU/GPU)
    float shutdown_temp_c = 85.0f;  // route all work off the NPU entirely
};

class NpuThermalRouter {
public:
    explicit NpuThermalRouter(NpuThermalPolicy policy = {});

    // Pure decision function -- identical shape to
    // ThermalRouter::allocation_fraction_for_temp, no hardware dependency.
    float allocation_fraction_for_temp(float temp_c) const;

    // Read current NPU/SoC temperature. On Apple Silicon this would go
    // through IOKit's SMC (System Management Controller) sensor keys
    // (there is no public per-block ANE sensor -- SMC exposes overall
    // SoC/package temperature, not an ANE-specific reading, a real
    // platform limitation worth documenting rather than assuming a
    // per-block sensor exists the way FPGA's XADC does). Hardware-gated,
    // unrun -- see npu_thermal_router.cpp.
    float read_npu_temp_c() const;

    // read_npu_temp_c() + allocation_fraction_for_temp() in one call --
    // the actual per-poll router step. Hardware-gated.
    float npu_allocation_fraction() const;

private:
    NpuThermalPolicy policy_;
};
