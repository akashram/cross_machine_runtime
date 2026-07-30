// npu_thermal_policy.cpp — NpuThermalRouter's pure decision logic. No
// hardware dependency: this is the half npu_thermal_sim.cpp links
// against to actually run and measure step 5's router locally.
// npu_thermal_router.cpp (hardware-gated, unrun) implements the other
// two (hardware-touching) methods in a separate translation unit, same
// separation as fpga_engine/thermal_router.

#include "npu_thermal_policy.h"

NpuThermalRouter::NpuThermalRouter(NpuThermalPolicy policy) : policy_(policy) {}

float NpuThermalRouter::allocation_fraction_for_temp(float temp_c) const {
    if (temp_c >= policy_.shutdown_temp_c) {
        return 0.0f;
    }
    if (temp_c >= policy_.throttle_temp_c) {
        return 0.5f;
    }
    return 1.0f;
}
