// thermal_policy.cpp — ThermalRouter's pure temperature -> allocation-
// fraction decision logic (thermal_router.h). No hardware dependency:
// this is the half thermal_router_sim.cpp links against to actually run
// and measure step 25's router locally. thermal_router.cpp implements
// the other two (hardware-touching) methods in a separate translation
// unit specifically so this one never needs XRT headers to compile.

#include "thermal_router.h"

ThermalRouter::ThermalRouter(ThermalPolicy policy) : policy_(policy) {}

float ThermalRouter::allocation_fraction_for_temp(float temp_c) const {
    if (temp_c >= policy_.shutdown_temp_c) {
        return 0.0f;
    }
    if (temp_c >= policy_.throttle_temp_c) {
        return 0.5f;
    }
    // Between warning_temp_c and throttle_temp_c the router still logs a
    // warning (fpga_allocation_fraction()'s job, since that's the method
    // that actually has a temperature reading to log) but doesn't change
    // allocation on its own -- warning_temp_c is an early-signal
    // threshold, not an allocation threshold.
    return 1.0f;
}
