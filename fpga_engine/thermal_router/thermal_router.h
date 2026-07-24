#pragma once
// thermal_router.h — FPGA die temperature -> workload allocation policy.
//
// PLAN.md step 25: "FPGA temperature above threshold -> reduce FPGA
// workload allocation in the router, measured response latency."
//
// Split into a pure decision function and a hardware-touching read, same
// separation xadc/ already anticipated (see xadc/parse_xadc_json.py's
// header: "Decide a workload-allocation fraction from the temperature.
// That's thermal_router's ThermalPolicy" / "Measure response latency...
// needs a real running router loop"):
//   - allocation_fraction_for_temp() is plain arithmetic on a float. No
//     XRT, no F1, no sensor. thermal_router_sim.cpp drives it with a
//     synthetic thermal trace and measures response latency locally.
//   - read_fpga_temp_c() / run_router_loop() (thermal_router.cpp) call
//     into XRT's real get_info<thermal>() sensor API (the same one
//     xadc/xadc_sensors.cpp uses) and are hardware-gated, unrun.

struct ThermalPolicy {
    float warning_temp_c  = 75.0f;  // log warning, no allocation change
    float throttle_temp_c = 85.0f;  // reduce FPGA allocation by 50%
    float shutdown_temp_c = 95.0f;  // route all work to CPU/GPU
};

class ThermalRouter {
public:
    explicit ThermalRouter(ThermalPolicy policy = {});

    // Pure decision function: given a temperature reading, what FPGA
    // allocation fraction (0.0-1.0) should the router use? No hardware
    // dependency -- this is what makes step 25's "measured response
    // latency" testable without an F1 instance.
    float allocation_fraction_for_temp(float temp_c) const;

    // Read current FPGA temperature via XRT's device sensor API (see
    // thermal_router.cpp / xadc/xadc_sensors.cpp). Hardware-gated.
    float read_fpga_temp_c() const;

    // read_fpga_temp_c() + allocation_fraction_for_temp() in one call --
    // the actual per-poll router step. Hardware-gated.
    float fpga_allocation_fraction() const;

private:
    ThermalPolicy policy_;
};
