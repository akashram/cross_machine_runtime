#pragma once
// TODO: implement on F1 with XADC temperature monitoring
// Reads FPGA die temperature via XADC, reduces FPGA workload when above threshold.

struct ThermalPolicy {
    float warning_temp_c  = 75.0f;  // log warning
    float throttle_temp_c = 85.0f;  // reduce FPGA allocation by 50%
    float shutdown_temp_c = 95.0f;  // route all work to CPU/GPU
};

class ThermalRouter {
public:
    explicit ThermalRouter(ThermalPolicy policy = {});

    // Read current FPGA temperature via XADC (Linux: /sys/bus/iio/... or xclmgmt)
    float read_fpga_temp_c() const;

    // Returns FPGA allocation fraction (0.0–1.0) based on current temperature.
    float fpga_allocation_fraction() const;
};
