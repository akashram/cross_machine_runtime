#pragma once
#include <vector>
#include <cstdint>
// TODO: implement on GPU hardware with NVML

struct PowerSample {
    uint64_t timestamp_us;  // CLOCK_MONOTONIC
    unsigned int power_mw;  // milliwatts from nvmlDeviceGetPowerUsage
    unsigned int temp_c;    // Celsius from nvmlDeviceGetTemperature
    uint64_t throttle_mask; // nvmlDeviceGetCurrentClocksThrottleReasons
};

struct PowerReport {
    std::vector<PowerSample> samples;
    double avg_power_w;
    double peak_power_w;
    unsigned int max_temp_c;
    bool throttled;
    double energy_j;  // integral of power over duration
};

class PowerMonitor {
public:
    explicit PowerMonitor(int device_idx = 0);
    ~PowerMonitor();

    void start();               // begin background polling (10ms interval)
    PowerSample sample();       // single synchronous sample
    PowerReport stop();         // stop polling, return aggregated report

private:
    int      device_idx_;
    void*    nvml_device_ = nullptr;
    bool     running_     = false;
    // TODO: background thread, sample buffer
};
