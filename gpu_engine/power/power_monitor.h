#pragma once
// NVML-based GPU power monitor with background polling thread.
//
// Records power draw (mW), temperature (°C), and throttle reasons at 10ms
// intervals while a workload runs, then produces a PowerReport with:
//   - avg_power_w, peak_power_w, energy_j (integral over time)
//   - max_temp_c, throttled (true if any throttle reason bits were set)
//
// Usage:
//   PowerMonitor pm(0);  // device 0
//   pm.start();
//   run_my_workload();
//   cudaDeviceSynchronize();
//   PowerReport r = pm.stop();
//   pm.print_report(r);
//
// Requires: NVML (libnvidia-ml.so) — always present on GPU instances.
// Link with: -lnvidia-ml

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <chrono>
#include <string>
#include <thread>
#include <vector>

// Forward-declare nvmlDevice_t to avoid including nvml.h in every TU.
// Callers that need full NVML access must include nvml.h themselves.
struct nvmlDevice_st;
using nvmlDevice_t = nvmlDevice_st*;

struct PowerSample {
    uint64_t     timestamp_us; // microseconds since start() call
    unsigned int power_mw;     // milliwatts from nvmlDeviceGetPowerUsage
    unsigned int temp_c;       // Celsius from nvmlDeviceGetTemperature
    uint64_t     throttle_mask; // nvmlDeviceGetCurrentClocksThrottleReasons
};

struct PowerReport {
    std::vector<PowerSample> samples;
    double       avg_power_w;
    double       peak_power_w;
    double       energy_j;          // trapezoidal integral of power×time
    unsigned int max_temp_c;
    bool         throttled;         // any throttle bits set during run
    double       duration_s;
};

class PowerMonitor {
public:
    explicit PowerMonitor(int device_idx = 0)
        : device_idx_(device_idx) {
        init_nvml();
    }

    ~PowerMonitor() {
        if (running_) stop();
        shutdown_nvml();
    }

    void start(int poll_interval_ms = 10) {
        if (running_) return;
        samples_.clear();
        start_time_ = std::chrono::high_resolution_clock::now();
        running_ = true;
        poll_thread_ = std::thread([this, poll_interval_ms]{
            while (running_.load(std::memory_order_relaxed)) {
                auto s = read_sample();
                {
                    std::lock_guard<std::mutex> g(mu_);
                    samples_.push_back(s);
                }
                std::this_thread::sleep_for(
                    std::chrono::milliseconds(poll_interval_ms));
            }
        });
    }

    PowerSample sample() {
        return read_sample();
    }

    PowerReport stop() {
        running_ = false;
        if (poll_thread_.joinable()) poll_thread_.join();

        PowerReport r;
        {
            std::lock_guard<std::mutex> g(mu_);
            r.samples = samples_;
        }

        if (r.samples.empty()) {
            r.avg_power_w = r.peak_power_w = r.energy_j = 0;
            r.max_temp_c  = 0; r.throttled = false; r.duration_s = 0;
            return r;
        }

        double sum_w = 0, peak_w = 0, energy = 0;
        unsigned int max_t = 0;
        bool throttled = false;

        for (size_t i = 0; i < r.samples.size(); ++i) {
            double pw = r.samples[i].power_mw / 1e3;
            sum_w += pw;
            if (pw > peak_w) peak_w = pw;
            if (r.samples[i].temp_c > max_t) max_t = r.samples[i].temp_c;
            if (r.samples[i].throttle_mask) throttled = true;

            // Trapezoidal integration
            if (i > 0) {
                double dt = (r.samples[i].timestamp_us - r.samples[i-1].timestamp_us) / 1e6;
                double pw_prev = r.samples[i-1].power_mw / 1e3;
                energy += 0.5 * (pw + pw_prev) * dt;
            }
        }

        r.avg_power_w  = sum_w / r.samples.size();
        r.peak_power_w = peak_w;
        r.energy_j     = energy;
        r.max_temp_c   = max_t;
        r.throttled    = throttled;
        r.duration_s   = (r.samples.back().timestamp_us) / 1e6;
        return r;
    }

    void print_report(const PowerReport& r) const {
        printf("=== Power Report ===\n");
        printf("  Duration     : %.3f s\n",    r.duration_s);
        printf("  Avg power    : %.1f W\n",    r.avg_power_w);
        printf("  Peak power   : %.1f W\n",    r.peak_power_w);
        printf("  Energy       : %.3f J\n",    r.energy_j);
        printf("  Max temp     : %u °C\n",     r.max_temp_c);
        printf("  Throttled    : %s\n",        r.throttled ? "YES" : "no");
        printf("  Samples      : %zu\n",       r.samples.size());
    }

private:
    int                device_idx_;
    nvmlDevice_t       nvml_device_ = nullptr;
    bool               nvml_ok_     = false;
    std::atomic<bool>  running_{false};
    std::thread        poll_thread_;
    std::mutex         mu_;
    std::vector<PowerSample> samples_;
    std::chrono::high_resolution_clock::time_point start_time_;

    void init_nvml();
    void shutdown_nvml();
    PowerSample read_sample();
};

// -------------------------------------------------------------------------
// Implementation (inline — header-only for single include)
// Requires linking with -lnvidia-ml.
// -------------------------------------------------------------------------

#ifdef POWER_MONITOR_IMPL
#include <nvml.h>

#define NVML_CHECK(call) do { \
    nvmlReturn_t _r = (call); \
    if (_r != NVML_SUCCESS) { \
        fprintf(stderr, "NVML error %s:%d — %s\n", \
                __FILE__, __LINE__, nvmlErrorString(_r)); \
    } \
} while (0)

inline void PowerMonitor::init_nvml() {
    nvmlReturn_t r = nvmlInit();
    if (r != NVML_SUCCESS) {
        fprintf(stderr, "nvmlInit failed: %s\n", nvmlErrorString(r));
        nvml_ok_ = false;
        return;
    }
    r = nvmlDeviceGetHandleByIndex(device_idx_,
            reinterpret_cast<nvmlDevice_t*>(&nvml_device_));
    if (r != NVML_SUCCESS) {
        fprintf(stderr, "nvmlDeviceGetHandleByIndex(%d) failed: %s\n",
                device_idx_, nvmlErrorString(r));
        nvml_ok_ = false;
        return;
    }
    nvml_ok_ = true;
}

inline void PowerMonitor::shutdown_nvml() {
    if (nvml_ok_) nvmlShutdown();
}

inline PowerSample PowerMonitor::read_sample() {
    PowerSample s{};
    auto now = std::chrono::high_resolution_clock::now();
    s.timestamp_us = std::chrono::duration_cast<std::chrono::microseconds>(
                         now - start_time_).count();

    if (!nvml_ok_) return s;

    unsigned int power_mw = 0, temp = 0;
    unsigned long long throttle = 0;

    nvmlDeviceGetPowerUsage(nvml_device_, &power_mw);
    nvmlDeviceGetTemperature(nvml_device_, NVML_TEMPERATURE_GPU, &temp);
    nvmlDeviceGetCurrentClocksThrottleReasons(nvml_device_, &throttle);

    s.power_mw     = power_mw;
    s.temp_c       = temp;
    s.throttle_mask = throttle;
    return s;
}

#else  // stub implementations for compilation without NVML

inline void PowerMonitor::init_nvml() {
    fprintf(stderr, "PowerMonitor: compiled without NVML. "
                    "Define POWER_MONITOR_IMPL and link -lnvidia-ml.\n");
}
inline void PowerMonitor::shutdown_nvml() {}
inline PowerSample PowerMonitor::read_sample() { return {}; }

#endif  // POWER_MONITOR_IMPL
