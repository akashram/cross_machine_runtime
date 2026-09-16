//===- rack_power_model.h - rack-level power/cooling capacity model -----===//
//
// PLAN.md Phase 22 step 5: sum per-device power to rack level, check
// against a rack power budget and cooling capacity (PUE, CRAC/CRAH
// tons-equivalent watts), flag overcommit.
//
// Real typed interface for measured wattage: `Device::reader` is a
// `WattageReader` -- a callable returning `std::optional<double>`. Any
// real telemetry call can be wrapped in one:
//   - gpu_engine/power::PowerMonitor::sample().power_mw / 1000.0
//     (real NVML nvmlDeviceGetPowerUsage(), hardware-gated -- no GPU here)
//   - fpga_engine/xadc's parsed electrical.json rail readings (real XRT
//     get_info<electrical>(), hardware-gated -- no FPGA here)
// Absent a real reader (or if it returns std::nullopt -- e.g. hardware
// present but the sensor call failed), a device falls back to its
// literature-grounded TDP figure. The two paths are represented
// DISTINCTLY (DeviceReading::measured) rather than silently blended, so a
// caller (or this file's own test) can always tell which one it got --
// same "code-complete, hardware-gated, upgrades automatically once
// hardware exists" shape as the rest of this repo, not a permanent
// simulation.
//
// Digital-device TDP constants reused directly from
// analog_engine/energy_model/energy_model.h's `digital_devices()` table
// (itself reused from npu_engine/cost_model/npu_cost_model.cpp's
// TOPS/W denominators), not re-guessed here: CPU (AVX-512 VNNI server)
// 150W, GPU (A100, dense INT8) 400W, NPU (Apple ANE, representative) 2W.
//
//===----------------------------------------------------------------------===//
#pragma once

#include <functional>
#include <optional>
#include <string>
#include <vector>

namespace hpc_cluster {

using WattageReader = std::function<std::optional<double>()>;

struct DeviceReading {
  std::string name;
  double watts;
  bool measured; // true: came from a real reader. false: TDP fallback.
};

struct Device {
  std::string name;
  double tdp_w;         // literature-grounded nameplate TDP (fallback)
  WattageReader reader;  // real measured-wattage source; empty if none

  DeviceReading read() const {
    if (reader) {
      if (auto m = reader()) return {name, *m, true};
    }
    return {name, tdp_w, false};
  }
};

// Same TDP figures as analog_engine/energy_model.h's digital_devices()
// table -- reused, not re-derived, so this model's literature fallback
// never quietly diverges from Phase 15/17's own numbers.
inline Device make_cpu_device(std::string name, WattageReader reader = {}) {
  return {std::move(name), 150.0, std::move(reader)};
}
inline Device make_gpu_a100_device(std::string name, WattageReader reader = {}) {
  return {std::move(name), 400.0, std::move(reader)};
}
inline Device make_npu_device(std::string name, WattageReader reader = {}) {
  return {std::move(name), 2.0, std::move(reader)};
}

struct RackConfig {
  std::string rack_name;
  double power_budget_w;      // rack PDU power budget (IT load)
  double cooling_capacity_w;  // CRAC/CRAH cooling capacity, watts-equivalent
                               // (1 ton of cooling = 3516.85 W; convert before
                               // constructing this struct -- see README)
  double pue;                 // Power Usage Effectiveness: total facility
                               // power = IT power * pue. 1.0 = no overhead
                               // (unrealistic); typical air-cooled ~1.5-1.6,
                               // efficient liquid-cooled ~1.1-1.2 (ASHRAE).
};

struct RackReport {
  std::vector<DeviceReading> devices;
  double total_it_power_w = 0.0;
  double total_facility_power_w = 0.0; // total_it_power_w * pue
  bool power_overcommit = false;   // total_it_power_w > power_budget_w
  bool cooling_overcommit = false; // total_facility_power_w > cooling_capacity_w
  int num_measured = 0;
  int num_fallback = 0;
};

inline RackReport analyze_rack(const std::vector<Device> &devices,
                                const RackConfig &cfg) {
  RackReport r;
  for (const auto &d : devices) {
    DeviceReading reading = d.read();
    r.total_it_power_w += reading.watts;
    if (reading.measured) ++r.num_measured; else ++r.num_fallback;
    r.devices.push_back(std::move(reading));
  }
  r.total_facility_power_w = r.total_it_power_w * cfg.pue;
  r.power_overcommit = r.total_it_power_w > cfg.power_budget_w;
  r.cooling_overcommit = r.total_facility_power_w > cfg.cooling_capacity_w;
  return r;
}

// 1 ton of refrigeration = 3516.85 W (ASHRAE standard conversion).
inline double tons_to_watts(double tons) { return tons * 3516.85; }

} // namespace hpc_cluster
