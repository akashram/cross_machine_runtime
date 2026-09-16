// rack_power_model_test.cpp -- PLAN.md Phase 22 step 5 self-test.
//
// Three cases:
//  1. A rack of TDP-fallback-only devices under budget -- no overcommit.
//  2. The same rack scaled up past budget -- both power and cooling
//     overcommit correctly flagged.
//  3. A device with a REAL measured-wattage reader whose value DIFFERS
//     from its TDP -- explicit definition-of-done check that the model
//     actually consumes the real reading, not just its literature
//     fallback (PLAN.md step 5's own "checked to actually consume a real
//     measured-wattage input" requirement).
#include "rack_power_model.h"

#include <cstdio>
#include <cstdlib>

using namespace hpc_cluster;

namespace {
int g_failures = 0;

void check(bool cond, const char *what) {
  std::printf("%s  %s\n", cond ? "PASS" : "FAIL", what);
  if (!cond) ++g_failures;
}
} // namespace

int main() {
  // Case 1: 4x A100 GPUs (TDP fallback, 400W each = 1600W IT power),
  // budget 5000W, PUE 1.5 (typical air-cooled per ASHRAE), cooling sized
  // for 10 tons (35168.5 W) -- comfortably under both.
  {
    std::vector<Device> devs;
    for (int i = 0; i < 4; ++i)
      devs.push_back(make_gpu_a100_device("gpu" + std::to_string(i)));
    RackConfig cfg{"rack-a", 5000.0, tons_to_watts(10.0), 1.5};
    RackReport r = analyze_rack(devs, cfg);

    check(r.total_it_power_w == 1600.0, "case1: total IT power = 4*400W = 1600W");
    check(r.total_facility_power_w == 2400.0, "case1: facility power = 1600*1.5 = 2400W");
    check(!r.power_overcommit, "case1: no power overcommit (1600 < 5000)");
    check(!r.cooling_overcommit, "case1: no cooling overcommit (2400 < 35168.5)");
    check(r.num_measured == 0 && r.num_fallback == 4,
          "case1: all 4 devices used TDP fallback (no real reader supplied)");
  }

  // Case 2: same device mix scaled to 20 GPUs (8000W IT power) against
  // the SAME 5000W budget -- power overcommit must trip; cooling budget
  // also tightened to 1 ton (3516.85 W) so facility power (12000W) trips
  // that too.
  {
    std::vector<Device> devs;
    for (int i = 0; i < 20; ++i)
      devs.push_back(make_gpu_a100_device("gpu" + std::to_string(i)));
    RackConfig cfg{"rack-b", 5000.0, tons_to_watts(1.0), 1.5};
    RackReport r = analyze_rack(devs, cfg);

    check(r.total_it_power_w == 8000.0, "case2: total IT power = 20*400W = 8000W");
    check(r.power_overcommit, "case2: power overcommit correctly flagged (8000 > 5000)");
    check(r.cooling_overcommit,
          "case2: cooling overcommit correctly flagged (12000 > 3516.85)");
  }

  // Case 3: definition-of-done check -- a device with a real reader
  // returning a measured value DIFFERENT from its TDP fallback (250W
  // measured vs. 400W TDP, the kind of gap real workload-dependent power
  // draw produces). The model must report the MEASURED value and mark it
  // measured=true, not silently fall back to TDP.
  {
    Device measured_gpu = make_gpu_a100_device(
        "gpu-measured", []() -> std::optional<double> { return 250.0; });
    Device fallback_gpu = make_gpu_a100_device("gpu-fallback"); // no reader

    std::vector<Device> devs{measured_gpu, fallback_gpu};
    RackConfig cfg{"rack-c", 5000.0, tons_to_watts(10.0), 1.4};
    RackReport r = analyze_rack(devs, cfg);

    check(r.devices[0].measured && r.devices[0].watts == 250.0,
          "case3: measured device reports 250W (real reading), not 400W TDP");
    check(!r.devices[1].measured && r.devices[1].watts == 400.0,
          "case3: fallback device reports 400W TDP (no reader supplied)");
    check(r.total_it_power_w == 650.0,
          "case3: total IT power correctly sums measured + fallback = 250+400=650W");
    check(r.num_measured == 1 && r.num_fallback == 1,
          "case3: exactly 1 measured, 1 fallback device counted");

    // A reader that returns nullopt (hardware present but sensor call
    // failed) must fall back to TDP too, not propagate the empty optional.
    Device failed_reader_gpu = make_gpu_a100_device(
        "gpu-failed-sensor", []() -> std::optional<double> { return std::nullopt; });
    DeviceReading fr = failed_reader_gpu.read();
    check(!fr.measured && fr.watts == 400.0,
          "case3: a reader returning nullopt falls back to TDP, not silently zero");
  }

  std::printf(g_failures == 0 ? "\nALL PASS\n" : "\n%d FAILURES\n", g_failures);
  return g_failures == 0 ? 0 : 1;
}
