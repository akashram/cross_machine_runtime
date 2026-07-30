# thermal — NPU power/thermal modeling

**Status: code-complete AND locally run (pure decision logic + synthetic-
event simulation, no hardware dependency); the real SMC sensor read is
code-complete and unrun (no stable public per-ANE-block sensor key, no
IOKit access from this environment).**

## What this measures
PLAN.md Phase 15 step 5: "mirrors `fpga_engine/thermal_router`'s real
thermal-response model; NPUs are power-efficiency-first hardware,
on-theme for this pattern."

## Design
Direct structural mirror of `fpga_engine/thermal_router`'s two-file split:
- `npu_thermal_policy.h`/`.cpp` — `NpuThermalRouter::allocation_fraction_for_temp()`,
  pure temperature -> allocation-fraction decision logic (1.0/0.5/0.0 at
  warning/throttle/shutdown thresholds). No hardware dependency.
- `npu_thermal_router.cpp` — the hardware-touching `read_npu_temp_c()`,
  which would go through IOKit/SMC sensor keys on Apple Silicon.
  Hardware-gated and unrun — and genuinely so, not just "no toolchain
  installed": there is no stable PUBLIC per-ANE-block SMC key the way
  FPGA's XADC exposes a documented per-die-region sensor, so even with
  IOKit access this would need reverse-engineered, chip-generation-
  specific keys (see the file's header comment).
- `npu_thermal_sim.cpp` — the portable half, run locally: drives the
  router against a synthetic NPU/SoC thermal event (first-order RC step
  response) and measures real response latency with `std::chrono`, same
  methodology as `thermal_router_sim.cpp`. NPU-specific constants:
  smaller thermal time constant (tau=4s vs FPGA's 15s — an integrated
  mobile SoC has far less thermal mass than a discrete card + heatsink)
  and a tighter poll interval (25ms vs 100ms).

## Results (captured 2026-07-30, `clang++ -O2 -std=c++17
npu_thermal_sim.cpp npu_thermal_policy.cpp -o npu_thermal_sim &&
./npu_thermal_sim`, this Mac)

```
=== npu_thermal_sim: synthetic NPU/SoC thermal event (RC step response) ===
ambient=35C steady-state=92C tau=4s poll-interval=25ms

decision-compute latency: 2.49ns/call (1000000 calls, sink=1.0)

throttle threshold (75C, allocation 1.0->0.5): true crossing t=4.839s, router reacted t=4.850s, response latency=10.65ms (poll-interval bound=25ms)
shutdown threshold (85C, allocation 0.5->0.0): true crossing t=8.389s, router reacted t=8.400s, response latency=11.44ms (poll-interval bound=25ms)
```

## Findings
- Same structural finding as `fpga_engine/thermal_router`'s: response
  latency is bounded by polling cadence (both crossings land well within
  the 25ms poll-interval bound), not by the router's own decision logic
  (2.49ns/call, ~7 orders of magnitude smaller than the poll interval).
- The NPU's much smaller `tau` means it crosses both thresholds far
  sooner in absolute simulated time than the FPGA model does for an
  equivalent steady-state overshoot — a real, structural consequence of
  an integrated SoC's lower thermal mass, captured purely through the
  `tau` constant without changing any router logic. This is the kind of
  device-specific behavior this model is meant to surface: same decision
  function, different physical response time.
- Real platform limitation surfaced while writing `npu_thermal_router.cpp`:
  unlike FPGA's XADC (a documented, stable per-die-region sensor API),
  there is no equivalently stable *public* per-ANE-block thermal sensor
  on Apple Silicon — SMC exposes overall SoC/package temperature via
  community-reverse-engineered keys that vary by chip generation. This is
  a genuine, disclosed gap in NPU thermal observability compared to
  FPGA's XADC, not a missing-toolchain excuse.

## Platform notes
No CMake target — manually built and run per the command above, same
convention as `fpga_engine/thermal_router/thermal_router_sim.cpp`.
