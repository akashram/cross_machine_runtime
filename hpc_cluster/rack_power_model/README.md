# rack_power_model -- rack-level power/cooling capacity model

**Status: code-complete AND locally run — pure CPU, no external
dependency.**

## What this measures

PLAN.md Phase 22 step 5: sum per-device power to rack level, check
against a rack power budget and cooling capacity (PUE, CRAC/CRAH
tons-equivalent watts), flag overcommit -- the same shape as
`analog_engine/energy_model`, scaled to rack level.

## Design

- **Real typed interface for measured wattage**: `Device::reader` is a
  `WattageReader` (`std::function<std::optional<double>()>`). Any real
  telemetry source can be wrapped in one -- `gpu_engine/power`'s real
  `PowerMonitor::sample()` (NVML `nvmlDeviceGetPowerUsage`, hardware-gated,
  unrun) or `fpga_engine/xadc`'s parsed `electrical.json` rail readings
  (real XRT `get_info<electrical>()`, hardware-gated, unrun). Absent a
  real reader, or if one returns `std::nullopt` (hardware present but the
  sensor call itself failed), a device falls back to its
  literature-grounded TDP figure -- the two paths are represented
  DISTINCTLY (`DeviceReading::measured`), never silently blended, and
  case 3's test below explicitly checks the model consumes a real
  measured value rather than defaulting to TDP whenever one is available.
  This is the same "code-complete, hardware-gated, upgrades
  automatically once hardware exists" shape as the rest of this repo, not
  a permanent simulation.
- **TDP fallback constants reused, not re-guessed**: CPU 150W, GPU (A100)
  400W, NPU (Apple ANE) 2W -- the exact same figures
  `analog_engine/energy_model.h`'s `digital_devices()` table already
  committed to (itself reused from `npu_engine/cost_model`), so this
  model's fallback numbers never quietly diverge from Phase 15/17's own.
- **Cooling capacity** takes watts directly; `tons_to_watts()` converts
  from tons of refrigeration (1 ton = 3516.85 W, the standard ASHRAE
  conversion) since real facility cooling specs are usually quoted in
  tons, not watts.

## Results (captured 2026-09-12, Apple clang 14, this Mac)

```
PASS  case1: total IT power = 4*400W = 1600W
PASS  case1: facility power = 1600*1.5 = 2400W
PASS  case1: no power overcommit (1600 < 5000)
PASS  case1: no cooling overcommit (2400 < 35168.5)
PASS  case1: all 4 devices used TDP fallback (no real reader supplied)
PASS  case2: total IT power = 20*400W = 8000W
PASS  case2: power overcommit correctly flagged (8000 > 5000)
PASS  case2: cooling overcommit correctly flagged (12000 > 3516.85)
PASS  case3: measured device reports 250W (real reading), not 400W TDP
PASS  case3: fallback device reports 400W TDP (no reader supplied)
PASS  case3: total IT power correctly sums measured + fallback = 250+400=650W
PASS  case3: exactly 1 measured, 1 fallback device counted
PASS  case3: a reader returning nullopt falls back to TDP, not silently zero

ALL PASS
```

## Findings

- **The model correctly consumes a real measured-wattage input when one
  is supplied, not just its literature-default fallback** -- case 3's
  explicit check (PLAN.md step 5's own definition-of-done requirement):
  a device whose real reader returns `250.0` reports exactly `250W` and
  `measured=true`, never silently substituting its `400W` TDP. A second
  device with no reader at all correctly falls back to TDP. A third
  device whose reader returns `std::nullopt` (the "hardware present, but
  the sensor call itself failed" case -- e.g. a real NVML/XRT call that
  errors) ALSO correctly falls back to TDP rather than propagating an
  empty reading as zero watts, which would silently under-report rack
  power.
- **Both overcommit flags (power and cooling) are independently
  triggerable**, not coupled: case 2 tightens the cooling budget
  specifically to confirm `cooling_overcommit` fires on the facility
  power figure (`IT power * PUE`), not the raw IT power figure
  `power_overcommit` checks -- a real rack can be within its power feed's
  budget while still exceeding its cooling system's capacity if PUE is
  high, and the model distinguishes the two failure modes rather than
  reporting one generic "overcommit" bit.

## Hardware notes

Pure CPU, no fab/GPU/FPGA access needed to run this model. The real
measured-wattage inputs it's designed to accept
(`gpu_engine/power::PowerMonitor`, `fpga_engine/xadc`'s parsed readings)
stay hardware-gated exactly as those steps' own READMEs already
document -- this step doesn't change that gap, it just gives the reading
a real place to land once it exists, instead of a permanent
literature-only model.
