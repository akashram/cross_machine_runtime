# thermal_router — FPGA temperature -> workload allocation, with measured response latency

**Status: the XRT-backed sensor read is code complete and unrun (F1
required); the router's decision logic and response-latency measurement
are real, locally-run code — not a stub, not a one-shot report parse.**

## What this measures
Three things, split the same way `xadc/` was (its own header explicitly
deferred this step's work: "Decide a workload-allocation fraction from
the temperature. That's thermal_router's ThermalPolicy" / "Measure
response latency... needs a real running router loop"):

1. **Decision logic** (`thermal_policy.cpp`): `ThermalRouter::
   allocation_fraction_for_temp()` — pure arithmetic mapping a
   temperature reading to an allocation fraction (1.0 / 0.5 / 0.0 per
   `ThermalPolicy`'s warning/throttle/shutdown thresholds). No hardware
   dependency, compiles and runs on this Mac today.
2. **Hardware sensor read** (`thermal_router.cpp`): `read_fpga_temp_c()`
   and `fpga_allocation_fraction()`, using the same XRT
   `get_info<xrt::info::device::thermal>()` API `xadc/xadc_sensors.cpp`
   uses, so the router's real production path and the locally-tested
   path in (3) make the identical decision for the same temperature —
   they share `thermal_policy.cpp`, not two independently-written
   copies of the threshold logic. Hardware-gated, unrun.
3. **Response-latency measurement** (`thermal_router_sim.cpp`): drives
   `allocation_fraction_for_temp()` against a synthetic FPGA die
   temperature trace — a first-order RC thermal step response, the
   standard shape for silicon warming under a sudden sustained load
   increase — sampled at a fixed polling interval, the same
   discretization a real host-side monitoring loop uses. This is what
   makes "measured response latency" (PLAN.md's own wording for this
   step) an actual measurement rather than a TODO: it's real code,
   actually run, actually timed with `std::chrono`, not a hand-computed
   estimate.

## Model caveats
`kTauSeconds` (thermal time constant) and `kPollIntervalMs` (host
monitoring cadence) are commonly-cited order-of-magnitude figures — large
silicon + heatsink mass gives FPGA die thermal step responses on the
order of seconds to tens of seconds; a few-hundred-ms polling cadence is
a typical monitoring-loop tradeoff between reaction time and PCIe/host
overhead — not datasheet numbers for the AWS F1 VU9P shell specifically,
same caveat style as every other `*_model.cpp` in `fpga_engine/`. What
doesn't depend on getting them exactly right: response latency to a
threshold crossing is always bounded by the polling interval, because the
router literally cannot see the temperature between polls — regardless
of the true thermal time constant.

## Results
**Response-latency measurement** (measured locally,
`clang++ -O2 -std=c++17 -Wall thermal_router_sim.cpp thermal_policy.cpp -o thermal_router_sim && ./thermal_router_sim`):

```
=== thermal_router_sim: synthetic FPGA thermal event (RC step response) ===
ambient=45C steady-state=100C tau=15s poll-interval=100ms

decision-compute latency: 4.29ns/call (1000000 calls, sink=1.0)

throttle threshold (85C, allocation 1.0->0.5): true crossing t=19.49s, router reacted t=19.50s, response latency=10.8ms (poll-interval bound=100ms)
shutdown threshold (95C, allocation 0.5->0.0): true crossing t=35.97s, router reacted t=36.00s, response latency=31.6ms (poll-interval bound=100ms)

response latency at both thresholds falls within the 100ms poll-interval bound, as it must -- the router cannot react faster than it can see a new reading. Decision-compute latency (4.29ns) is ~2.3e+07x smaller than the poll interval, confirming the bottleneck is polling cadence, not the router's own logic; a tighter response-latency budget is a poll-interval tuning question (more frequent XRT sensor reads), not a router-logic optimization.
```

Both crossings land well inside the 100ms bound (10.8ms and 31.6ms) —
which makes sense given the polling loop samples every 100ms starting
from t=0 and the true crossing times (19.49s, 35.97s) don't happen to
land right after a poll boundary; a different `kPollIntervalMs` phase
offset would move these numbers within [0, 100]ms without changing the
finding. The real result is the ~7 orders of magnitude gap between
decision-compute latency and poll interval: if response time needs to
improve, the lever is polling frequency, not the router's own logic,
which is already effectively free.

**Hardware** — TODO: run `thermal_router.cpp` against a real F1 XRT
device and fill in:

| Metric | Value |
|---|---|
| `read_fpga_temp_c()` wall-clock latency (XRT round-trip) | TODO |
| Real thermal event response latency (induced load, measured `xbutil examine` polling) | TODO |
| Confirmed `thermal.json` field name/schema | TODO |

## Files
- `thermal_router.h` — shared interface: `ThermalPolicy`, `ThermalRouter`.
- `thermal_policy.cpp` — portable decision logic; run it via
  `thermal_router_sim.cpp`, no XRT dependency.
- `thermal_router.cpp` — hardware-gated XRT sensor read. Requires XRT
  headers to compile; not part of the local build.
- `thermal_router_sim.cpp` — portable response-latency harness; run it
  directly (see command above).

## Hardware notes
- Required for `thermal_router.cpp`: AWS F1, XRT runtime, enumerated
  FPGA device.
- Once run: induce a real thermal event (sustained kernel load) and time
  `fpga_allocation_fraction()`'s reaction against `xbutil examine`'s own
  polling, to check `thermal_router_sim.cpp`'s poll-interval-bound
  prediction against a real measurement.

---

**Phase 7 (FPGA Backend) is now 25/25 — code-complete.** See
`fpga_engine/README.md` and `CLAUDE.md` for the phase-level summary and
which steps' hardware halves are still TODO pending F1 access.
