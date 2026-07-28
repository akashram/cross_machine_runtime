# chaos

**Status: two of PLAN.md's three named scenarios code-complete AND
actually run locally; the third (GPU node kill -> elastic resharding)
stays a documented gap, same honest status `networking/chaos/` already
carries for network-partition recovery.**

## What this measures

PLAN.md Phase 10 step 6: inject network partition → verify recovery, kill
GPU node → verify elastic resharding, inject FPGA thermal event → verify
router response — all documented with recovery time measurements.

## Design

Reuses this repo's own real components rather than inventing new
fault-injection machinery:

- **Raft leader kill → re-election** (`run_raft_leader_kill_scenario`):
  the same 3-node real loopback-TCP-mesh cluster
  `networking/raft/raft_test.cpp` already builds for correctness testing
  — but this scenario adds the explicit **recovery-time measurement**
  PLAN.md step 6 actually asks for (`raft_test.cpp` only checks the
  re-election happens within a generous timeout bound, doesn't report the
  real wall-clock number as a metric). Also verifies the new leader can
  still commit post-recovery, not just that a leader exists.
- **FPGA thermal event → router response** (`run_fpga_thermal_scenario`):
  the real `ThermalRouter`/`ThermalPolicy` decision logic from
  `fpga_engine/thermal_router` (no hardware dependency — pure function of
  a temperature reading), driven through a synthetic multi-stage trace
  (normal → warning → throttle → shutdown → recovery) checking the
  allocation-fraction sequence is correct at every point, plus a
  monotonic-safety check (allocation never increases as temperature
  rises). Recovery *time* for this scenario is **not re-measured** here —
  `fpga_engine/thermal_router/thermal_router_sim.cpp` already produced
  that real number (an RC step-response thermal model at a 100ms poll
  interval: 10.8ms at throttle, 31.6ms at shutdown) — cited in this
  scenario's output instead of duplicating that simulation.
- **Kill GPU node → elastic resharding**: left as a documented gap. A
  real chaos test for this needs actual GPU hardware running
  `distributed_training/`'s ZeRO/checkpoint-resharding code path under a
  genuine node failure — this repo has never run that on real hardware
  (see that phase's own hardware-gated status), so faking the scenario
  here would prove nothing. Printed explicitly in this suite's own output
  rather than silently omitted.
- **Network partition**: not duplicated here —
  `networking/chaos/`'s `tc netem` scripts already exist for this,
  gated on Linux + a live multi-node cluster (see that directory's own
  README). This suite's output points to it rather than re-stating the
  same gap redundantly.

## Results (captured 2026-07-27, Apple clang 14 / `-std=c++2b`, this Mac)

```
=== chaos scenario: kill the Raft leader ===
PASS  healthy cluster elects exactly one leader before fault injection
PASS  a new leader is elected among survivors after the kill
PASS  the new leader can still commit after recovery
  recovery time (leader stop -> new leader elected): 183.7 ms

=== chaos scenario: inject an FPGA thermal event ===
  temp=60.0C -> allocation=1.00 (expected 1.00) OK
  temp=78.0C -> allocation=1.00 (expected 1.00) OK
  temp=87.0C -> allocation=0.50 (expected 0.50) OK
  temp=97.0C -> allocation=0.00 (expected 0.00) OK
  temp=70.0C -> allocation=1.00 (expected 1.00) OK
PASS  router allocation fraction matches policy at every point in the thermal trace
PASS  allocation never increases as temperature rises (monotonic safety property)
```

## Findings

- Raft leader-kill recovery time (183.7ms) is real wall-clock, dominated
  by `raft.cpp`'s election-timeout randomization range, not network
  latency (loopback TCP on one Mac) — a real multi-node deployment over
  an actual network would add real RTT on top of this, but the
  election-timeout floor is the dominant term either way, matching what
  the Raft paper itself documents as the expected recovery-time driver.
- The thermal scenario's monotonic-safety check is a real, meaningful
  property beyond just "matches the lookup table at 5 points" — it's the
  kind of regression a future `ThermalPolicy` threshold change could
  silently violate (e.g. an off-by-one in threshold comparisons that
  happens to pass the 5 sampled points above but isn't actually
  monotonic everywhere) without this check.

## Hardware notes
- Raft scenario: none — real loopback TCP, no hardware dependency.
- FPGA scenario: none for the decision-logic check run here; the cited
  recovery-time numbers came from real hardware-free simulation
  (`thermal_router_sim.cpp`), not real F1 hardware (that binary itself is
  still hardware-gated for its `read_fpga_temp_c()`/`fpga_allocation_fraction()`
  methods — see `fpga_engine/thermal_router/README.md`).
- GPU node kill / elastic resharding: real GPU hardware required, not
  attempted.
