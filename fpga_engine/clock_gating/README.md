# clock_gating — conditional clock enables and dynamic power reduction

**Status: hardware gated-clock synthesis is code complete and unrun (F1
required); the duty-cycle model and delta-parsing logic are measured
locally.**

## What this measures
Three things, deliberately kept separate since only one needs an F1 instance:

1. **Duty-cycle model** (`clock_gating_model.cpp`): clock gating only
   saves power on cycles where the gated logic wouldn't have changed
   value anyway — this model predicts the dynamic power reduction as a
   function of how often that's true (duty cycle), including the fixed
   overhead of the gating logic itself (BUFGCE + enable tree), and finds
   the break-even duty cycle above which gating is a net loss. Compiles
   and runs on this Mac today — no Vivado needed.
2. **Source-level fix** (`gated_mlp.cpp`): an `ml_kernel_mlp` variant with
   an explicit `valid` input gating every register write, giving the
   `hidden[]`/`out[]` register banks the common enable Vivado's
   `-gated_clock_conversion` is documented to detect and promote to a
   real gated clock. Standalone — not folded into step 14's
   `ml_kernel.cpp`, same reasoning as `timing_closure/pipelined_tree_reduce.cpp`:
   whether this is worth adding depends on the traffic pattern (see
   model), it isn't a strict improvement.
3. **Delta parsing** (`power_delta.py`): parses an ungated/gated
   `report_power` pair and computes the measured reduction, so it can be
   compared against the model's prediction once real numbers exist.
   Plain Python, no FPGA dependency — its self-test (below) is real,
   passing output today.

## Model caveats
`kGatingOverheadFrac` is a first-order engineering approximation (BUFGCE
+ enable-tree power as a fraction of ungated dynamic power), not a
datasheet quote — the real number depends on how much logic shares the
gated enable and how it's placed, unknown without real synthesis. What
doesn't depend on getting that constant exactly right: reduction is
structurally negative as duty cycle approaches 100% (gating something
that's always active is pure overhead) and grows toward `(1 - overhead)`
as duty cycle approaches 0.

## Results
**Duty-cycle model** (measured locally,
`clang++ -O2 -std=c++17 clock_gating_model.cpp -o clock_gating_model && ./clock_gating_model`):

```
=== dynamic power reduction vs. duty cycle (gating overhead = 2%) ===
duty=100.0% | gated/ungated dynamic power = 1.020 | reduction =   -2.0% (NET LOSS — do not gate)
duty= 50.0% | gated/ungated dynamic power = 0.520 | reduction =  +48.0%
duty= 25.0% | gated/ungated dynamic power = 0.270 | reduction =  +73.0%
duty= 10.0% | gated/ungated dynamic power = 0.120 | reduction =  +88.0%
duty=  5.0% | gated/ungated dynamic power = 0.070 | reduction =  +93.0%
duty=  1.0% | gated/ungated dynamic power = 0.030 | reduction =  +97.0%

break-even duty cycle = 98.0% — gate only if the design's expected duty cycle is below this; above it, gating overhead exceeds its own savings.

=== applied to gated_mlp.cpp (ml_kernel_mlp + valid-gated registers) ===
Bursty inference (valid asserted ~1/10 cycles, duty=10%): predicted reduction = 88.0% — gating is worth adding.
Saturated streaming (valid asserted every cycle, duty=100%): predicted reduction = -2.0% — gating is a net loss; do not add it to a design run this way.
```

**Delta-parsing self-test** (measured locally, `python3 power_delta.py --self-test`,
two synthetic `report_power`-shaped reports, gated dynamic power chosen to
land at the model's own 10%-duty-cycle prediction so the two independently
written pieces can be checked against each other):

```
power_delta._self_test: OK — synthetic reduction = 88.0% (expected 88.0%)
```

That 88.0% synthetic-vs-model agreement is by construction (the synthetic
gated report's dynamic power was picked to hit it), not evidence the
model is accurate — it only proves the parser's delta math is correct.
The real test is comparing `power_delta.py`'s output against
`clock_gating_model.cpp`'s prediction once `power_gating.tcl` has run on
F1 at a real, non-synthetic duty cycle.

**Hardware** — TODO: run `power_gating.tcl` against `gated_mlp.cpp` at a
chosen `-duty_cycle`, then `power_delta.py reports/power_ungated.rpt
reports/power_gated.rpt --predicted-pct <model's number at that duty cycle>`:

| Duty cycle | Predicted reduction | Measured reduction | Diff |
|---|---|---|---|
| TODO | TODO | TODO | TODO |

## Files
- `clock_gating_model.cpp` — portable, no Vivado dependency; run it directly.
- `gated_mlp.cpp` — the HLS-side fix (unrun). Deliberately standalone,
  not folded into `ml_kernel.cpp` — see the file header for why.
- `power_gating.tcl` — the Vivado-side flow: builds `gated_mlp.cpp` with
  and without `-gated_clock_conversion`, annotates the `valid` port's
  switching activity to match a chosen duty cycle via
  `set_switching_activity`, and runs `report_power` for both.
- `power_delta.py` — portable, no Vivado dependency; parses the two
  `report_power` outputs and computes the measured reduction. Duplicates
  (rather than imports) `power_ci/parse_power_report.py`'s parsing
  approach so this step stays runnable standalone.

## Hardware notes
- Required: AWS F1, Vitis HLS + Vivado 2022.x
- Run: `vivado -mode batch -source power_gating.tcl -tclargs -top gated_mlp -ip_dir <dir> -xdc <constraints> -duty_cycle 0.10 -outdir <dir>`
