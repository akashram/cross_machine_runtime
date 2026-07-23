# power_ci

**Status: code complete — requires AWS F1 (Vivado, self-hosted runner) to run.**

## What this builds
Every bitstream built by `tcl_pipeline/synth.tcl` already runs
`report_power` into `reports/power.rpt`. This step turns that raw text
report into something CI can act on:

- `parse_power_report.py` parses Vivado's `report_power` text format
  (total on-chip power, dynamic/static split, junction temp, thermal
  margin, per-component breakdown) into JSON, and exits non-zero if total
  power exceeds a `--budget-w` threshold — the same "gate, don't just
  report" pattern as `tcl_pipeline`'s timing-closure check.
- `.github/workflows/fpga-power-ci.yml` runs the build + power check on a
  self-hosted `f1`-labeled runner and archives both the raw `.rpt` and the
  parsed `.json` as a CI artifact.

The workflow is `workflow_dispatch`-only for now (manually triggered, one
kernel at a time) rather than wired to every push, since no F1 self-hosted
runner exists yet — see the workflow file's header comment for when to flip
that.

## Results
TODO: run on F1 hardware for a real bitstream's power numbers. The parser
itself has been exercised locally (no FPGA/Vivado dependency — it's plain
Python) against a synthetic report shaped like Vivado's documented
`report_power` text format (Xilinx UG907): it correctly extracts the
summary fields and per-component breakdown, and the `--budget-w` gate
correctly passes at a 5.0 W budget and fails (exit 1) at a 3.0 W budget
for a synthetic 4.322 W total. That only proves the text parsing and gate
logic are correct — it says nothing about what a real bitstream draws.

| Kernel | Total on-chip power (W) | Dynamic (W) | Static (W) | Budget (W) | Pass? |
|--------|--------------------------|--------------|------------|------------|-------|
| TODO | TODO | TODO | TODO | TODO | TODO |

## Hardware notes
- Required: F1 instance registered as a self-hosted GitHub Actions runner
  (label `f1`), Vivado 2022.x licensed
- Manual run: `python3 parse_power_report.py <report_power.rpt> --budget-w 10.0`
